#include "control.h"
#include "numeric_typedefs.h"
#include "error_message.h"
#include "stats_display.h"
#include "stringutil.h"

#include <windows.h>

#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>

static constexpr u8 sections = 3,titleW = 9,columns = 5;
static constexpr u16 counterOffset = (sections + 3) * 2;
static constexpr TCHAR titles[sections + 1][titleW + 1] =
{
    TEXT("wait genS"),
    TEXT("start GPU"),
    TEXT("IO thread"),
    TEXT("  total  ")
};

static HWND window;
static HANDLE stdOut,stdIn;
static WORD defaultAttribute;
static DWORD restoreOutMode,restoreInMode,dummy;
static CONSOLE_SCREEN_BUFFER_INFO bufferInfo;

static constexpr CONSOLE_CURSOR_INFO invisibleCursor {1,false};
static constexpr WORD highlightAttribute =
    BACKGROUND_BLUE | BACKGROUND_GREEN | BACKGROUND_RED | BACKGROUND_INTENSITY;

static u16 width,colW;
static u8 selected;

static std::chrono::nanoseconds profilingData[sections + 1][4] {};

static u64 iteration = 0;
void setIteration(const u64 itr) {iteration = itr;}
static void printIteration()
{
    const str string = TO_STRING(iteration);
    CHECK_WINAPI(WriteConsoleOutputCharacter(
        stdOut,
        string.c_str(),
        (DWORD)string.length(),
        {9,counterOffset + 3},
        &dummy
    ));
}

void initStats()
{
    // set profiling data minimums
    for(u8 i = 0;i < sections + 1;++i)
        profilingData[i][0] = std::chrono::nanoseconds(~0Ui64 >> 1);
    
    // get handles
    stdOut = CreateConsoleScreenBuffer
    (
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        CONSOLE_TEXTMODE_BUFFER,
        NULL
    );
    CHECK_WINAPI(stdOut != INVALID_HANDLE_VALUE);
    CHECK_WINAPI(SetConsoleActiveScreenBuffer(stdOut));
    CHECK_WINAPI((stdIn = GetStdHandle(STD_INPUT_HANDLE)) != INVALID_HANDLE_VALUE);
    CHECK_WINAPI((window = GetConsoleWindow()) != NULL);
    CHECK_WINAPI(ShowWindow(window,SW_MAXIMIZE));

    // get initial conditions
    CHECK_WINAPI(GetConsoleScreenBufferInfo(stdOut,&bufferInfo));
    CHECK_WINAPI(GetConsoleMode(stdOut,&restoreOutMode));
    CHECK_WINAPI(GetConsoleMode(stdIn,&restoreInMode));

    // cleanup on exit
    std::atexit([]()
    {
        CloseHandle(stdOut);
        SetConsoleMode(stdIn,restoreInMode);
    });

    // disable user input
    CHECK_WINAPI(SetConsoleCursorInfo(stdOut,&invisibleCursor));
    CHECK_WINAPI(SetConsoleMode(stdOut,restoreOutMode & ~ENABLE_WRAP_AT_EOL_OUTPUT));
    CHECK_WINAPI(SetConsoleMode
    (
        stdIn,
        restoreInMode & ~
        (
            ENABLE_ECHO_INPUT |
            ENABLE_INSERT_MODE |
            ENABLE_LINE_INPUT |
            ENABLE_MOUSE_INPUT |
            ENABLE_QUICK_EDIT_MODE |
            ENABLE_WINDOW_INPUT
        )
    ));

    // get console dimensions
    width = bufferInfo.dwSize.X;
    defaultAttribute = bufferInfo.wAttributes;
    colW = (width - titleW - columns - 2) / columns;

    // divider positions
    const u8 div[]
    {
        1,
        (u8)(2 + titleW),
        (u8)(3 + titleW + colW),
        (u8)(4 + titleW + 2 * colW),
        (u8)(5 + titleW + 3 * colW),
        (u8)(6 + titleW + 4 * colW)
    };

    u8 line = 0;

    // profiler header
    for(const u8 d : div)
        CHECK_WINAPI(FillConsoleOutputCharacter(stdOut,TEXT('|'),1,{d,line},&dummy));
    CHECK_WINAPI(WriteConsoleOutputCharacter
    (
        stdOut,
        TEXT("lo"),
        2,
        {(i16)(3 + titleW + ((colW - 2) / 2)),line},
        &dummy
    ));
    CHECK_WINAPI(WriteConsoleOutputCharacter
    (
        stdOut,
        TEXT("hi"),
        2,
        {(i16)(4 + titleW + colW + ((colW - 2) / 2)),line},
        &dummy
    ));
    CHECK_WINAPI(WriteConsoleOutputCharacter
    (
        stdOut,
        TEXT("avg"),
        3,
        {(i16)(5 + titleW + 2 * colW + ((colW - 3) / 2)),line},
        &dummy
    ));
    CHECK_WINAPI(WriteConsoleOutputCharacter
    (
        stdOut,
        TEXT("cur"),
        3,
        {(i16)(6 + titleW + 3 * colW + ((colW - 3) / 2)),line},
        &dummy
    ));
    CHECK_WINAPI(WriteConsoleOutputCharacter
    (
        stdOut,
        TEXT("total"),
        5,
        {(i16)(7 + titleW + 4 * colW + ((colW - 5) / 2)),line++},
        &dummy
    ));

    // separator
    CHECK_WINAPI(FillConsoleOutputCharacter(stdOut,TEXT('='),width,{0,line++},&dummy));

    // profiler rows
    for(u8 s = 0;s < sections + 1;++s)
    {
        if(s)
        {
            CHECK_WINAPI(FillConsoleOutputCharacter(stdOut,TEXT('-'),width,{0,line},&dummy));
            for(const u8 d : div)
                CHECK_WINAPI(FillConsoleOutputCharacter(stdOut,TEXT('+'),1,{d,line},&dummy));
            ++line;
        }

        CHECK_WINAPI(FillConsoleOutputCharacter(stdOut,TEXT(' '),width,{0,line},&dummy));
        for(const u8 d : div)
            CHECK_WINAPI(FillConsoleOutputCharacter(stdOut,TEXT('|'),1,{d,line},&dummy));
        CHECK_WINAPI(WriteConsoleOutputCharacter(stdOut,titles[s],titleW,{2,line++},&dummy));
    }

    // separator
    CHECK_WINAPI(FillConsoleOutputCharacter(stdOut,TEXT('='),width,{0,line++},&dummy));

    // stats header
    CHECK_WINAPI(WriteConsoleOutputCharacter(stdOut,TEXT("Stats:"),6,{0,line++},&dummy));

    // separator
    CHECK_WINAPI(FillConsoleOutputCharacter(stdOut,TEXT('-'),width,{0,line++},&dummy));

    // total count
    CHECK_WINAPI(WriteConsoleOutputCharacter(stdOut,TEXT("  Total: 0"),10,{0,line++},&dummy));

    // current count
    CHECK_WINAPI(WriteConsoleOutputCharacter(stdOut,TEXT("Current: 0"),10,{0,line++},&dummy));

    // total count
    CHECK_WINAPI(WriteConsoleOutputCharacter(stdOut,TEXT("Highest: 0"),10,{0,line++},&dummy));

    // iterations
    CHECK_WINAPI(WriteConsoleOutputCharacter(stdOut,TEXT("   Iter: 0"),10,{0,line++},&dummy));

    // uptime
    CHECK_WINAPI(WriteConsoleOutputCharacter(stdOut,TEXT(" uptime: 0"),10,{0,line},&dummy));

    // select first section
    CHECK_WINAPI(FillConsoleOutputCharacter(stdOut,TEXT('>'),1,{0,2},&dummy));
    CHECK_WINAPI(FillConsoleOutputAttribute(stdOut,highlightAttribute,titleW,{2,2},&dummy));
    selected = 0;

    // print iteration if recovery file was loaded
    printIteration();
}

static str fmtTime(const std::chrono::nanoseconds time)
{
    sstream output;
    if(time.count() > std::nano::den * std::chrono::minutes::period::num)
    {
        // format H+:MM:SS
        const std::chrono::hh_mm_ss fmt(time);
        output << fmt.hours().count()
               << TEXT(':')
               << std::setw(2) << std::setfill(TEXT('0')) << fmt.minutes().count()
               << TEXT(':')
               << std::setw(2) << std::setfill(TEXT('0')) << fmt.seconds().count();
    }
    else
    {
        if(time.count() > std::nano::den)
            // format s
            output << (time.count() / std::nano::den)
                   << TEXT('s');
        else if(time.count() > std::nano::den / std::milli::den)
            // format ms
            output << (time.count() / (std::nano::den / std::milli::den))
                   << TEXT("ms");
        else if(time.count() > std::nano::den / std::micro::den)
            // format µs
            output << (time.count() / (std::nano::den / std::micro::den))
                   << TEXT("µs");
        else
            // formate ns
            output << time.count()
                   << TEXT("ns");
    }
    return output.str();
}

void nextSection(const std::chrono::nanoseconds time)
{
    std::chrono::nanoseconds (&section)[4] = profilingData[selected];

    const short row = (selected + 1) * 2;
    CHECK_WINAPI(FillConsoleOutputAttribute(stdOut,defaultAttribute,titleW,{2,row},&dummy));
 
    // lo
    if(time < section[0])
    {
        const str lo = fmtTime(section[0] = time);
        CHECK_WINAPI(FillConsoleOutputCharacter
        (
            stdOut,
            TEXT(' '),
            colW,
            {(i16)(3 + titleW),row},
            &dummy
        ));
        CHECK_WINAPI(WriteConsoleOutputCharacter
        (
            stdOut,
            lo.c_str(),
            (DWORD)lo.length(),
            {(i16)(3 + titleW + (colW - lo.length()) / 2),row},
            &dummy
        ));
    }
    
    // hi
    if(time > section[1])
    {
        const str hi = fmtTime(section[1] = time);
        CHECK_WINAPI(FillConsoleOutputCharacter
        (
            stdOut,
            TEXT(' '),
            colW,
            {(i16)(4 + titleW + colW),row},
            &dummy
        ));
        CHECK_WINAPI(WriteConsoleOutputCharacter
        (
            stdOut,
            hi.c_str(),
            (DWORD)hi.length(),
            {(i16)(4 + titleW + colW + (colW - hi.length()) / 2),row},
            &dummy
        ));
    }
    
    // avg
    {
        const str avg = fmtTime((section[2] += time) / (iteration + 1));
        CHECK_WINAPI(FillConsoleOutputCharacter
        (
            stdOut,
            TEXT(' '),
            colW,
            {(i16)(5 + titleW + 2 * colW),row},
            &dummy
        ));
        CHECK_WINAPI(WriteConsoleOutputCharacter
        (
            stdOut,
            avg.c_str(),
            (DWORD)avg.length(),
            {(i16)(5 + titleW + 2 * colW + (colW - avg.length()) / 2),row},
            &dummy
        ));
    }
    
    // cur
    {
        const str cur = fmtTime(time);
        CHECK_WINAPI(FillConsoleOutputCharacter
        (
            stdOut,
            TEXT(' '),
            colW,
            {(i16)(6 + titleW + 3 * colW),row},
            &dummy
        ));
        CHECK_WINAPI(WriteConsoleOutputCharacter
        (
            stdOut,
            cur.c_str(),
            (DWORD)cur.length(),
            {(i16)(6 + titleW + 3 * colW + (colW - cur.length()) / 2),row},
            &dummy
        ));
    }

    // total
    {
        const str total = fmtTime(section[3] += time);
        CHECK_WINAPI(FillConsoleOutputCharacter
        (
            stdOut,
            TEXT(' '),
            colW,
            {(i16)(7 + titleW + 4 * colW),row},
            &dummy
        ));
        CHECK_WINAPI(WriteConsoleOutputCharacter
        (
            stdOut,
            total.c_str(),
            (DWORD)total.length(),
            {(i16)(7 + titleW + 4 * colW + (colW - total.length()) / 2),row},
            &dummy
        ));
    }
    
    i16 nextRow;
    if(selected == sections)
    {
        for(u8 r = 4;r <= row;r += 2)
            CHECK_WINAPI(FillConsoleOutputCharacter(stdOut,TEXT(' '),1,{0,r},&dummy));
        nextRow = 2;
        selected = 0;
        printIteration();
        ++iteration;
    }
    else
    {
        CHECK_WINAPI(FillConsoleOutputCharacter(stdOut,TEXT('|'),1,{0,row},&dummy));
        nextRow = row + 2;
        ++selected;
    }
    CHECK_WINAPI(FillConsoleOutputCharacter(stdOut,TEXT('>'),1,{0,nextRow},&dummy));
    CHECK_WINAPI(FillConsoleOutputAttribute(stdOut,highlightAttribute,titleW,{2,nextRow},&dummy));
}

void updateCounters(const u64 current)
{
    static u64 total = 0,highest = 0;
    
    const str totalStr = TO_STRING(total += current);
    CHECK_WINAPI(WriteConsoleOutputCharacter(
        stdOut,
        totalStr.c_str(),
        (DWORD)totalStr.length(),
        {9,counterOffset},
        &dummy
    ));

    const str currentStr = TO_STRING(current);
    CHECK_WINAPI(FillConsoleOutputCharacter
    (
        stdOut,
        TEXT(' '),
        width - 9,
        {9,counterOffset + 1},
        &dummy
    ));
    CHECK_WINAPI(WriteConsoleOutputCharacter(
        stdOut,
        currentStr.c_str(),
        (DWORD)currentStr.length(),
        {9,counterOffset + 1},
        &dummy
    ));

    if(highest < current)
    {
        highest = current;
        CHECK_WINAPI(WriteConsoleOutputCharacter(
            stdOut,
            currentStr.c_str(),
            (DWORD)currentStr.length(),
            {9,counterOffset + 2},
            &dummy
        ));
    }
}

void updateDuration()
{
    static const std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    const std::chrono::hh_mm_ss duration(std::chrono::steady_clock::now() - startTime);
    const str string =
    (
        sstream()
            << duration.hours().count()
            << TEXT(':')
            << std::setw(2) << std::setfill(TEXT('0'))<< duration.minutes().count()
            << TEXT(':')
            << std::setw(2) << std::setfill(TEXT('0')) << duration.seconds().count()
    ).str();
    CHECK_WINAPI(WriteConsoleOutputCharacter(
        stdOut,
        string.c_str(),
        (DWORD)string.length(),
        {9,counterOffset + 4},
        &dummy
    ));
}

void setSubstatus(const char *status)
{
    const str string
    #ifdef UNICODE
        = ((sstream)(sstream() << status)).str();
    #else
        (status);
    #endif
    CHECK_WINAPI(FillConsoleOutputCharacter
    (
        stdOut,
        TEXT(' '),
        width,
        {0,counterOffset + 5},
        &dummy
    ));
    CHECK_WINAPI(WriteConsoleOutputCharacter(
        stdOut,
        string.c_str(),
        (DWORD)string.length(),
        {9,counterOffset + 5},
        &dummy
    ));
}