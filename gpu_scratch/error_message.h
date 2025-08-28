#pragma once
#include "control.h"

#ifdef SHOW_ERROR
#include "stringutil.h"
#include "numeric_typedefs.h"
#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <windows.h>

static void __check_winapi(BOOL flag,const char *file,const u32 line)
{
    const DWORD errorMessageID = GetLastError();
    if(errorMessageID == ERROR_SUCCESS && flag) return;
    
    LPTSTR messageBuffer = nullptr;
    const size_t size = FormatMessage
    (
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        0,
        errorMessageID,
        MAKELANGID(LANG_NEUTRAL,SUBLANG_DEFAULT),
        (LPTSTR)&messageBuffer,
        0,
        0
    );
    sstream stream = sstream() << file << TEXT("\nLine: ") << line << TEXT('\n');
    stream.write(messageBuffer,size);
    const str msg = stream.str();
    MessageBox(NULL,msg.c_str(),TEXT("WINDOWS ERROR"),MB_OK);
    LocalFree(messageBuffer);
    
    throw std::logic_error(TO_CHAR8(msg));
}
#define CHECK_WINAPI(x) __check_winapi(x,__FILE__,__LINE__)
static void __chk_cuda(const cudaError_t e,const char *file,const int line)
{
    if(e != cudaSuccess)
    {
        const str msg =
        (
            sstream()
                << file
                << TEXT("\nLine: ")
                << line
                << TEXT('\n')
                << cudaGetErrorName(e)
                << TEXT(": ")
                << cudaGetErrorString(e)
        ).str();
        MessageBox(NULL,msg.c_str(),TEXT("CUDA ERROR"),MB_OK);
        throw std::logic_error(TO_CHAR8(msg));
    }
}
#define CHECK_CUDA(e) __chk_cuda(e,__FILE__,__LINE__)
#else
#define CHECK_WINAPI(e) e
#define CHECK_CUDA(e) e
#endif