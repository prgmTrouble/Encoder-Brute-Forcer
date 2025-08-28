#pragma once
#include "control.h"
#include <sstream>
#include <string>

#ifdef UNICODE
typedef std::wostringstream sstream;
typedef std::wstring str;
static std::string __char8(const str &msg)
{
    std::ostringstream str;
    for(const wchar_t c : msg)
        str.put((char)c);
    return str.str();
}
#define TO_CHAR8(x) __char8(x)
#define TO_STRING std::to_wstring
#else
typedef std::ostringstream sstream;
typedef std::string str;
#define TO_CHAR8(x) x
#define TO_STRING std::to_string
#endif