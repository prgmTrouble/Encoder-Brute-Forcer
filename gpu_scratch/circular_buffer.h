#pragma once
#include "numeric_typedefs.h"
#include <utility>
#ifdef DEBUG
#include <stdexcept>
#endif

template<typename T,typename C = u8,u64 N = 128ui64>
struct circlebuf
{
    T buffer[N];
    C l,r;
    bool reversed;

    #pragma warning(disable:26495)
    circlebuf() : l(0),r(0),reversed(false) {}
    #pragma warning(default:26495)
    
    circlebuf& operator=(const circlebuf &);
    circlebuf& operator=(circlebuf &&);

    bool empty() const;
    bool full() const;
    C size() const;

    void swap(circlebuf&);

    void push(const T&);
    void push(T&&);
    template<typename...A> void emplace(A&&...);
    T&& pop();
};

template<typename T,typename C,u64 N>
circlebuf<T,C,N>& circlebuf<T,C,N>::operator=(const circlebuf &other)
{
    l = r = 0;
    reversed = false;
    for(C i = other.l;i < (other.reversed ? N : other.r);++r,++i)
        buffer[r] = other.buffer[i];
    if(other.reversed && other.r != N)
        for(C i = 0;i < other.r;++r,++i)
            buffer[r] = other.buffer[i];
    return *this;
}
template<typename T,typename C,u64 N>
circlebuf<T,C,N>& circlebuf<T,C,N>::operator=(circlebuf &&other)
{
    l = r = 0;
    reversed = false;
    for(C i = other.l;i < (other.reversed ? N : other.r);++r,++i)
        buffer[r] = std::move(other.buffer[i]);
    if(other.reversed && other.r != N)
        for(C i = 0;i < other.r;++r,++i)
            buffer[r] = std::move(other.buffer[i]);
    other.l = other.r = 0;
    other.reversed = false;
    return *this;
}

template<typename T,typename C,u64 N>
bool circlebuf<T,C,N>::empty() const {return !reversed && l == r;}
template<typename T,typename C,u64 N>
bool circlebuf<T,C,N>::full() const {return (l == r && reversed) || (!l && r == N);}

template<typename T,typename C,u64 N>
C circlebuf<T,C,N>::size() const {return reversed ? (C)(N - l + r) : (C)(r - l);}
    
template<typename T,typename C,u64 N>
void circlebuf<T,C,N>::swap(circlebuf &other)
{
    C i = 0,j = 0;
    T tbuf[N];
    for(C k = l;k < reversed ? N : r;++i,++k)
        tbuf[i] = std::move(buffer[k]);
    if(reversed && r != N)
        for(C k = 0;k < r;++i,++k)
            tbuf[i] = std::move(buffer[k]);
    for(C k = other.l;k < other.reversed ? N : other.r;++j,++k)
        buffer[j] = std::move(other.buffer[k]);
    if(other.reversed && other.r != N)
        for(C k = 0;k < other.r;++j,++k)
            buffer[j] = std::move(other.buffer[k]);
    for(C k = 0;k < i;++k)
        other.buffer[k] = std::move(tbuf[k]);
    l = other.l = 0;
    r = j;
    other.r = i;
    reversed = other.reversed = false;
}

template<typename T,typename C,u64 N>
template<typename ...A>
void circlebuf<T,C,N>::emplace(A&&...args)
{
    push(T(std::forward<A>(args)...));
}

template<typename T,typename C,u64 N>
void circlebuf<T,C,N>::push(const T &t)
{
    #ifdef DEBUG
    if(full()) throw std::logic_error("push called on full buffer");
    #endif

    if(r == N)
    {
        reversed = true;
        r = 1;
        buffer[0] = t;
    }
    else
        buffer[r++] = t;
}

template<typename T,typename C,u64 N>
void circlebuf<T,C,N>::push(T &&t)
{
    #ifdef DEBUG
    if(full()) throw std::logic_error("push called on full buffer");
    #endif

    if(r == N)
    {
        reversed = true;
        r = 1;
        buffer[0] = std::forward<T>(t);
    }
    else
        buffer[r++] = std::forward<T>(t);
}

template<typename T,typename C,u64 N>
T&& circlebuf<T,C,N>::pop()
{
    #ifdef DEBUG
    if(empty()) throw std::logic_error("pop called on empty buffer");
    #endif

    const C ref = l++;
    if(l == r || l == N)
    {
        reversed = false;
        if(l == r)
            r = 0;
        l = 0;
    }

    return std::move(buffer[ref]);
}