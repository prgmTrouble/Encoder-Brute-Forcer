#include "control.h"
#include "numeric_typedefs.h"
#include "error_message.h"
#include "circular_buffer.h"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "intellisense.h"
#include "stats_display.h"

#include <atomic>
#include <bit>
#include <chrono>
#include <fstream>
#include <mutex>
#include <stdio.h>
#include <iostream>
#include <thread>
#include <vector>

static constexpr const char *OUTPUT_FILE = "output_log";
static constexpr const char *RECOVERY_FILE = "recovery";

static constexpr u8 u1024Size = 1024/(8 * sizeof(u64));
static constexpr u8 sizeS = 41;
static constexpr u16 sizeP = 820;

static constexpr u64 threadCount = 1Ui64 << 22;
static constexpr u16 blockSize = 1024,gridSize = threadCount / blockSize;
// cannot have more than 65536 registers per block, which translatest to 64 registers per thread
static constexpr u8 maxReg = 65536 / blockSize;
// max number of results to write to the output file before quitting (~1GB)
static constexpr u16 maxResults = 1Ui16 << 13;

// While the state space is `COMB(1023,41)` (~2^245), 2^64 nanoseconds is ~300 years anyway.
static u64 iteration = 0;
static std::atomic_bool kill = false;

/* copies a u1024 buffer */
static constexpr __device__ __host__ void u1024cpy(const u64 (&src)[u1024Size],u64 (&dst)[u1024Size])
{
    for(u8 i = 0;i < u1024Size;++i)
        dst[i] = src[i];
}

/*/ ========================================= GPU CODE ========================================= /*/

static __device__ u64 gpuS[threadCount][u1024Size];
static __device__ u32 gpuR[threadCount / 32];

/* counts trailing zeros (GPU code) */
static __device__ u16 ctzd(const u64 (&s)[u1024Size])
{
    u16 mask = 0; // flag non-zero words
    u8 data[u1024Size]; // index of first non-zero bit in each word
    for(u8 c = 0;c < u1024Size;++c)
    {
        mask |= (s[c] != 0) << c;
        data[c] = __ffsll(s[c]);
    }
    const u8 t = __ffs(mask);
    return ((u16)t << 6) | data[t]; // 64 * (# zero words) + (# zero bits in first non-zero word)
}

/* checks if an S maps to a valid P on the GPU */
static __global__ void __maxnreg__(maxReg) validP(const u64 limit)
{
    const u64 id = blockIdx.x * blockDim.x + threadIdx.x;
    const u64 (&s)[u1024Size] = gpuS[id];

    // map S to P
    u64 s2[u1024Size];
    u1024cpy(s,s2);
    u64 s3[u1024Size];
    u64 p[u1024Size] {0};
    for(u8 i = 0;i < sizeS - 1;++i)
    {
        const u16 a = ctzd(s2);
        s2[a >> 6] &= ~(1Ui64 << (a & 63));
        u1024cpy(s2,s3);
        for(u8 j = i + 1;j < sizeS;++j)
        {
            const u16 b = ctzd(s3);
            s3[b >> 6] &= ~(1Ui64 << (b & 63));
            const u16 c = a ^ b;
            p[c >> 6] |= 1Ui64 << (c & 63);
        }
    }

    // validate P
    u16 popc = 0;
    u8 valid = (id < limit) & (((s[0] | p[0]) & 0x1FFFE) == 0x1FFFE);
    for(u8 i = 0;i < u1024Size;++i)
    {
        valid &= !(s[i] & p[i]);
        popc += __popcll(p[i]);
    }
    valid &= popc == sizeP;

    // collect results
    u32 result = valid << (threadIdx.x & 31);
    result |= __shfl_down_sync(0xFFFFFFFF,result,16);
    result |= __shfl_down_sync(0xFFFFFFFF,result,8);
    result |= __shfl_down_sync(0xFFFFFFFF,result,4);
    result |= __shfl_down_sync(0xFFFFFFFF,result,2);
    result |= __shfl_down_sync(0xFFFFFFFF,result,1);
    if(!(threadIdx.x & 31))
        gpuR[id >> 5] = result;
}

/*/ ==================================== ASYNC S GENERATOR ==================================== /*/

static u64 cpuS[2][threadCount + 1][u1024Size];

/* gets the next permutation of bits in S */
static u64 populateS(const u8 parity) //TODO verify correctness
{
    u64 (&s)[threadCount + 1][u1024Size] = cpuS[parity];
    u1024cpy(cpuS[parity ^ 1][threadCount],s[0]);
    for(u64 t = 1;t <= threadCount;++t)
    {
        u1024cpy(s[t - 1],s[t]);
        // increment by the lowest set bit
        u8 nzByte;
        for(nzByte = 0;!s[t][nzByte];++nzByte);
        if(!(s[t][nzByte] += (1Ui64 << std::countr_zero(s[t][nzByte]))))
            while(++nzByte < u1024Size && !(++s[t][++nzByte]));
        if(nzByte == u1024Size) // detect overflow
            return t;
        // backfill missing bits
        u16 popc = sizeS;
        do popc -= std::popcount(s[t][nzByte]); while(++nzByte < u1024Size);
        s[t][0] |= ((1Ui64 << popc) - 1) << 1; // assumes sizeS <= 63
    }
    return threadCount;
}

static std::mutex lockS[2];
static std::condition_variable conditionS;
static std::atomic<u8> dirty;
static u64 limit[] {threadCount,threadCount};

static void asyncS()
{
    u8 parity = 1;
    bool sentinel;
    do
    {
        // wait until parity is dirty
        std::unique_lock lock(lockS[parity ^= 1]);
        conditionS.wait(lock,[parity]() {return (dirty & (1 << parity)) || kill;});
        if(kill) return;

        // populate parity
        limit[parity] = populateS(parity);
        sentinel = limit[parity] == threadCount;
        dirty &= 1 << (parity ^ 1);

        // notify main thread
        lock.unlock();
        conditionS.notify_one();
    }
    while(sentinel && !kill);
}

/*/ =================================== ASYNC OUTPUT THREAD =================================== /*/

static std::mutex lockIO;
static std::condition_variable conditionIO;
struct Result
{
    u64 *data;
    u8 count;

    Result() : data(nullptr),count(0) {}
    Result(u64 *data,u8 count) : data(data),count(count) {}
    Result(const Result &other) noexcept : data(other.data),count(other.count) {}
    Result(Result &&other) noexcept : data(other.data),count(other.count)
    {
        other.data = nullptr;
        other.count = 0;
    }
    ~Result() {if(data) delete[] data;}

    Result& operator=(const Result &other)
    {
        if(data) delete data;
        data = other.data;
        count = other.count;
        return *this;
    }
    Result& operator=(Result &&other) noexcept
    {
        if(data) delete data;
        data = other.data;
        other.data = nullptr;
        count = other.count;
        other.count = 0;
        return *this;
    }
};
static circlebuf<Result> results;
struct Recovery
{
    u64 iteration;
    u64 s[u1024Size];

    Recovery() : iteration(),s() {}
    Recovery(const u64 iteration,const u64(&s)[u1024Size]) : iteration(iteration)
    {
        for(u8 i = 0;i < u1024Size;++i)
            this->s[i] = s[i];
    }
    Recovery(const Recovery &other) : Recovery(other.iteration,other.s) {}
    Recovery(Recovery &&other) : Recovery(other.iteration,other.s) {}

    Recovery& operator=(const Recovery &other)
    {
        iteration = other.iteration;
        for(u8 i = 0;i < u1024Size;++i)
            s[i] = other.s[i];
        return *this;
    }
    Recovery& operator=(Recovery &&other)
    {
        iteration = other.iteration;
        for(u8 i = 0;i < u1024Size;++i)
            s[i] = other.s[i];
        return *this;
    }
};
static circlebuf<Recovery> recovery;

static constexpr u8 recoverySlots = 8;

static void asyncIO()
{
    u8 slot = 0;
    std::ofstream outputStream(OUTPUT_FILE,std::ios::out | std::ios::binary),
                  recoveryStream(RECOVERY_FILE,std::ios::out | std::ios::binary);
    circlebuf<Result> localResults;
    circlebuf<Recovery> localRecovery;
    while(!kill)
    {
        std::unique_lock lock(lockIO);
        conditionIO.wait(lock,[]() {return !results.empty() || !recovery.empty()|| kill;});
        localRecovery = std::move(recovery);
        localResults = std::move(results);
        lock.unlock();

        while(!localRecovery.empty())
        {
            const Recovery r = localRecovery.pop();
            recoveryStream
                .write(reinterpret_cast<const i8*>(&r),sizeof(Recovery))
                .flush();
            slot = (slot + 1) % recoverySlots;
            if(!slot)
                recoveryStream.seekp(0);
        }
        while(!localResults.empty())
        {
            const Result r = localResults.pop();
            outputStream
                .write(reinterpret_cast<i8*>(r.data),(u64)r.count * u1024Size * sizeof(u64))
                .flush();
        }
    }
    lockIO.lock();
    while(!recovery.empty())
    {
        const Recovery r = recovery.pop();
        recoveryStream
            .write(reinterpret_cast<const i8*>(&r),sizeof(Recovery))
            .flush();
        slot = (slot + 1) % recoverySlots;
        if(!slot)
            recoveryStream.seekp(0);
    }
    while(!results.empty())
    {
        const Result result = results.pop();
        if(result.data)
            outputStream
                .write(reinterpret_cast<i8*>(result.data),(u64)result.count * u1024Size * sizeof(u64))
                .flush();
    }
    lockIO.unlock();
}

/*/ ==================================== ASYNC STAT THREAD ==================================== /*/

static std::mutex lockStats;
static std::condition_variable conditionStats;
static circlebuf<std::chrono::nanoseconds> sectionTime;
static circlebuf<u64> resultCount;

static void asyncStats()
{
    initStats();
    updateDuration();
    circlebuf<std::chrono::nanoseconds> localTime;
    circlebuf<u64> localResult;
    while(!kill)
    {
        if(lockStats.try_lock())
        {
            localTime = std::move(sectionTime);
            localResult = std::move(resultCount);
            lockStats.unlock();
        }
        while(!localTime.empty())
            nextSection(localTime.pop());
        while(!localResult.empty())
            updateCounters(localResult.pop());
        updateDuration();
    }
    lockStats.lock();
    while(!sectionTime.empty())
        nextSection(sectionTime.pop());
    while(!resultCount.empty())
        updateCounters(resultCount.pop());
    lockStats.unlock();
    updateDuration();
    setSubstatus("execution finished, press any key to exit...");
}

/*/ ======================================= MAIN THREAD ======================================= /*/

static u32 cpuR[threadCount / 32];
#define PROFILE_START \
{\
    const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#define PROFILE_END \
    const std::chrono::nanoseconds duration = std::chrono::steady_clock::now() - begin;\
    lockStats.lock();\
    sectionTime.push(duration);\
    lockStats.unlock();\
}

static void run()
{
    dirty = 3;
    std::jthread genS(asyncS),io(asyncIO),stats(asyncStats);
    u64 localLimit;
    u8 parity = 1;
    u64 resultTotal = 0;
    circlebuf<Result> localResults;
    do
    {
        const std::chrono::steady_clock::time_point sectionStart = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(lockS[parity ^= 1],std::defer_lock);

        // wait until parity is clean
        PROFILE_START
        lock.lock();
        conditionS.wait(lock,[parity]() {return !(dirty & (1 << parity));});
        PROFILE_END

        // launch GPU kernel
        PROFILE_START
        CHECK_CUDA(cudaMemcpyToSymbol(gpuS,cpuS[parity],threadCount * u1024Size * sizeof(u64)));
        localLimit = limit[parity];
        void *args[] = {&localLimit};
        CHECK_CUDA(cudaLaunchKernel((void*)validP,gridSize,blockSize,args,0,NULL));
        CHECK_CUDA(cudaMemcpyFromSymbol(cpuR,gpuR,(threadCount / 32) * sizeof(u32)));
        PROFILE_END

        // collect results
        u64 current = 0;
        PROFILE_START
        // convert results to flat array of u64 and enqueue for IO
        for(u32 i = 0;i < threadCount / 32;++i)
        {
            const u8 popc = std::popcount(cpuR[i]);
            current += popc;
            if(popc)
            {
                u64 *result = new u64[popc * u1024Size];
                for(u8 j = 0;j < popc;++j)
                {
                    const u8 bit = std::countr_zero(cpuR[i]);
                    cpuR[i] &= ~(1Ui64 << bit);
                    for(u8 k = 0;k < u1024Size;++k)
                        result[j * u1024Size + k] = cpuS[parity][i * 32 + bit][k];
                }
                localResults.emplace(result,popc);
            }
        }
        // notify IO thread
        lockIO.lock();
        results = std::move(localResults);
        recovery.emplace(iteration,cpuS[parity][threadCount]);
        lockIO.unlock();
        conditionIO.notify_one();
        // notify S generator thread
        dirty |= 1 << parity;
        lock.unlock();
        conditionS.notify_one();
        PROFILE_END

        // publish result count
        lockStats.lock();
        resultCount.push(current);
        sectionTime.push(std::chrono::steady_clock::now() - sectionStart);
        lockStats.unlock();

        resultTotal += current;
        ++iteration;
    }
    while(localLimit == threadCount && resultTotal < maxResults);
    kill = true;
    conditionIO.notify_one();
    conditionS.notify_one();
}

int main()
{
    // switch to the best GPU
    {
        i32 deviceCount;
        CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
        i32 bestDevice = 0,bestMajor,bestMinor;
        {
            cudaDeviceProp device;
            CHECK_CUDA(cudaGetDeviceProperties(&device,0));
            bestMajor = device.major;
            bestMinor = device.minor;
        }
        for(i32 i = 1;i < deviceCount;++i)
        {
            cudaDeviceProp device;
            CHECK_CUDA(cudaGetDeviceProperties(&device,i));
            if(device.major > bestMajor || (device.major == bestMajor && device.minor > bestMinor))
            {
                bestDevice = i;
                bestMajor = device.major;
                bestMinor = device.minor;
            }
        }
        CHECK_CUDA(cudaSetDevice(bestDevice));
    
        cudaDeviceProp device;
        CHECK_CUDA(cudaGetDeviceProperties(&device,bestDevice));
        std::cout << "Selected Device Properties: " << std::endl
                  << "compute capability: " << device.major << '.' << device.minor << std::endl
                  << "regsPerBlock: " << device.regsPerBlock << std::endl
                  << "regsPerMultiprocessor: " << device.regsPerMultiprocessor << std::endl
                  << "maxGridSize: " << device.maxGridSize << std::endl
                  << "maxBlocksPerMultiProcessor: " << device.maxBlocksPerMultiProcessor << std::endl
                  << "maxThreadsDim: " << device.maxThreadsDim << std::endl
                  << "maxThreadsPerBlock: " << device.maxThreadsPerBlock << std::endl
                  << "maxThreadsPerMultiProcessor: " << device.maxThreadsPerMultiProcessor << std::endl
                  << "totalConstMem: " << device.totalConstMem << std::endl
                  << "totalGlobalMem:" << device.totalGlobalMem << std::endl
                  << "warpSize: " << device.warpSize << std::endl;
    }

    // load recovery data
    {
        Recovery best;
        bool hasRecovery = false;
        std::ifstream in(RECOVERY_FILE,std::ios::in | std::ios::binary);
        while(in && !in.eof())
        {
            Recovery r;
            in.read(reinterpret_cast<i8*>(&r),sizeof(Recovery));
            u16 popc = 0;
            for(const u64 x : r.s)
                popc += std::popcount(x);
            if(popc != sizeS)
                continue;
            if(!hasRecovery || best.iteration < r.iteration)
                best = r;
            hasRecovery = true;
        }
        if(hasRecovery)
        {
            setIteration(iteration = best.iteration + 1);
            for(u8 i = 0;i < u1024Size;++i)
                cpuS[1][threadCount][i] = best.s[i];
        }
        else
            cpuS[1][threadCount][0] = ((1Ui64 << sizeS) - 1) << 1;
    }

    // run the kernel
    run();

    // cleanup
    CHECK_CUDA(cudaDeviceReset());
    std::cin.get();
}