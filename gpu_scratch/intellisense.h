#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
void __syncwarp(unsigned mask = ~0);
template<typename T>
T __shfl_down_sync(unsigned,T,unsigned,int width = 32);
int __ffs(int);
int __ffsll(long long int);
int __popcll(unsigned long long int);
#define CONFIG2(grid,block)
#define CONFIG3(grid,block,sh_mem)
#define CONFIG4(grid,block,sh_mem,stream)
#define __maxnreg__(x)
#else
#define CONFIG2(grid,block) <<<grid,block>>>
#define CONFIG3(grid,block,sh_mem) <<<grid,block,sh_mem>>>
#define CONFIG4(grid,block,sh_mem,stream) <<<grid,block,sh_mem,stream>>>
#endif // __INTELLISENSE__