/*
 This code generates random Erdos Renyi graph using cuda.
 The corresponding author is Sadegh Nobari

 If use please cite:
 @inproceedings{Nobari:2011,
 author = {Nobari, Sadegh and Lu, Xuesong and Karras, Panagiotis and Bressan, St\'{e}phane},
 title = {Fast random graph generation},
 booktitle = {Proceedings of the 14th International Conference on Extending Database Technology},
 series = {EDBT/ICDT '11},
 year = {2011},
 isbn = {978-1-4503-0528-0},
 location = {Uppsala, Sweden},
 pages = {331--342},
 numpages = {12},
 url = {http://doi.acm.org/10.1145/1951365.1951406},
 doi = {http://doi.acm.org/10.1145/1951365.1951406},
 acmid = {1951406},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {Erd\H{o}s-r\'{e}nyi, Gilbert, parallel algorithm, random graphs},
 } 

 Last update 19 Jun 2011
*/

/*
After introducing the CURAND library in 2011, it is recommended to use the CURAND uniform random number generator.
please check the comments in the initialization sections for more detail.
In kernel function RND simply generetes a uniform random number.
*/

#ifndef _FastGG_KERNEL_H_
#define _FastGG_KERNEL_H_
#include "FastGG.h"
#define RND curand_uniform(&localState) //Output range excludes 0.0f but includes 1.0f
////////////////////////////////////////////////////////////////////////////////
//! RNG init kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void
initRNG(curandState * const rngStates, const unsigned int seed)
{
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialise the RNG
    curand_init(seed, tid, 0, &rngStates[tid]);
}
////////////////////////////////////////////////////////////////////////////////
//! PER Kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void
Kernel_PER( UINT Seed,VTYPE * EdgeList,VTYPE * Valids,VTYPE pXMaxRND,UINT ItemsPerThread,VTYPE NumItems)
{
      
  const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  //Initialization
  VTYPE pos, data;//,c;
  //const UINT cycles=10;
  const unsigned long long A    = 16807;      //ie 7**5
  data = Seed+tid*16807;
  
  pos = tid*ItemsPerThread;
  //Skip the first Cycle numbers
  //for (c=1; c<=cycles ; c++)
  //{
      data=(data*A)%M;
      data=(data*A)%M;
      data=(data*A)%M;
      
  //}
                
  for(; pos < (tid+1)*ItemsPerThread; pos++)
        {
           //Generating Random Number
           data=(data*A)%M;
           if(data<pXMaxRND)
           {
                EdgeList[pos]=pos;
                Valids[pos]=1;
           }
           else
                Valids[pos]=0;
        }
}
////////////////////////////////////////////////////////////////////////////////
//! PXER Kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void
Kernel_PZER( UINT Seed,VTYPE * Results,UINT ItemsPerThread, float log1p ,float logPmax,VTYPE offset,VTYPE NumItems)
{
      
  const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  //Initialization
  VTYPE pos, data;
  //const UINT cycles=3;
  const unsigned long long A    = 16807;      //ie 7**5
  data = Seed+tid*16807;
  
  pos = tid*ItemsPerThread;
  
  //Skip the first Cycle numbers
  //for (c=1; c<=cycles ; c++)
  //{
      data=(data*A)%M;
      data=(data*A)%M;
      data=(data*A)%M;
      
  //}
                
  for(; pos < (tid+1)*ItemsPerThread & pos<NumItems; pos++)
        {
           //Generating Random Number
           //temp = data * A;
           //data = temp%M;//temp - M * floor ( temp * reciprocal_m );
           data=(data*A)%M;
           //computing skip and wrtie it to global memory
          
            if(log1p==0 && tid==0)
            Results[0]=M-1;
            else if(log1p==M)
            Results[pos]=1;
            else
            Results[pos]=ceil( (log((double)data)*log1p)-logPmax)+1;
                
        }
}
////////////////////////////////////////////////////////////////////////////////
//! PPreZER Kernels for precomputations 1 to 10
////////////////////////////////////////////////////////////////////////////////
__global__ void
Kernel_PPreZER10( UINT Seed,VTYPE * Results,UINT ItemsPerThread, float log1p ,float logPmax,VTYPE offset,VTYPE NumItems,VTYPE precomp1,VTYPE precomp2,VTYPE precomp3,VTYPE precomp4,VTYPE precomp5,VTYPE precomp6,VTYPE precomp7,VTYPE precomp8,VTYPE precomp9,VTYPE precomp10)
{
      
  const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  //Initialization
  VTYPE pos, data,skip;
  //const UINT cycles=3;
  const unsigned long long A    = 16807;      //ie 7**5
  data = Seed+tid;
  
  pos = tid*ItemsPerThread;
  
  //Skip the first Cycle numbers
  //for (c=1; c<=cycles ; c++)
  //{
      data=(data*A)%M;
      data=(data*A)%M;
      data=(data*A)%M;
      
  //}
                
  for(; pos < (tid+1)*ItemsPerThread & pos<NumItems; pos++)
        {
           //Generating Random Number
           //temp = data * A;
           //data = temp%M;//temp - M * floor ( temp * reciprocal_m );
           data=(data*A)%M;
           skip=M-data;
           //computing skip and wrtie it to global memory
           if(skip<precomp1)
                skip=1;
           else if(skip<precomp2)
                skip=2;
           else if(skip<precomp3)
                skip=3;
           else if(skip<precomp4)
                skip=4;
           else if(skip<precomp5)
                skip=5;
           else if(skip<precomp6)
                skip=6;
           else if(skip<precomp7)
                skip=7;
           else if(skip<precomp8)
                skip=8;
           else if(skip<precomp9)
                skip=9;
           else if(skip<precomp10)
                skip=10;
           else
           {
                if(log1p==0)
                skip=M;
                else
                skip=ceil( (log((double)data)*log1p)-logPmax)+1;
           }
                      Results[pos]=skip;
                      
        }
}
__global__ void
Kernel_PPreZER9( UINT Seed,VTYPE * Results,UINT ItemsPerThread, float log1p ,float logPmax,VTYPE offset,VTYPE NumItems,VTYPE precomp1,VTYPE precomp2,VTYPE precomp3,VTYPE precomp4,VTYPE precomp5,VTYPE precomp6,VTYPE precomp7,VTYPE precomp8,VTYPE precomp9)
{
      
  const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  //Initialization
  VTYPE pos, data,skip;
  //const UINT cycles=3;
  const unsigned long long A    = 16807;      //ie 7**5
  data = Seed+tid;
  
  pos = tid*ItemsPerThread;
  
  //Skip the first Cycle numbers
  //for (c=1; c<=cycles ; c++)
  //{
      data=(data*A)%M;
      data=(data*A)%M;
      data=(data*A)%M;
      
  //}
                
  for(; pos < (tid+1)*ItemsPerThread & pos<NumItems; pos++)
        {
           //Generating Random Number
           //temp = data * A;
           //data = temp%M;//temp - M * floor ( temp * reciprocal_m );
           data=(data*A)%M;
           skip=M-data;
           //computing skip and wrtie it to global memory
           if(skip<precomp1)
                skip=1;
           else if(skip<precomp2)
                skip=2;
           else if(skip<precomp3)
                skip=3;
           else if(skip<precomp4)
                skip=4;
           else if(skip<precomp5)
                skip=5;
           else if(skip<precomp6)
                skip=6;
           else if(skip<precomp7)
                skip=7;
           else if(skip<precomp8)
                skip=8;
           else if(skip<precomp9)
                skip=9;
           else
           {
                if(log1p==0)
                skip=M;
                else
                skip=ceil( (log((double)data)*log1p)-logPmax)+1;
           }
           
           Results[pos]=skip;
                
        }
}
__global__ void
Kernel_PPreZER8( UINT Seed,VTYPE * Results,UINT ItemsPerThread, float log1p ,float logPmax,VTYPE offset,VTYPE NumItems,VTYPE precomp1,VTYPE precomp2,VTYPE precomp3,VTYPE precomp4,VTYPE precomp5,VTYPE precomp6,VTYPE precomp7,VTYPE precomp8)
{
      
  const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  //Initialization
  VTYPE pos, data,skip;
  //const UINT cycles=3;
  const unsigned long long A    = 16807;      //ie 7**5
  data = Seed+tid;
  
  pos = tid*ItemsPerThread;
  
  //Skip the first Cycle numbers
  //for (c=1; c<=cycles ; c++)
  //{
      data=(data*A)%M;
      data=(data*A)%M;
      data=(data*A)%M;
      
  //}
                
  for(; pos < (tid+1)*ItemsPerThread & pos<NumItems; pos++)
        {
           //Generating Random Number
           //temp = data * A;
           //data = temp%M;//temp - M * floor ( temp * reciprocal_m );
           data=(data*A)%M;
           skip=M-data;
           //computing skip and wrtie it to global memory
           if(skip<precomp1)
                skip=1;
           else if(skip<precomp2)
                skip=2;
           else if(skip<precomp3)
                skip=3;
           else if(skip<precomp4)
                skip=4;
           else if(skip<precomp5)
                skip=5;
           else if(skip<precomp6)
                skip=6;
           else if(skip<precomp7)
                skip=7;
           else if(skip<precomp8)
                skip=8;
           else
           {
                if(log1p==0)
                skip=M;
                else
                skip=ceil( (log((double)data)*log1p)-logPmax)+1;
           }
           
           Results[pos]=skip;
                
        }
}
__global__ void
Kernel_PPreZER7( UINT Seed,VTYPE * Results,UINT ItemsPerThread, float log1p ,float logPmax,VTYPE offset,VTYPE NumItems,VTYPE precomp1,VTYPE precomp2,VTYPE precomp3,VTYPE precomp4,VTYPE precomp5,VTYPE precomp6,VTYPE precomp7)
{
      
  const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  //Initialization
  VTYPE pos, data,skip;
  //const UINT cycles=3;
  const unsigned long long A    = 16807;      //ie 7**5
  data = Seed+tid;
  
  pos = tid*ItemsPerThread;
  
  //Skip the first Cycle numbers
  //for (c=1; c<=cycles ; c++)
  //{
      data=(data*A)%M;
      data=(data*A)%M;
      data=(data*A)%M;
      
  //}
                
  for(; pos < (tid+1)*ItemsPerThread & pos<NumItems; pos++)
        {
           //Generating Random Number
           //temp = data * A;
           //data = temp%M;//temp - M * floor ( temp * reciprocal_m );
           data=(data*A)%M;
           skip=M-data;
           //computing skip and wrtie it to global memory
           if(skip<precomp1)
                skip=1;
           else if(skip<precomp2)
                skip=2;
           else if(skip<precomp3)
                skip=3;
           else if(skip<precomp4)
                skip=4;
           else if(skip<precomp5)
                skip=5;
           else if(skip<precomp6)
                skip=6;
           else if(skip<precomp7)
                skip=7;
           else
           {
                if(log1p==0)
                skip=M;
                else
                skip=ceil( (log((double)data)*log1p)-logPmax)+1;
           }
           
           Results[pos]=skip;
                
        }
}
__global__ void
Kernel_PPreZER6( UINT Seed,VTYPE * Results,UINT ItemsPerThread, float log1p ,float logPmax,VTYPE offset,VTYPE NumItems,VTYPE precomp1,VTYPE precomp2,VTYPE precomp3,VTYPE precomp4,VTYPE precomp5,VTYPE precomp6)
{
      
  const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  //Initialization
  VTYPE pos, data,skip;
  //const UINT cycles=3;
  const unsigned long long A    = 16807;      //ie 7**5
  data = Seed+tid;
  
  pos = tid*ItemsPerThread;
  
  //Skip the first Cycle numbers
  //for (c=1; c<=cycles ; c++)
  //{
      data=(data*A)%M;
      data=(data*A)%M;
      data=(data*A)%M;
      
  //}
                
  for(; pos < (tid+1)*ItemsPerThread & pos<NumItems; pos++)
        {
           //Generating Random Number
           //temp = data * A;
           //data = temp%M;//temp - M * floor ( temp * reciprocal_m );
           data=(data*A)%M;
           skip=M-data;
           //computing skip and wrtie it to global memory
           if(skip<precomp1)
                skip=1;
           else if(skip<precomp2)
                skip=2;
           else if(skip<precomp3)
                skip=3;
           else if(skip<precomp4)
                skip=4;
           else if(skip<precomp5)
                skip=5;
           else if(skip<precomp6)
                skip=6;
           else
           {
                if(log1p==0)
                skip=M;
                else
                skip=ceil( (log((double)data)*log1p)-logPmax)+1;
           }
           
           Results[pos]=skip;
                
        }
}
__global__ void
Kernel_PPreZER5( UINT Seed,VTYPE * Results,UINT ItemsPerThread, float log1p ,float logPmax,VTYPE offset,VTYPE NumItems,VTYPE precomp1,VTYPE precomp2,VTYPE precomp3,VTYPE precomp4,VTYPE precomp5)
{
      
  const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  //Initialization
  VTYPE pos, data,skip;
  //const UINT cycles=3;
  const unsigned long long A    = 16807;      //ie 7**5
  data = Seed+tid;
  
  pos = tid*ItemsPerThread;
  
  //Skip the first Cycle numbers
  //for (c=1; c<=cycles ; c++)
  //{
      data=(data*A)%M;
      data=(data*A)%M;
      data=(data*A)%M;
      
  //}
                
  for(; pos < (tid+1)*ItemsPerThread & pos<NumItems; pos++)
        {
           //Generating Random Number
           //temp = data * A;
           //data = temp%M;//temp - M * floor ( temp * reciprocal_m );
           data=(data*A)%M;
           skip=M-data;
           //computing skip and wrtie it to global memory
           if(skip<precomp1)
                skip=1;
           else if(skip<precomp2)
                skip=2;
           else if(skip<precomp3)
                skip=3;
           else if(skip<precomp4)
                skip=4;
           else if(skip<precomp5)
                skip=5;
           else
           {
                if(log1p==0)
                skip=M;
                else
                skip=ceil( (log((double)data)*log1p)-logPmax)+1;
           }
           
           Results[pos]=skip;
                
        }
}
__global__ void
Kernel_PPreZER4( UINT Seed,VTYPE * Results,UINT ItemsPerThread, float log1p ,float logPmax,VTYPE offset,VTYPE NumItems,VTYPE precomp1,VTYPE precomp2,VTYPE precomp3,VTYPE precomp4)
{
      
  const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  //Initialization
  VTYPE pos, data,skip;
  //const UINT cycles=3;
  const unsigned long long A    = 16807;      //ie 7**5
  data = Seed+tid;
  
  pos = tid*ItemsPerThread;
  
  //Skip the first Cycle numbers
  //for (c=1; c<=cycles ; c++)
  //{
      data=(data*A)%M;
      data=(data*A)%M;
      data=(data*A)%M;
      
  //}
                
  for(; pos < (tid+1)*ItemsPerThread & pos<NumItems; pos++)
        {
           //Generating Random Number
           //temp = data * A;
           //data = temp%M;//temp - M * floor ( temp * reciprocal_m );
           data=(data*A)%M;
           skip=M-data;
           //computing skip and wrtie it to global memory
           if(skip<precomp1)
                skip=1;
           else if(skip<precomp2)
                skip=2;
           else if(skip<precomp3)
                skip=3;
           else if(skip<precomp4)
                skip=4;
           else
           {
                if(log1p==0)
                skip=M;
                else
                skip=ceil( (log((double)data)*log1p)-logPmax)+1;
           }
           
           Results[pos]=skip;
                
        }
}
__global__ void
Kernel_PPreZER3( UINT Seed,VTYPE * Results,UINT ItemsPerThread, float log1p ,float logPmax,VTYPE offset,VTYPE NumItems,VTYPE precomp1,VTYPE precomp2,VTYPE precomp3)
{
      
  const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  //Initialization
  VTYPE pos, data,skip;
  //const UINT cycles=3;
  const unsigned long long A    = 16807;      //ie 7**5
  data = Seed+tid;
  
  pos = tid*ItemsPerThread;
  
  //Skip the first Cycle numbers
  //for (c=1; c<=cycles ; c++)
  //{
      data=(data*A)%M;
      data=(data*A)%M;
      data=(data*A)%M;
      
  //}
                
  for(; pos < (tid+1)*ItemsPerThread & pos<NumItems; pos++)
        {
           //Generating Random Number
           //temp = data * A;
           //data = temp%M;//temp - M * floor ( temp * reciprocal_m );
           data=(data*A)%M;
           skip=M-data;
           //computing skip and wrtie it to global memory
           if(skip<precomp1)
                skip=1;
           else if(skip<precomp2)
                skip=2;
           else if(skip<precomp3)
                skip=3;
           else
           {
                if(log1p==0)
                skip=M;
                else
                skip=ceil( (log((double)data)*log1p)-logPmax)+1;
           }
           
           Results[pos]=skip;
                
        }
}
__global__ void
Kernel_PPreZER2( UINT Seed,VTYPE * Results,UINT ItemsPerThread, float log1p ,float logPmax,VTYPE offset,VTYPE NumItems,VTYPE precomp1,VTYPE precomp2)
{
      
  const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  //Initialization
  VTYPE pos, data,skip;
  //const UINT cycles=3;
  const unsigned long long A    = 16807;      //ie 7**5
  data = Seed+tid;
  
  pos = tid*ItemsPerThread;
  
  //Skip the first Cycle numbers
  //for (c=1; c<=cycles ; c++)
  //{
      data=(data*A)%M;
      data=(data*A)%M;
      data=(data*A)%M;
      
  //}
                
  for(; pos < (tid+1)*ItemsPerThread & pos<NumItems; pos++)
        {
           //Generating Random Number
           //temp = data * A;
           //data = temp%M;//temp - M * floor ( temp * reciprocal_m );
           data=(data*A)%M;
           skip=M-data;
           //computing skip and wrtie it to global memory
           if(skip<precomp1)
                skip=1;
           else if(skip<precomp2)
                skip=2;
           else
           {
                if(log1p==0)
                skip=M;
                else
                skip=ceil( (log((double)data)*log1p)-logPmax)+1;
           }
           
           Results[pos]=skip;
                
        }
}
__global__ void
Kernel_PPreZER1( UINT Seed,VTYPE * Results,UINT ItemsPerThread, float log1p ,float logPmax,VTYPE offset,VTYPE NumItems,VTYPE precomp1)
{
      
  const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  //Initialization
  VTYPE pos, data,skip;
  //const UINT cycles=3;
  const unsigned long long A    = 16807;      //ie 7**5
  data = Seed+tid;
  
  pos = tid*ItemsPerThread;
  
  //Skip the first Cycle numbers
  //for (c=1; c<=cycles ; c++)
  //{
      data=(data*A)%M;
      data=(data*A)%M;
      data=(data*A)%M;
      
  //}
                
  for(; pos < (tid+1)*ItemsPerThread & pos<NumItems; pos++)
        {
           //Generating Random Number
           //temp = data * A;
           //data = temp%M;//temp - M * floor ( temp * reciprocal_m );
           data=(data*A)%M;
           skip=M-data;
           //computing skip and wrtie it to global memory
           if(skip<precomp1)
                skip=1;
           else
           {
                if(log1p==0)
                skip=M;
                else
                skip=ceil( (log((double)data)*log1p)-logPmax)+1;
           }
           
           Results[pos]=skip;
                
        }
}
////////////////////////////////////////////////////////////////////////////////
//! Kernel Invokers
////////////////////////////////////////////////////////////////////////////////
void initRNGInvoker( dim3 dimGrid, dim3 dimBlock, curandState * const rngStates, const unsigned int seed)
{
    initRNG<<< dimGrid, dimBlock >>>( rngStates, seed );
}
void PER_Invoker( dim3 dimGrid,dim3 dimBlock,UINT Seed,VTYPE * EdgeList,VTYPE * Valids,VTYPE pXMaxRND,UINT ItemsPerThread,VTYPE NumItems)
{
    Kernel_PER<<< dimGrid, dimBlock >>>(Seed,EdgeList,Valids,pXMaxRND,ItemsPerThread,NumItems);
}
void PZER_Invoker( dim3 dimGrid,dim3 dimBlock,UINT Seed,VTYPE * Results,UINT ItemsPerThread, float log1p ,float logPmax,VTYPE offset,VTYPE NumItems)
{
    Kernel_PZER<<< dimGrid, dimBlock >>>(Seed,Results,ItemsPerThread,log1p,logPmax,offset,NumItems);
}
void PPreZER_Invoker( dim3 dimGrid,dim3 dimBlock,UINT Seed,VTYPE * Results,UINT ItemsPerThread, float log1p ,float logPmax,VTYPE offset,VTYPE NumItems,VTYPE * pres,VTYPE numpre)
{
    if(numpre==10)
        Kernel_PPreZER10<<< dimGrid, dimBlock >>>(Seed,Results,ItemsPerThread,log1p,logPmax,offset,NumItems,pres[0],pres[1],pres[2],pres[3],pres[4],pres[5],pres[6],pres[7],pres[8],pres[9]);
    else if(numpre==9)
        Kernel_PPreZER9<<< dimGrid, dimBlock >>>(Seed,Results,ItemsPerThread,log1p,logPmax,offset,NumItems,pres[0],pres[1],pres[2],pres[3],pres[4],pres[5],pres[6],pres[7],pres[8]);
    else if(numpre==8)
        Kernel_PPreZER8<<< dimGrid, dimBlock >>>(Seed,Results,ItemsPerThread,log1p,logPmax,offset,NumItems,pres[0],pres[1],pres[2],pres[3],pres[4],pres[5],pres[6],pres[7]);
    else if(numpre==7)
        Kernel_PPreZER7<<< dimGrid, dimBlock >>>(Seed,Results,ItemsPerThread,log1p,logPmax,offset,NumItems,pres[0],pres[1],pres[2],pres[3],pres[4],pres[5],pres[6]);
    else if(numpre==6)
        Kernel_PPreZER6<<< dimGrid, dimBlock >>>(Seed,Results,ItemsPerThread,log1p,logPmax,offset,NumItems,pres[0],pres[1],pres[2],pres[3],pres[4],pres[5]);
    else if(numpre==5)
        Kernel_PPreZER5<<< dimGrid, dimBlock >>>(Seed,Results,ItemsPerThread,log1p,logPmax,offset,NumItems,pres[0],pres[1],pres[2],pres[3],pres[4]);
    else if(numpre==4)
        Kernel_PPreZER4<<< dimGrid, dimBlock >>>(Seed,Results,ItemsPerThread,log1p,logPmax,offset,NumItems,pres[0],pres[1],pres[2],pres[3]);
    else if(numpre==3)
        Kernel_PPreZER3<<< dimGrid, dimBlock >>>(Seed,Results,ItemsPerThread,log1p,logPmax,offset,NumItems,pres[0],pres[1],pres[2]);
    else if(numpre==2)
        Kernel_PPreZER2<<< dimGrid, dimBlock >>>(Seed,Results,ItemsPerThread,log1p,logPmax,offset,NumItems,pres[0],pres[1]);
    else
        Kernel_PPreZER1<<< dimGrid, dimBlock >>>(Seed,Results,ItemsPerThread,log1p,logPmax,offset,NumItems,pres[0]);
    
}
#endif // #ifndef _FastGG_KERNEL_H_
