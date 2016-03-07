/*
 This code generates random Erdos Renyi graph using CUDA.
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

#define a_PER			1
#define a_PZER			2
#define a_PPreZER		3


#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes, system

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <iostream>
#include <fstream>
#include <algorithm>

// includes, kernels
#include <FastGG_kernel.cu>
#include "include\cudpp.h"

using namespace std;

int numThreadsPerBlock = 32;
int numBlocks = 4;

int runs = 1;
int algorithm=1;
VTYPE Vertices=1000;
float P=0.1;
VTYPE precomputing=10;
VTYPE seed=7;
VTYPE iternum = 10;
#define RNDC ((double)rand()/RAND_MAX)

/************************************************************************/
/* Init CUDA                                                            */
/************************************************************************/
#if __DEVICE_EMULATION__

bool InitCUDA(void){return true;}

#else
bool InitCUDA(void)
{
    int count = 0;
    int i = 0;

    cudaGetDeviceCount(&count);
    if(count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if(prop.major >= 1) {
                break;
            }
        }
    }
    if(i == count) {
        fprintf(stderr, "There is no device supporting CUDA.\n");
        return false;
    }
    cudaSetDevice(i);

    printf("CUDA initialized.\n");
    return true;
}

#endif

/************************************************************************/
/* Main function                                                        */
// Times, i.e. elapsed [i]
// 0 PRNG + Skip
// 1 Scan / Compact
// 2 Transfer
// 3 Total
// 4 Total - Transfer
/************************************************************************/
ofstream Logs1;
ofstream Logs2;

int main(int argc, char* argv[])
{
    if(!InitCUDA()) {
        return 0;
    }
    parse_args(argc,argv);
    CUT_DEVICE_INIT(1, argv);

    double elapsedAVE[5];
    double* elapsed=new double[5];
    VTYPE * othersAVE=new VTYPE[2];
    VTYPE * others=new VTYPE[2];

	elapsedAVE[0]=0.0;elapsedAVE[1]=0.0;elapsedAVE[2]=0.0;elapsedAVE[3]=0.0;elapsedAVE[4]=0.0;
	othersAVE[0]=0;othersAVE[1]=0;
	
    if( algorithm == a_PER ){
		//PER algorithm
		cout << "*** Running PER algorithm ***" << endl;

        for(int r=0;r<runs;r++)
        {
            runPER(argc,argv,Vertices,r+seed,elapsed,others);
            elapsedAVE[0]+=elapsed[0];elapsedAVE[1]+=elapsed[1];elapsedAVE[2]+=elapsed[2];elapsedAVE[3]+=elapsed[3];elapsedAVE[4]+=elapsed[4];
            if(r%1==0)cudaThreadExit();
        }

        elapsedAVE[0]/=runs;elapsedAVE[1]/=runs;elapsedAVE[2]/=runs;elapsedAVE[3]/=runs;elapsedAVE[4]/=runs;
        Logs2.open("TIMES.csv",ios_base::app);
        Logs2<<elapsedAVE[4]<<","<<elapsedAVE[3]<<","<<elapsedAVE[0]+elapsedAVE[1]+elapsedAVE[2]<<","<<elapsedAVE[0]<<","<<elapsedAVE[1]<<","<<elapsedAVE[2]<<","<<runs<<","<<P<<","<<16<<endl;
        Logs2.close();
    }
	else if( algorithm == a_PZER ){
        //PZER algorithm
		cout << "*** Running PZER algorithm ***" << endl;
            
        for(int r=0;r<runs;r++)
            {
                runPZER( argc, argv,Vertices,r+seed,elapsed,others);
                elapsedAVE[0]+=elapsed[0];elapsedAVE[1]+=elapsed[1];elapsedAVE[2]+=elapsed[2];elapsedAVE[3]+=elapsed[3];elapsedAVE[4]+=elapsed[4];
                othersAVE[0]+=others[0];othersAVE[1]+=others[1];
                if(r%2==0)cudaThreadExit();
            }
        elapsedAVE[0]/=runs;elapsedAVE[1]/=runs;elapsedAVE[2]/=runs;elapsedAVE[3]/=runs;elapsedAVE[4]/=runs;
        othersAVE[0]/=runs;othersAVE[1]/=runs;
        Logs2.open("TIMES.csv",ios_base::app);
        Logs2<<elapsedAVE[4]<<","<<elapsedAVE[3]<<","<<elapsedAVE[0]+elapsedAVE[1]+elapsedAVE[2]<<","<<elapsedAVE[0]<<","<<elapsedAVE[1]<<","<<elapsedAVE[2]<<","<<runs<<","<<P<<","<<othersAVE[0]<<","<<othersAVE[1]<<endl;
        Logs2.close();
    }
    else if( algorithm == a_PPreZER ){
		//PPreZER
		cout << "*** Running PPreZER algorithm ***" << endl;

        for(int r=0;r<runs;r++)
        {
            runPPreZER( argc, argv,Vertices,r+seed,elapsed,others,10/*precomputing*/);
            elapsedAVE[0]+=elapsed[0];elapsedAVE[1]+=elapsed[1];elapsedAVE[2]+=elapsed[2];elapsedAVE[3]+=elapsed[3];elapsedAVE[4]+=elapsed[4];
            othersAVE[0]+=others[0];othersAVE[1]+=others[1];
            if(r%2==0)cudaThreadExit();

        }

        elapsedAVE[0]/=runs;elapsedAVE[1]/=runs;elapsedAVE[2]/=runs;elapsedAVE[3]/=runs;elapsedAVE[4]/=runs;
        othersAVE[0]/=runs;othersAVE[1]/=runs;
        Logs2.open("TIMES.csv",ios_base::app);
        Logs2<<"PRE"<<precomputing<<","<<elapsedAVE[4]<<","<<elapsedAVE[3]<<","<<elapsedAVE[0]+elapsedAVE[1]+elapsedAVE[2]<<","<<elapsedAVE[0]<<","<<elapsedAVE[1]<<","<<elapsedAVE[2]<<","<<runs<<","<<P<<","<<othersAVE[0]<<","<<othersAVE[1]<<endl;
        Logs2.close();
    }

}


int NextMul(int Number)
{
    VTYPE temp=numThreadsPerBlock*numBlocks;
    temp=temp*ceil((double)Number/(double)(temp));
    return temp;//min(temp,MAXITEMS);
} 
int NextPow2(int Number)
{
    float temp=log((double)Number)/log(2.0);
    if(floor(temp)==temp)return pow(2.0,(int)temp);
    return pow(2.0,(int)temp+1);
} 

void usage(char *program_name, int status) {
    if (status == EXIT_SUCCESS)
    {
        cout << "Usage: " << program_name << " -n vertices -p probability" << endl
            << "  Generates a random Erdos-Renyi graph with a fixed number of vertices (Gnp)" << endl
			<< "-h for usage information" << endl
			<< "-n vertices -p probability" << endl
			<< "-r number of runs -s Seed" << endl
			<< "-a Algorithm " << a_PER << ":PER " << a_PZER << ":PZER " << a_PPreZER << ":PPreZER"<< endl
            <<"FastGG.exe -n 1000 -p .1 -r 1 -a 1"<<endl;
    }
    else
    {
        cerr << "Try '" << program_name << " -h' for usage information." << endl;
    }
    exit(status);
}

void parse_args(int argc, char **argv) {

    char c;
    for (int x= 1; x < argc; ++x)
    {
        switch (argv[x][1]) {
			case 'h':       /* help */
				usage(argv[0], EXIT_SUCCESS);
				break;
			case 'n':       /* n vertices */
				sscanf(argv[++x], "%u", &Vertices);
				break;
			case 'a':       /* Algorithm */
				sscanf(argv[++x], "%u", &algorithm);
				break;
			case 'p':       /* connection probability */
				sscanf(argv[++x], "%f", &P);
				break;
			case 'r':       /* Runs */
				sscanf(argv[++x], "%u", &runs);
				break;
			case 's':       /* Seed */
				sscanf(argv[++x], "%u", &seed);
				break;
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Run PER for CUDA
////////////////////////////////////////////////////////////////////////////////
VTYPE runPER( int argc, char** argv, VTYPE v, UINT seed, double * elapsed, VTYPE * others) 
{

	//Initialization

	/*
    //CUDA Random generator initializing
    curandState *devStates;
    CUDA_SAFE_CALL(cudaMalloc((void **)&devStates, no_T *sizeof(curandState)));
    initRNGInvoker( dimGrid, dimBlock, devStates, seed ); // initializing the RNG
    
    rewireInvoker_assignment( dimGrid, dimBlock, n, k, d_adjM, edgePerT, devStates, P , d_orders, NULL , false);
	*/
    float p=P;
    VTYPE pXMaxRND=p*M;
    unsigned long long TotalGeneratedE=0;
    unsigned long long TotalE=pow((double)v,2);
    unsigned long long TotalEP=p*TotalE;
    
    VTYPE NumberofPRNs=TotalE;
    NumberofPRNs/=v/1000;
    
    VTYPE IPT=ceil((double)NumberofPRNs/(numBlocks*numThreadsPerBlock));	//Items Per Thread

    unsigned long long offset=1;

    //Logging
    ofstream Logs;

    VTYPE retval = 0;

    CUDPPConfiguration config;
    config.algorithm = CUDPP_COMPACT;
    config.datatype = CUDPP_UINT;
    config.options = CUDPP_OPTION_FORWARD;
    CUDPPHandle plan;
    CUDPPResult result = CUDPP_SUCCESS;
    result = cudppPlan(&plan, config, NumberofPRNs, 1, 0);

    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error creating plan for Compact\n");
        retval = 1;
        return retval;
    }

    VTYPE memSize = sizeof(VTYPE) * NumberofPRNs;
    //cout<<"Memory="<<memSize/1000000<<endl;
    // allocate host memory to store the input data
    VTYPE* EdgeList;
    unsigned int *h_isValid;


    cutilSafeCall( cudaHostAlloc( (void**)&EdgeList, memSize, 0 ) );
    cutilSafeCall( cudaHostAlloc( (void**)&h_isValid, memSize, 0 ) );

    // allocate device memory input and output arrays
    VTYPE* d_EdgeList     = NULL;
    VTYPE* d_EdgeListC     = NULL;

    VTYPE* d_Valids   = NULL;
    VTYPE* d_numValid  = NULL;

    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_EdgeList, memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_EdgeListC, memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_Valids, sizeof(VTYPE) * NumberofPRNs));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_numValid, sizeof(VTYPE)));

    VTYPE *numValidElements = (VTYPE*)malloc(sizeof(VTYPE));

    CUDA_SAFE_CALL( cudaMemset(d_EdgeList, 0, memSize));
    CUDA_SAFE_CALL( cudaMemset(d_Valids, 0, memSize));

    // define grid and block size
    //cudaDeviceProp deviceProp;
    //cudaGetDeviceProperties(&deviceProp, cutGetMaxGflopsDeviceId());
    //printf("\nmultiprocessors=%d", deviceProp.multiProcessorCount);
    //printf(" cores=%d\n", 8 * deviceProp.multiProcessorCount);

    // Compute number of blocks needed based on array size and desired block size
    if(numThreadsPerBlock==0)
    {
        if(NumberofPRNs<=MAX_Threads)
            numThreadsPerBlock=NumberofPRNs;
        else
        {
            numThreadsPerBlock=MAX_Threads;
            numBlocks=(NumberofPRNs%(MAX_Threads)==0)?NumberofPRNs/(MAX_Threads):(NumberofPRNs/(MAX_Threads))+1;
        }
    }
    cout << "Threads=" << numThreadsPerBlock<< " Blocks=" << numBlocks << endl;
    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);

    //Starting the time
    unsigned int timer1=0,timer2=0,timer3=0,timer4 = 0,timer5=0;
    cutilCheckError( cutCreateTimer( &timer1));
    cutilCheckError( cutCreateTimer( &timer2));
    cutilCheckError( cutCreateTimer( &timer3));
    //Warm UP
    PER_Invoker(dimGrid, dimBlock,seed,d_EdgeList,d_Valids,pXMaxRND,IPT,NumberofPRNs);
    cudaThreadSynchronize();

    VTYPE iters=ceil((double)TotalE/NumberofPRNs);
    
    cout<<"Execute ";
    for(VTYPE it=0;it<iters;it++)
    {
        seed=(unsigned)time(0)/seed;

        cutilCheckError( cutStartTimer( timer1));
        // Execute The Random Generator together with checking

        PER_Invoker(dimGrid, dimBlock,seed,d_EdgeList,d_Valids,pXMaxRND,IPT,NumberofPRNs);
        //cutilSafeCall( cudaMemcpy( EdgeList, d_EdgeList, memSize,cudaMemcpyDeviceToHost) );
        //cutilSafeCall( cudaMemcpy( h_isValid, d_Valids, memSize,cudaMemcpyDeviceToHost) );

        cudaThreadSynchronize();
        
        cutilCheckError( cutStopTimer( timer1));
        //Compact

        cutilCheckError( cutStartTimer( timer2));

        cudppCompact(plan, d_EdgeListC, d_numValid, d_EdgeList, d_Valids,NumberofPRNs);
        cudaThreadSynchronize();

        cutilCheckError( cutStopTimer( timer2));
        //End of Compact

        cutilCheckError( cutStartTimer( timer3));

        // get number of valid elements back to host
        CUDA_SAFE_CALL( cudaMemcpy(numValidElements, d_numValid, sizeof(VTYPE),cudaMemcpyDeviceToHost) );

        // allocate host memory to store the output data

        // copy result from device to host
        CUDA_SAFE_CALL(cudaMemcpy(EdgeList, d_EdgeListC,sizeof(VTYPE) * *numValidElements,cudaMemcpyDeviceToHost));

        cutilCheckError( cutStopTimer( timer3));

        //Check the breaking condition

        TotalGeneratedE+=*numValidElements;

        /*result = cudppDestroyPlan(plan);
        if (result != CUDPP_SUCCESS)
        {
        printf("Error destroying CUDPPPlan for Scan\n");
        }*/


    }//End For



    // A Full execution
    cutilCheckError( cutCreateTimer( &timer4));
    cutilCheckError( cutStartTimer( timer4));
    for(VTYPE it=0;it<iters;it++)
    {

        // Execute The Random Generator together with checking
        PER_Invoker(dimGrid, dimBlock,seed+it*107,d_EdgeList,d_Valids,pXMaxRND,IPT,NumberofPRNs);

        cudppCompact(plan, d_EdgeListC, d_numValid, d_EdgeList, d_Valids,NumberofPRNs);
        // get number of valid elements back to host
        CUDA_SAFE_CALL( cudaMemcpy(numValidElements, d_numValid, sizeof(VTYPE),cudaMemcpyDeviceToHost) );
        CUDA_SAFE_CALL(cudaMemcpy(EdgeList, d_EdgeListC,sizeof(VTYPE) * *numValidElements,cudaMemcpyDeviceToHost));

    }//End For
    cudaThreadSynchronize();

    cutilCheckError( cutStopTimer( timer4));

    // Executing without transfer

    cutilCheckError( cutCreateTimer( &timer5));
    cutilCheckError( cutStartTimer( timer5));
    for(VTYPE it=0;it<iters;it++)
    {

        // Execute The Random Generator together with checking
        PER_Invoker(dimGrid, dimBlock,seed+it*seed,d_EdgeList,d_Valids,pXMaxRND,IPT,NumberofPRNs);

        cudppCompact(plan, d_EdgeListC, d_numValid, d_EdgeList, d_Valids,NumberofPRNs);

    }//End For
    cudaThreadSynchronize();

    cutilCheckError( cutStopTimer( timer5));

    elapsed[0]=cutGetTimerValue( timer1);
    elapsed[1]=cutGetTimerValue( timer2);
    elapsed[2]=cutGetTimerValue( timer3);
    elapsed[3]=cutGetTimerValue( timer4);
    elapsed[4]=cutGetTimerValue( timer5);

    cutilCheckError( cutDeleteTimer( timer1));
    cutilCheckError( cutDeleteTimer( timer2));
    cutilCheckError( cutDeleteTimer( timer3));
    cutilCheckError( cutDeleteTimer( timer4));
    cutilCheckError( cutDeleteTimer( timer5));
    others[1]=NumberofPRNs;

    if(*numValidElements!=0)
    {
        VTYPE i=*numValidElements-1;
        offset=(iters-1)*NumberofPRNs;
        TotalGeneratedE-=*numValidElements;
        while(EdgeList[i--]+offset>TotalE);
        TotalGeneratedE+=i;
        others[1]=NumberofPRNs-i;
    }
    Logs.open("PER.csv",ios_base::app);
    printf("Time = %f (ms)\n", elapsed[0]+elapsed[1]+elapsed[2]);
    others[0]=((TotalEP>TotalGeneratedE)?(TotalEP-TotalGeneratedE):(TotalGeneratedE-TotalEP));

    Logs<<elapsed[4]<<","<<elapsed[3]<<","<<elapsed[0]+elapsed[1]+elapsed[2]<<","<<elapsed[0]<<","<<elapsed[1]<<","<<elapsed[2]<<","<<v<<","<<TotalEP<<","<<TotalGeneratedE<<","<<others[0]<<","<<p<<","<<IPT<<","<<seed<<","<<iters<<endl;
    cout<<"E: Estimated="<<TotalEP<<" Generated="<<TotalGeneratedE<<" Differs="<<others[0]<<" P="<< p << " Seed=" << seed <<endl;
    Logs.close();

    // cleanup memory
    free( EdgeList);
    free( h_isValid);
    cudaFree( d_EdgeList);
    cudaFree( d_Valids);
    cudaFree( d_numValid);
    //cudaThreadExit();
    return retval;

}

////////////////////////////////////////////////////////////////////////////////
//! Run PZER for CUDA
////////////////////////////////////////////////////////////////////////////////
VTYPE runPZER( int argc, char** argv, VTYPE v, UINT seed,double * elapsed, VTYPE * others) 
{
    //Initialization
    /*
    //CUDA Random generator initializing
    curandState *devStates;
    CUDA_SAFE_CALL(cudaMalloc((void **)&devStates, no_T *sizeof(curandState)));
    initRNGInvoker( dimGrid, dimBlock, devStates, seed ); // initializing the RNG
    
    rewireInvoker_assignment( dimGrid, dimBlock, n, k, d_adjM, edgePerT, devStates, P , d_orders, NULL , false);
	*/
    float p=P;
    unsigned long long TotalGeneratedE=0;
    unsigned long long TotalE=pow((double)v,2);
    unsigned long long TotalEP=p*TotalE;
    const int lambda = 3;

    VTYPE STD=sqrt(TotalEP*(1-p));
    VTYPE NumberofPRNs=TotalEP+lambda*STD;
    NumberofPRNs/=v/1000;
    
    VTYPE ItemsPerThread=ceil((double)NumberofPRNs/(numBlocks*numThreadsPerBlock));
    unsigned long long offset=1;

    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_UINT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
    CUDPPResult ScanResult;
    CUDPPHandle scanplan = 0;

    double log1p=0;
    if(p==0){NumberofPRNs=1;ItemsPerThread=1;}

    if(p==0)
        log1p=0;
    else if(p== 1)
        log1p=M;
    else
        log1p=1.0/log(1.0-p);

    double logPmax=log((double)M)*log1p;logPmax++;

    //Logging
    ofstream Logs;

    //PRNG
    VTYPE * Results,* d_Results,i;
    // define grid and block size
    /*
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, cutGetMaxGflopsDeviceId());
    printf("\nNumber of multiprocessors=%d\n", deviceProp.multiProcessorCount);
    printf("Number of cores=%d\n", 8 * deviceProp.multiProcessorCount);*/



    // Compute number of blocks needed based on array size and desired block size
    if(numThreadsPerBlock==0)
    {
        if(NumberofPRNs<=MAX_Threads)
            numThreadsPerBlock=NumberofPRNs;
        else
        {
            numThreadsPerBlock=MAX_Threads;
            numBlocks=(NumberofPRNs%(MAX_Threads)==0)?NumberofPRNs/(MAX_Threads):(NumberofPRNs/(MAX_Threads))+1;
        }
    }
    
    cout << "Threads=" << numThreadsPerBlock<< " Blocks=" << numBlocks << endl;

    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);
    // allocate Host memory
    unsigned int Size_Results = sizeof( VTYPE ) * NumberofPRNs;
    
	cutilSafeCall( cudaHostAlloc( (void**)&Results, Size_Results, 0 ) );


    // allocate Device memory
    cutilSafeCall( cudaMalloc( (void**) &d_Results, Size_Results));

    //Starting the time
    unsigned int timer1=0,timer2=0,timer3=0,timer4 = 0,timer5=0;
    cutilCheckError( cutCreateTimer( &timer1));
    cutilCheckError( cutCreateTimer( &timer2));
    cutilCheckError( cutCreateTimer( &timer3));
    VTYPE iters=0;
    // Execute

    while(1)
    {
        seed=(unsigned)time(0)/seed;
        iters++;
        cutilCheckError( cutStartTimer( timer1));
        // Execute The Random Generator together with computing Skips

        PZER_Invoker(dimGrid, dimBlock,seed,d_Results,ItemsPerThread,log1p,logPmax,offset,NumberofPRNs);
        //cutilSafeCall( cudaMemcpy( Results, d_Results, Size_Results,cudaMemcpyDeviceToHost) );

        //cutilSafeCall( cudaMemcpy( Results, d_Results, Size_Results,cudaMemcpyDeviceToHost) );
        //    VTYPE SUM=0;
        //    for(i=0;i<m;i++)SUM+=Results[i];

        //cutilSafeCall( cudaMemcpy( Results, d_Results, Size_Results,cudaMemcpyDeviceToHost) );

        //Scan
        cudaThreadSynchronize();

        cutilCheckError( cutStopTimer( timer1));
        cutilCheckError( cutStartTimer( timer2));


        ScanResult = cudppPlan(&scanplan, config, NumberofPRNs, 1, 0);  

        if (CUDPP_SUCCESS != ScanResult)
        {
            printf("Error creating CUDPPPlan\n");
            exit(-1);
        }

        // Run the scan
        cudppScan(scanplan, d_Results,d_Results, NumberofPRNs);

        /*ScanResult = cudppDestroyPlan(scanplan);
        if (CUDPP_SUCCESS != ScanResult)
        {
        printf("Error destroying CUDPPPlan\n");
        exit(-1);
        }*/
        cudaThreadSynchronize();

        cutilCheckError( cutStopTimer( timer2));
        cutilCheckError( cutStartTimer( timer3));

        //Read the Edge indexes
        cutilSafeCall( cudaMemcpy( Results, d_Results, Size_Results,cudaMemcpyDeviceToHost) );
        
		offset+=Results[NumberofPRNs-1];
        if(offset > TotalE) break;

        TotalGeneratedE+=NumberofPRNs;
        
        cutilCheckError( cutStopTimer( timer3));
    }//End while


    i=0;
    offset-=Results[NumberofPRNs-1];
    while(Results[i++]+offset<TotalE);
    TotalGeneratedE+=i;

    // Full execution
    offset=1;
    cutilCheckError( cutCreateTimer( &timer4));
    cutilCheckError( cutStartTimer( timer4));
    iters=0;
    while(1)
    {
        iters++;
        PZER_Invoker(dimGrid, dimBlock,seed,d_Results,ItemsPerThread,log1p,logPmax,offset,NumberofPRNs);
        ScanResult = cudppPlan(&scanplan, config, NumberofPRNs, 1, 0);  
        cudppScan(scanplan, d_Results,d_Results, NumberofPRNs);
        //Read the Edge indexes
        cutilSafeCall( cudaMemcpy( Results, d_Results, Size_Results,cudaMemcpyDeviceToHost) );
        offset+=Results[NumberofPRNs-1];
        if(offset > TotalE) break;

    }//End while
    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer4));

    // Executing without transfer

    cutilCheckError( cutCreateTimer( &timer5));
    cutilCheckError( cutStartTimer( timer5));

    for(int it=0;it<iters;it++)
    {
        PZER_Invoker(dimGrid, dimBlock,seed,d_Results,ItemsPerThread,log1p,logPmax,offset,NumberofPRNs);
        ScanResult = cudppPlan(&scanplan, config, NumberofPRNs, 1, 0);  
        cudppScan(scanplan, d_Results,d_Results, NumberofPRNs);
    }//End while
    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer5));

    elapsed[0]=cutGetTimerValue( timer1);
    elapsed[1]=cutGetTimerValue( timer2);
    elapsed[2]=cutGetTimerValue( timer3);
    elapsed[3]=cutGetTimerValue( timer4);
    elapsed[4]=cutGetTimerValue( timer5);

    cutilCheckError( cutDeleteTimer( timer1));
    cutilCheckError( cutDeleteTimer( timer2));
    cutilCheckError( cutDeleteTimer( timer3));
    cutilCheckError( cutDeleteTimer( timer4));
    cutilCheckError( cutDeleteTimer( timer5));


    Logs.open("PZER.csv",ios_base::app);
    printf("Time = %f (ms)\n", elapsed[3]);
    others[0]=((TotalEP>TotalGeneratedE)?(TotalEP-TotalGeneratedE):(TotalGeneratedE-TotalEP));
    others[1]=NumberofPRNs-i;
    Logs<<elapsed[4]<<","<<elapsed[3]<<","<<elapsed[2]<<","<<elapsed[1]<<","<<elapsed[0]<<","<<v<<","<<TotalEP<<","<<TotalGeneratedE<<","<<others[0]<<","<<p<<","<<seed<<","<<iters<<endl;
    cout<<"E: Estimated="<<TotalEP<<" Generated="<<TotalGeneratedE<<" Differs="<<others[0]<<" P="<< p << " Seed=" << seed <<endl;


    Logs.close();
    //Write the graph to a file?

    free( Results);
    CUDA_SAFE_CALL(cudaFree(d_Results));
    return iters;

}

////////////////////////////////////////////////////////////////////////////////
//! Run PPreZER for CUDA
////////////////////////////////////////////////////////////////////////////////
VTYPE runPPreZER( int argc, char** argv, VTYPE v, UINT seed,double * elapsed, VTYPE * others, VTYPE precomp)
{
    //Initialization
	/*
    //CUDA Random generator initializing
    curandState *devStates;
    CUDA_SAFE_CALL(cudaMalloc((void **)&devStates, no_T *sizeof(curandState)));
    initRNGInvoker( dimGrid, dimBlock, devStates, seed ); // initializing the RNG
    
    rewireInvoker_assignment( dimGrid, dimBlock, n, k, d_adjM, edgePerT, devStates, P , d_orders, NULL , false);
	*/
    float p=P;
    unsigned long long TotalGeneratedE=0;
    unsigned long long TotalE=pow((double)v,2);
    unsigned long long TotalEP=p*TotalE;

    VTYPE STD=sqrt(TotalEP*(1-p));
    VTYPE NumberofPRNs=TotalEP+3*STD;
    NumberofPRNs/=v/1000;
    
    VTYPE ItemsPerThread=ceil((double)NumberofPRNs/(numBlocks*numThreadsPerBlock));
    unsigned long long offset=1;

    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_UINT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
    CUDPPResult ScanResult;
    CUDPPHandle scanplan = 0;

    double log1p=0;
    if(p==0){NumberofPRNs=1;ItemsPerThread=1;}

    if(p==0)
        log1p=0;
    else if(p== 1)
        log1p=M;
    else
        log1p=1.0/log(1.0-p);

    double logPmax=log((double)M)*log1p;logPmax++;

    //Precomputing
    double * dcumP=new double[precomp];
    VTYPE * cumP=new VTYPE[precomp];
    for(int i = 0; i < precomp; i++)
    {
        dcumP[i] = pow(1-p, i) * p;
    }
    cumP[0]=dcumP[0]*M;

    for(int i = 1; i < precomp; i++)
    {
        dcumP[i] += dcumP[i-1];
        cumP[i]=dcumP[i]*M;
    }

    //Logging
    ofstream Logs;

    //PRNG
    VTYPE * Results,* d_Results,i;
    // define grid and block size
	/* cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, cutGetMaxGflopsDeviceId());
    printf("\nNumber of multiprocessors=%d\n", deviceProp.multiProcessorCount);
    printf("Number of cores=%d\n", 8 * deviceProp.multiProcessorCount);*/



    // Compute number of blocks needed based on array size and desired block size
    if(numThreadsPerBlock==0)
    {
        if(NumberofPRNs<=MAX_Threads)
            numThreadsPerBlock=NumberofPRNs;
        else
        {
            numThreadsPerBlock=MAX_Threads;
            numBlocks=(NumberofPRNs%(MAX_Threads)==0)?NumberofPRNs/(MAX_Threads):(NumberofPRNs/(MAX_Threads))+1;
        }
    }
    cout << "Threads=" << numThreadsPerBlock<< " Blocks=" << numBlocks << endl;

    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);
    // allocate Host memory
    unsigned int Size_Results = sizeof( VTYPE ) * NumberofPRNs;
    //Results=(VTYPE *)malloc(Size_Results);
    cutilSafeCall( cudaHostAlloc( (void**)&Results, Size_Results, 0 ) );


    // allocate Device memory
    cutilSafeCall( cudaMalloc( (void**) &d_Results, Size_Results));

    //Starting the time
    unsigned int timer1=0,timer2=0,timer3=0,timer4 = 0,timer5=0;
    cutilCheckError( cutCreateTimer( &timer1));
    cutilCheckError( cutCreateTimer( &timer2));
    cutilCheckError( cutCreateTimer( &timer3));
    VTYPE iters=0;

    //Execute
    
    while(1)
    {
        seed=(unsigned)time(0)/seed;

        iters++;
        cutilCheckError( cutStartTimer( timer1));
        // Execute The Random Generator together with computing Skips

        PPreZER_Invoker(dimGrid, dimBlock,seed,d_Results,ItemsPerThread,log1p,logPmax,offset,NumberofPRNs,cumP,precomp);

        //cutilSafeCall( cudaMemcpy( Results, d_Results, Size_Results,cudaMemcpyDeviceToHost) );

        //RandGenWithSkipInvoker(dimGrid, dimBlock,1,d_Results,ItemsPerThread,log1p,logPmax,offset,NumberofPRNs);

        //cutilSafeCall( cudaMemcpy( Results, d_Results, Size_Results,cudaMemcpyDeviceToHost) );
        //    VTYPE SUM=0;
        //    for(i=0;i<m;i++)SUM+=Results[i];

        //cutilSafeCall( cudaMemcpy( Results, d_Results, Size_Results,cudaMemcpyDeviceToHost) );

        //Scan
        cudaThreadSynchronize();

        cutilCheckError( cutStopTimer( timer1));
        cutilCheckError( cutStartTimer( timer2));


        ScanResult = cudppPlan(&scanplan, config, NumberofPRNs, 1, 0);  

        if (CUDPP_SUCCESS != ScanResult)
        {
            printf("Error creating CUDPPPlan\n");
            exit(-1);
        }

        // Run the scan
        cudppScan(scanplan, d_Results,d_Results, NumberofPRNs);

        /*ScanResult = cudppDestroyPlan(scanplan);
        if (CUDPP_SUCCESS != ScanResult)
        {
        printf("Error destroying CUDPPPlan\n");
        exit(-1);
        }*/
        cudaThreadSynchronize();

        cutilCheckError( cutStopTimer( timer2));
        cutilCheckError( cutStartTimer( timer3));

        //Read the Edge indexes
        cutilSafeCall( cudaMemcpy( Results, d_Results, Size_Results,cudaMemcpyDeviceToHost) );


        offset+=Results[NumberofPRNs-1];
        if(offset > TotalE) break;

        TotalGeneratedE+=NumberofPRNs;
        
        cutilCheckError( cutStopTimer( timer3));
    }//End while


    i=0;
    offset-=Results[NumberofPRNs-1];
    while(Results[i++]+offset<TotalE);
    TotalGeneratedE+=i;

    // Full execution
    offset=1;
    cutilCheckError( cutCreateTimer( &timer4));
    cutilCheckError( cutStartTimer( timer4));
    iters=0;
    while(1)
    {
        iters++;
        PPreZER_Invoker(dimGrid, dimBlock,seed,d_Results,ItemsPerThread,log1p,logPmax,offset,NumberofPRNs,cumP,precomp);
        ScanResult = cudppPlan(&scanplan, config, NumberofPRNs, 1, 0);  
        cudppScan(scanplan, d_Results,d_Results, NumberofPRNs);
        //Read the Edge indexes
        cutilSafeCall( cudaMemcpy( Results, d_Results, Size_Results,cudaMemcpyDeviceToHost) );
        offset+=Results[NumberofPRNs-1];
        if(offset > TotalE) break;

    }//End while
    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer4));

    // Executing without transfer

    cutilCheckError( cutCreateTimer( &timer5));
    cutilCheckError( cutStartTimer( timer5));

    for(int it=0;it<iters;it++)
    {
        PPreZER_Invoker(dimGrid, dimBlock,seed,d_Results,ItemsPerThread,log1p,logPmax,offset,NumberofPRNs,cumP,precomp);
        ScanResult = cudppPlan(&scanplan, config, NumberofPRNs, 1, 0);  
        cudppScan(scanplan, d_Results,d_Results, NumberofPRNs);
    }//End while
    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer5));

    elapsed[0]=cutGetTimerValue( timer1);
    elapsed[1]=cutGetTimerValue( timer2);
    elapsed[2]=cutGetTimerValue( timer3);
    elapsed[3]=cutGetTimerValue( timer4);
    elapsed[4]=cutGetTimerValue( timer5);

    cutilCheckError( cutDeleteTimer( timer1));
    cutilCheckError( cutDeleteTimer( timer2));
    cutilCheckError( cutDeleteTimer( timer3));
    cutilCheckError( cutDeleteTimer( timer4));
    cutilCheckError( cutDeleteTimer( timer5));



    Logs.open("PPreZER.csv",ios_base::app);
    printf("Time = %f (ms)\n", elapsed[0]+elapsed[1]+elapsed[2]);
    others[0]=((TotalEP>TotalGeneratedE)?(TotalEP-TotalGeneratedE):(TotalGeneratedE-TotalEP));
    others[1]=NumberofPRNs-i;
    Logs<<"PRE"<<precomp<<","<<elapsed[4]<<","<<elapsed[3]<<","<<elapsed[0]+elapsed[1]+elapsed[2]<<","<<elapsed[0]<<","<<elapsed[1]<<","<<elapsed[2]<<","<<v<<","<<TotalEP<<","<<TotalGeneratedE<<","<<others[0]<<","<<p<<","<<ItemsPerThread<<","<<seed<<","<<iters<<endl;
    cout<<"E: Estimated="<<TotalEP<<" Generated="<<TotalGeneratedE<<" Differs="<<others[0]<<" P="<< p << " Seed=" << seed <<endl;

    
    Logs.close();
    //Write the graph to a file?

    free( Results);
    CUDA_SAFE_CALL(cudaFree(d_Results));
    return iters;

}