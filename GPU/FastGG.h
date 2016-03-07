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


#include "cutil.h"
// includes, project
//#include <shrQATest.h>
#include <cutil_inline.h>
#include <curand_kernel.h>
const int MAX_Threads=512;

typedef unsigned int VTYPE;
typedef unsigned int UINT;
#define VTYPE_MAX     ((VTYPE)~((VTYPE)0))
#define WTYPE_MAX     4294967295
const UINT M = 2147483647;//65535;
const double minM = /*-1.7E+308;*/- 2147483647.0 ;//65535;
const UINT MAXITEMS =60000000;//8388608=2^23

typedef VTYPE AdjType ;
#define uv2i(u,v,n)  ( u * n + v - 1 - ( ( u * ( u + 3 ) ) /2 ) )


// declaration, forward
VTYPE runPER( int argc, char** argv, VTYPE v, UINT seed, double * elapsed, VTYPE * others);
VTYPE runPZER( int argc, char** argv, VTYPE v, UINT seed, double * elapsed, VTYPE * others);
VTYPE runPPreZER( int argc, char** argv, VTYPE v, UINT seed, double * elapsed, VTYPE * others, VTYPE precomp);
void parse_args(int argc, char **argv);
void PZER_Invoker( dim3 dimGrid,dim3 dimBlock,UINT Seed,VTYPE * Results,UINT ItemsPerThread, float log1p ,float logPmax,VTYPE offset,VTYPE NumItems);
void PER_Invoker( dim3 dimGrid,dim3 dimBlock,UINT Seed,VTYPE * EdgeList,VTYPE * Valids,VTYPE pXMaxRND,UINT ItemsPerThread,VTYPE NumItems);
void PPreZER_Invoker( dim3 dimGrid,dim3 dimBlock,UINT Seed,VTYPE * Results,UINT ItemsPerThread, float log1p ,float logPmax,VTYPE offset,VTYPE NumItems,VTYPE * pres,VTYPE numpre);


void initRNGInvoker( dim3 dimGrid, dim3 dimBlock, curandState * const rngStates, const unsigned int seed);
