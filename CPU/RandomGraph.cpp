// RandomGraph.cpp : Generate random Erdos-Renyi graphs
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

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <string>
#include <math.h>
#include <time.h>
#include <vector>
#include <bitset>
using namespace std;

#define MAX 2147483647

/****************************************
***      Random Number Generator      ***
****************************************/
unsigned int seed=1;
unsigned int RandomLCG()
{
	unsigned long long const a    = 16807;      //ie 7**5
	const unsigned long long m    = 2147483647; //ie 2**31-1
	unsigned long long temp;
	temp=(unsigned long long)seed * a;
	seed = (temp)%m;
	return seed;
}


/****************************************
***         Usage of the File         ***
****************************************/
int exit_with_help()
{
	cout<<"*******************************************************************************"<<endl;
	cout<<"Usage: ./YOUR_COMPILED_FILE -a ALGORITHM -n NUMBER_OF_VERTICES -p PROBABILITY -o OUTPUT_FILE"<<endl;
	cout<<"Parameters:"<<endl;
	cout<<"a: the algorithm, 1.ER; 2.ZER; 3.PreZER"<<endl;
	cout<<"n: the number of vertices in the generated graph"<<endl;
	cout<<"p: the probability for each edge to be generated"<<endl;
	cout<<"o: the name of the output file"<<endl;
	cout<<"*******************************************************************************"<<endl;
	return 1;
}


int main(int argc, char* argv[])
{
	
	int algorithm = -1;
	unsigned long long number_of_vertices = -1;
	double probability = -1;
	char *file_name;
	
	for(int i = 1; i < argc; i++)
	{
		if(argv[i][0] != '-')
		{
			exit_with_help();
			return -1;
		}

		if(++i >= argc)
		{
			exit_with_help();
			return -1;
		}
			

		switch(argv[i-1][1])
		{
		case 'a':
			algorithm = atoi(argv[i]);
			break;
		case 'n':
			number_of_vertices = atoi(argv[i]);
			break;
		case 'p':
			probability = atof(argv[i]);
			break;
		case 'o':
			file_name = argv[i];
			break;
		default:
			exit_with_help();
			return 0;
		}
	}

	switch(algorithm)
	{
		/**************************
		********    ER    *********
		**************************/
		case 1:
			{
				ofstream fout(file_name);
				seed = (unsigned)time(0);
				for(int i = 0; i < number_of_vertices-1; i++)
				{
					for(int j = i+1; j < number_of_vertices; j++)
					{
						if((double)RandomLCG() / MAX < probability)
						{
							fout<<i<<"\t"<<j<<endl;
						}
					}
				}	
				fout.close();
			}
		break;
		
		/**************************
		********    ZER    ********
		**************************/
		case 2:
			{
				ofstream fout(file_name);
				seed = (unsigned)time(0);
				unsigned long long maximum_edges = number_of_vertices * (number_of_vertices - 1) / 2;
				double denominator = -1;
				if(probability != 0 || probability != 1)
					denominator = 1/log(1-probability);
				double logMax = log((double)MAX);
				unsigned long long skip = -1;
				unsigned long long sv = -1;
				unsigned long long tv = -1;
				unsigned long long square = 4 * number_of_vertices * number_of_vertices - 4 * number_of_vertices - 7;	//used in decoding		
			
				for(unsigned long long i = 0; i < maximum_edges; i++)
				{
					skip = ceil((log((double)RandomLCG())-logMax) * denominator - 1);					//compute the value of skip
					i += skip;		
					if(i >= maximum_edges)
						break;	
					/***********decoding to (source_vertex, target vertex) pair***********/																
					if(i < number_of_vertices - 1)													
					{
						sv = 0;
						tv = i + 1;
					}
					else
					{																			
						sv = ceil((2 * number_of_vertices - 1 - sqrt(square - 8 * i)) / 2 - 1);
						tv = sv + (i + 1 - (2 * number_of_vertices - sv - 1) * sv / 2);
					}																	
					/***********decoding to (source_vertex, target vertex) pair***********/				
					fout<<sv<<"\t"<<tv<<endl;
				}		
				fout.close();
			}
		break;
		
		
		/**************************
		******    PreZER    *******
		**************************/
		case 3:
			{
				ofstream fout(file_name);
				seed = (unsigned)time(0);
				unsigned long long maximum_edges = number_of_vertices * (number_of_vertices - 1) / 2;
				double denominator = -1;
				if(probability != 0 || probability != 1)
					denominator = 1/log(1-probability);
				double logMax = log((double)MAX);
				unsigned long long skip = -1;
				unsigned long long sv = -1;
				unsigned long long tv = -1;
				unsigned long long square = 4 * number_of_vertices * number_of_vertices - 4 * number_of_vertices - 7;	//used in decoding		
				int number_of_precomputing = 10;
				double prob_of_skip[20];
				double cum_prob_of_skip[20];
				double nextDouble = -1;
			
				for(int i = 0; i < number_of_precomputing; i++)
				{
					prob_of_skip[i] = pow(1-probability, i) * probability;
				}
				for(int i = 0; i < number_of_precomputing; i++)
				{
					cum_prob_of_skip[i] = 0;
					for(int j = 0; j <= i; j++)
					{
						cum_prob_of_skip[i] += prob_of_skip[j];
					}
				}
				for(unsigned long long i = 0; i < maximum_edges; i++)
				{
					nextDouble = (double)RandomLCG()/MAX;
					int k;
					for(k = 0; k < number_of_precomputing; k++)
					{
						if(cum_prob_of_skip[k] > nextDouble)
							break;
					}
					if(k < number_of_precomputing)							//compute the value of skip
					{
						skip = k;
					}
					else
					{
						skip = ceil(log(1-nextDouble) * denominator - 1);
					}
					i += skip;
					if(i >= maximum_edges)
						break;	
					/***********decoding to (source_vertex, target vertex) pair***********/																
					if(i < number_of_vertices - 1)													
					{
						sv = 0;
						tv = i + 1;
					}
					else
					{																			
						sv = ceil((2 * number_of_vertices - 1 - sqrt(square - 8 * i)) / 2 - 1);
						tv = sv + (i + 1 - (2 * number_of_vertices - sv - 1) * sv / 2);
					}																	
					/***********decoding to (source_vertex, target vertex) pair***********/		
					fout<<sv<<"\t"<<tv<<"\t"<<i<<endl;
				}
				fout.close();
			}
		break;
		
		default:
			break;
	}




}