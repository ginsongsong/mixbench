/**
 * main-hip-ro.cpp: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <string.h>
#include "lhiputil.h"
#include "mix_kernels_hip.h"
#include "version_info.h"


void init_vector(double *v, size_t datasize){
	for(int i=0; i<(int)datasize; i++)
		v[i] = i;
}




void fma( int VECTOR_SIZE) {
	printf("mixbench-hip/read-only (%s)\n", VERSION_INFO);

	unsigned int datasize = VECTOR_SIZE*sizeof(double);

	hipSetDevice(0);
	StoreDeviceInfo(stdout);

	size_t freeCUDAMem, totalCUDAMem;
	hipMemGetInfo(&freeCUDAMem, &totalCUDAMem);
	printf("Total GPU memory %lu, free %lu\n", totalCUDAMem, freeCUDAMem);
	printf("Buffer size:          %dMB\n", datasize/(1024*1024));

	double *c;
	c = (double*)malloc(datasize);
	init_vector(c, VECTOR_SIZE);

	mixbenchGPU(c, VECTOR_SIZE);

	free(c);

}

int main(int argc, char* argv[]){

for(int j=64; j <= 52480 ; j+=64)
{
   printf("FMA %d\n", j*256);
   fma(256*j);
}
printf("FMA last%d\n", 32*1024*1024);
fma(32*1024*1024); 
return 0;
}
