#include <despot/GPUutil/GPUrandom.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <curand.h>
#include<curand_kernel.h>


using namespace std;


namespace despot {
#define DIM 128
static DEVICE curandState *devState=NULL;//random number generator states: cuRand library


__global__ void initCurand(unsigned long seed, int num_particles){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx==0)
    	devState=(curandState*)malloc(num_particles*sizeof(curandState));
}

__global__ void initCurand_STEP2(unsigned long seed, int num_particles){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx<num_particles)
    {
    	curand_init(seed, idx, 0, &devState[idx]);
    }
}
__global__ void clearCurand(){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx==0 && devState==NULL)
    {free(devState);devState=NULL;}
}
__global__ void genRandList(curandState *state, float *a, int num_particles){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<num_particles)
    	a[idx] = curand_uniform(&state[idx]);
}


/*DEVICE Dvc_Random::Dvc_Random(double seed) :
	seed_((unsigned) (RAND_MAX * seed)) {
}
DEVICE Dvc_Random::Dvc_Random(unsigned seed) :
	seed_(seed) {
}

DEVICE unsigned Dvc_Random::seed() {
	return seed_;
}*/

/*DEVICE unsigned Dvc_Random::NextUnsigned() {
	return rand_r(&seed_);
}*/


HOST void Dvc_Random::init(int num_particles)
{
	//cout<<__FUNCTION__<<": "<<(num_particles+DIM-1)/DIM<<endl;
	initCurand<<<1,1>>>(1,num_particles);
	HANDLE_ERROR(cudaDeviceSynchronize());
	initCurand_STEP2<<<(num_particles+DIM-1)/DIM,DIM>>>(1,num_particles);
	HANDLE_ERROR(cudaDeviceSynchronize());
}

HOST void Dvc_Random::clear()
{
	clearCurand<<<1,1>>>();
	HANDLE_ERROR(cudaDeviceSynchronize());
}


DEVICE int Dvc_Random::NextInt(int n, int i)//i indicates which curand state to use
{
	// return (int) (n * ((double) rand_r(&seed_) / RAND_MAX));
	return curand(devState+i) % n;
}

DEVICE int Dvc_Random::NextInt(int min, int max, int i) {
	return curand(devState+i) % (max - min) + min;
}

DEVICE double Dvc_Random::NextDouble(double min, double max, int i) {
	return (double) curand_uniform_double(devState+i) * (max - min) + min;
}

DEVICE double Dvc_Random::NextDouble(int i) {
	return (double) curand_uniform_double(devState+i);
}

DEVICE double Dvc_Random::NextGaussian(double mean, double delta, int i) {
	double u = curand_normal_double(devState+i);
	return u*delta+mean;
}

DEVICE int Dvc_Random::NextCategory(int num_catogories, const double * category_probs, int i) {
	return GetCategory(num_catogories,category_probs, NextDouble(i));
}
DEVICE int Dvc_Random::GetCategory(int num_catogories, const double * category_probs, double rand_num) {
	int c = 0;
	double sum = category_probs[0];
	while (sum < rand_num && c<num_catogories) {
		c++;
		sum += category_probs[c];
	}
	return c;
}

} // namespace despot
