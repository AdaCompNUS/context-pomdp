/*! \file   ThreadId.cpp
	\date   Sunday July 24, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for global thread id utility functions.
*/

#include <despot/GPUutil/GPUThreadId.h>

namespace archaeopteryx
{

namespace util
{

__device__ unsigned int threadId()
{
	unsigned int idX = threadIdx.x;
	unsigned int idY = threadIdx.y * blockDim.x;
	unsigned int idZ = threadIdx.z * blockDim.x * blockDim.y;
	
	unsigned int blockThreads = blockDim.x * blockDim.y * blockDim.z;
	
	unsigned int bidX = blockIdx.x * blockThreads;
	unsigned int bidY = blockIdx.y * gridDim.x * blockThreads;
	unsigned int bidZ = blockIdx.z * gridDim.x * gridDim.y * blockThreads;
	
	return idX + idY + idZ + bidX + bidY + bidZ;
}

}

}

