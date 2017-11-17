#include "despot/GPUutil/GPUmemorypool.h"


namespace despot {
/*
template<class T>
GPU_MemoryPool<T>::Dvc_Chunk::~Dvc_Chunk()
	{
		HANDLE_ERROR(cudaFree(Objects));
	}
	template<class T>
	GPU_MemoryPool<T>::Dvc_Chunk::Dvc_Chunk()
	{
		cudaMalloc((void**)&Objects, sizeof(T)*Size);
		//HANDLE_ERROR();
	}*/

/*
	template<class T>
	void GPU_MemoryPool<T>::DeleteAll() {
		for (chunk_iterator_ i_chunk = chunks_.begin();
			i_chunk != chunks_.end(); ++i_chunk)
			delete *i_chunk;
		chunks_.clear();
		freehead_=NULL;
		freepos_=NULL;
		num_allocated_ = 0;
	}
*/

	/*template<class T>
	GPU_MemoryPool<T>::~GPU_MemoryPool() {
		DeleteAll();
	}*/
}
