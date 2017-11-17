#ifndef GPUHISTORY_H
#define GPUHISTORY_H

#include <vector>
#include <despot/util/util.h>
#include <despot/GPUcore/GPUglobals.h>
#include <despot/core/history.h>
#include <despot/GPUcore/CudaInclude.h>

namespace despot {

/**
 * Action-observation history.
 */
class Dvc_History {
public:

	HOST static void InitInGPU(int num_particles, Dvc_History* Dvc_history, int length,void* cuda_stream=NULL);

	HOST static void CopyToGPU(int num_particles,int* particleIDs, Dvc_History* Dvc_history, History* history);
	HOST static void Dvc_Add(Dvc_History* Dvc_history,int action, OBS_TYPE obs, void* cudaStream=NULL);
	HOST static void Dvc_Trunc(Dvc_History* Dvc_history, int size, void* cudaStream=NULL);

	DEVICE void Add(int action, OBS_TYPE obs) {

		if(actions_)actions_[currentSize_]=action;
		if(observations_)observations_[currentSize_]=obs;
		currentSize_++;
	}

	DEVICE void RemoveLast() {
		if(actions_)actions_[currentSize_-1]=-1;
		if(observations_)observations_[currentSize_-1]=-1;
		currentSize_--;
	}

	DEVICE int Action(int t) const {
		return actions_[t];
	}

	DEVICE OBS_TYPE Observation(int t) const {
		return observations_[t];
	}

	DEVICE size_t Size() const {
		return currentSize_;
	}

	DEVICE void Truncate(int d) {
		currentSize_=d;
	}

	DEVICE int LastAction() const {
		return actions_[currentSize_-1];
	}

	DEVICE OBS_TYPE LastObservation() const {
		return observations_[currentSize_-1];
	}

	/*HOST Dvc_History Suffix(int s) const {
		Dvc_History history;
		for (int i = s; i < Size(); i++)
			history.Add(Action(i), Observation(i));
		return history;
	}*/

	/*friend HOST std::ostream& operator<<(std::ostream& os, const Dvc_History& history) {
		for (int i = 0; i < history.Size(); i++)
			os << "(" << history.Action(i) << ", " << history.Observation(i)
				<< ") ";
		return os;
	}*/
/**/
	void CreateMemoryPool(int mode=0) const;
	void DestroyMemoryPool(int mode=0) const;
public:
	int* actions_;
	OBS_TYPE* observations_;
	int currentSize_;
};

} // namespace despot

namespace std {
/*
// NOTE: disabled C++11 feature
template<>
struct hash<Dvc_History> {
	size_t operator()(const Dvc_History& h) const {
		size_t seed = 0;
		for (int i = 0; i < h.Size(); i++) {
			hash_combine(seed, h.Action(i));
			hash_combine(seed, h.Observation(i));
		}
		return seed;
	}
};
*/

/*
template<>
HOST DEVICE
struct less<despot::Dvc_History> {
	bool operator()(const despot::Dvc_History& h1, const despot::Dvc_History& h2) const {
		int N = h1.Size() < h2.Size() ? h1.Size() : h2.Size();

		for (int i = 0; i < N; i++) {
			if (h1.Action(i) < h2.Action(i))
				return true;
			if (h1.Action(i) > h2.Action(i))
				return false;
			if (h1.Observation(i) < h2.Observation(i))
				return true;
			if (h1.Observation(i) > h2.Observation(i))
				return false;
		}
		return false;
	}
};*/

}
#endif
