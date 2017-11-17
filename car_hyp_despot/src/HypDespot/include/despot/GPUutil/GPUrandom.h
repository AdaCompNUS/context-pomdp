#ifndef GPURANDOM_H
#define GPURANDOM_H

#include <vector>
#include <despot/GPUcore/CudaInclude.h>


namespace despot {

class Dvc_Random {
public:
	//unsigned seed_;

public:
	//static Dvc_Random RANDOM;

	//static void CopyToGPU(Dvc_Random* des, const Random* src)
	//{
	//	;//des->seed_=src->seed_;
	//}

	//DEVICE Dvc_Random(double seed);
	//DEVICE Dvc_Random(unsigned seed);

	//DEVICE unsigned seed();

	static void init(int num_particles);
	static void clear();

	//DEVICE unsigned NextUnsigned();
	DEVICE int NextInt(int n, int i);
	DEVICE int NextInt(int min, int max, int i);

	DEVICE double NextDouble(int i);
	DEVICE double NextDouble(double min, double max, int i);

	DEVICE double NextGaussian(double mean, double delta,int i);

	DEVICE int NextCategory(int num_catogories,const double* category_probs, int i);

	/*template<class T>
	DEVICE T NextElement(const std::vector<T>& vec) {
		return vec[NextInt(vec.size())];
	}*/

	DEVICE static int GetCategory(int num_catogories,const double* category_probs,
		double rand_num);/**/
};
//Dvc_Random Dvc_Random::RANDOM((unsigned) 0);

} // namespace despot

#endif
