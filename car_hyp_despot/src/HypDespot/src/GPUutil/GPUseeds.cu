#include <despot/GPUutil/GPUseeds.h>

using namespace std;

namespace despot {

/*
int Dvc_Seeds::num_assigned_seeds_ = 0;
unsigned Dvc_Seeds::root_seed_ = 0;
Dvc_Random Dvc_Seeds::seed_gen_ = Dvc_Random((unsigned) 0);

void Dvc_Seeds::root_seed(unsigned value) {
	root_seed_ = value;
	seed_gen_ = Dvc_Random(root_seed_);
}

unsigned Dvc_Seeds::Next() {
	// return root_seed_ ^ (num_assigned_seeds_++);
	return seed_gen_.NextUnsigned();
}

vector<unsigned> Dvc_Seeds::Next(int n) {
	vector<unsigned> seeds;
	for (int i = 0; i < n; i++)
		seeds.push_back(Next());
	return seeds;
}
*/

} // namespace despot
