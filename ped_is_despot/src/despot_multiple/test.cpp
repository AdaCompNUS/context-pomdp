//#include "gamma.h"
//#include "dirichlet.h"
#include "particle.h"
#include <iostream>
#include <vector>

// for testing the Dirichlet distribution

using namespace std;

int main() {
	/*
	int D = 5;
	double alpha = 1;
	Dirichlet dir(vector<double>(D, alpha));

	int N = 1000;
	vector<double> sum(D, 0.0);
	for (int i = 0 ; i < N; i++) {
		vector<double> x = dir.next();
	
		for (int j = 0; j < x.size(); j++ )
			sum[j] += x[j];

		for (int j = 0; j < x.size(); j++ )
			cout << x[j] <<" ,";
		cout  << endl;
	}

	cout << "avg" << endl;
	for(int i=0; i<sum.size(); i++)
		cout << sum[i]/N << ", ";
	cout << endl;
	*/

	vector<Particle<int>*> particles;
	Particle<int>* particle = new Particle<int>(10, 0, 1.0);
	particle->SetAllocated();
	particles.push_back(particle);

	particle = new Particle<int>(12, 1, 1.0);
	particle->SetAllocated();
	particles.push_back(particle);

	cout << particles.size() << " particles " << endl;
	for(auto particle : particles)
		cout << particle->IsAllocated() << endl;

	particles = decltype(particles)(particles.begin(), particles.begin());
	cout << particles.size() << " particles " << endl;
	for(auto particle : particles)
		cout << particle->IsAllocated() << endl;

	vector<Particle<int>*> new_particles = std::move(particles);

	cout << new_particles.size() << " particles " << endl;
	for(auto particle : new_particles)
		cout << particle->IsAllocated() << endl;

	return 0;
}
