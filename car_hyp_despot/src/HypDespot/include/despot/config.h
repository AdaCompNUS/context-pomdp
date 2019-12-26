#ifndef CONFIG_H
#define CONFIG_H

#include <string>

namespace despot {

struct Config {
	int search_depth;
	double discount;
	unsigned int root_seed;
	double time_per_move;  // CPU time available to construct the search tree
	int num_scenarios;
	double pruning_constant;
	double xi; // xi * gap(root) is the target uncertainty at the root.
	int sim_len; // Number of steps to run the simulation for.
  std::string default_action;
	int max_policy_sim_len; // Maximum number of steps for simulating the default policy
	double noise;
	bool silence;
	bool useGPU;
	std::string rollout_type;
	int GPUid;
	bool use_multi_thread_;
	int NUM_THREADS;
	int exploration_mode;
	double exploration_constant;
	double exploration_constant_o;
	bool enable_despot_thread;
	bool experiment_mode;
	int despot_thread_gap;
	int expanstion_switch_thresh;
	double time_scale;

	Config() :
		search_depth(90),
		discount(0.95),
		root_seed(42),
		time_per_move(1),
		num_scenarios(500),
		pruning_constant(0),
		xi(0.95),
		sim_len(90),
		default_action(""),
		max_policy_sim_len(90),
		noise(0.1),
		silence(false),
	    useGPU(false),
	    rollout_type("BLIND"),
	    GPUid(0),
	    use_multi_thread_(false),
	    NUM_THREADS(0),
	    exploration_mode(0),
	    exploration_constant(0.3),
	    exploration_constant_o(0.3),
	    enable_despot_thread(false),
	    experiment_mode(false),
		despot_thread_gap(10000000),
		expanstion_switch_thresh(2),
		time_scale(1.0)
	{
		rollout_type = "INDEPENDENT";
	}

	void text(){
		printf("Globals::config:\n");
		printf("=> search_depth=%d\n", search_depth);
		printf("=> discount=%f\n", discount);
		printf("=> root_seed=%d\n", root_seed);
		printf("=> num_scenarios=%d\n", num_scenarios);
		printf("=> time_per_move=%f\n", time_per_move);
		printf("=> pruning_constant=%f\n", pruning_constant);
		printf("=> xi=%f\n", xi);
		printf("=> sim_len=%d\n", sim_len);
		printf("=> default_action=%s\n", default_action.c_str());
		printf("=> max_policy_sim_len=%d\n", max_policy_sim_len);
		printf("=> noise=%f\n", noise);
		printf("=> silence=%d\n", silence);
		printf("=> useGPU=%d\n", useGPU);
		printf("=> rollout_type=%s\n", rollout_type.c_str());
		printf("=> GPUid=%d\n", GPUid);
		printf("=> use_multi_thread_=%d\n", use_multi_thread_);
		printf("=> exploration_constant_o=%f\n", exploration_constant_o);
		printf("=> exploration_constant=%f\n", exploration_constant);
		printf("=> enable_despot_thread=%d\n", enable_despot_thread);
		printf("=> despot_thread_gap=%d\n", despot_thread_gap);
		printf("=> expanstion_switch_thresh=%d\n", expanstion_switch_thresh);
		printf("=> time_scale=%f\n", time_scale);
	}
};

} // namespace despot

#endif
