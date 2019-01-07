/*
 * neural_prior.h
 *
 *  Created on: Dec 11, 2018
 *      Author: panpan
 */

#ifndef NEURAL_PRIOR_H_
#define NEURAL_PRIOR_H_

#include <despot/core/prior.h>

#include "opencv4/opencv2/imgproc.hpp"
#include "opencv4/opencv2/imgcodecs.hpp"
#include "opencv4/opencv2/highgui.hpp"


#include "disabled_util.h"
#include "despot/interface/pomdp.h"
#include <despot/core/mdp.h>
#include "despot/core/globals.h"
#include "despot/util/coord.h"

//#include "lower_bound.h"
//#include "upper_bound.h"
//#include "string.h"

#include "param.h"
#include "state.h"
#include "WorldModel.h"
#include <cmath>
#include <utility>
#include <string>
#include "math_utils.h"
#include <nav_msgs/OccupancyGrid.h>
#include <vector>
#include "despot/core/node.h"

#include <ros/ros.h>
#include <query_nn/TensorData.h>

using namespace cv;

//class VNode;

//#ifndef __CUDACC__

class PedNeuralSolverPrior:public SolverPrior{
	WorldModel& world_model;
//	cuda::GpuMat gImage;

	COORD point_to_indices(COORD pos, COORD origin, double resolution, int dim) const;
	void add_in_map(cv::Mat map_tensor, COORD indices, double map_intensity, double map_intensity_scale);
	std::vector<COORD> get_transformed_car(const CarStruct car, COORD origin, double resolution);
	void Process_states(std::vector<despot::VNode*> nodes, const vector<PomdpState*>& hist_states, const vector<int> hist_ids);

	void Process_map(cv::Mat& src_image, at::Tensor& des_tensor, std::string flag);

	enum UPDATE_MODES { FULL=0, PARTIAL=1 };

	enum {
		VALUE, ACC_PI, ACC_MU, ACC_SIGMA, ANG, VEL_PI, VEL_MU, VEL_SIGMA,
		                   CAR_VALUE_0, RES_IMAGE_0
	};

	struct map_properties{

		double resolution;
		COORD origin;
		int dim;
		double map_intensity;
		int new_dim; // down sampled resolution

		double map_intensity_scale;
		double downsample_ratio;
	};

private:
	at::Tensor empty_map_tensor_;
	at::Tensor map_tensor_;
	std::vector<at::Tensor> map_hist_tensor_;
	std::vector<at::Tensor> car_hist_tensor_;
	at::Tensor goal_tensor;

	std::vector<const despot::VNode*> map_hist_links;
	std::vector<const despot::VNode*> car_hist_links;
	std::vector<double> hist_time_stamps;

	const despot::VNode* goal_link;

	cv::Mat map_image_;
	cv::Mat rescaled_map_;
	std::vector<cv::Mat> map_hist_images_;
	std::vector<cv::Mat> car_hist_images_;
	cv::Mat goal_image_;

	vector<cv::Point3f> car_shape;

	COORD root_car_pos_;

	std::vector<at::Tensor> map_hist_;
	std::vector<at::Tensor> car_hist_;

	cv::Mat map_hist_image_;
	cv::Mat car_hist_image_;
public:

	COORD root_car_pos(){return root_car_pos_;}

	void root_car_pos(double x, double y){
		root_car_pos_.x = x;
		root_car_pos_.y = y;
	}

	at::Tensor Process_state_to_map_tensor(const State* s);
	at::Tensor Process_state_to_car_tensor(const State* s);
	at::Tensor Process_path_tensor(const State* s);

	at::Tensor last_car_tensor(){
		return car_hist_.back();
	}

	void add_car_tensor(at::Tensor t){
		car_hist_.push_back(t);
	}

	at::Tensor last_map_tensor(){
		return map_hist_.back();
	}

	void add_map_tensor(at::Tensor t){
		map_hist_.push_back(t);
	}

	void Add_tensor_hist(const State* s);
	void Trunc_tensor_hist(int size);
	int Tensor_hist_size();

public:

	PedNeuralSolverPrior(const DSPOMDP* model, WorldModel& world);
//	virtual const std::vector<double>& ComputePreference();
//
//	virtual double ComputeValue();

	void Compute(vector<torch::Tensor>& images, vector<despot::VNode*>& vnode);
	void ComputeMiniBatch(vector<torch::Tensor>& images, vector<despot::VNode*>& vnode);

	void ComputeValue(vector<torch::Tensor>& images, vector<despot::VNode*>& vnode);
	void ComputeMiniBatchValue(vector<torch::Tensor>& images, vector<despot::VNode*>& vnode);

	void ComputePreference(vector<torch::Tensor>& images, vector<despot::VNode*>& vnode);
	void ComputeMiniBatchPref(vector<torch::Tensor>& images, vector<despot::VNode*>& vnode);

	void Process_history(despot::VNode* cur_node, int);
	std::vector<torch::Tensor> Process_history_input(despot::VNode* cur_node);

	std::vector<torch::Tensor> Process_nodes_input(const std::vector<despot::VNode*>& vnodes, const std::vector<State*>& vnode_states);
//	at::Tensor Combine_images(const at::Tensor& node_image, const at::Tensor& hist_images){return torch::zeros({1,1,1});}
	torch::Tensor Combine_images(despot::VNode* cur_node);

	void Reuse_history(int new_channel, int start_channel, int mode);
	void get_history(int mode, despot::VNode* cur_node, std::vector<despot::VNode*>& parents_to_fix_images,
			vector<PomdpState*>& hist_states, vector<int>& hist_ids);

	void Record_hist_len();

	void print_prior_actions(ACT_TYPE);

	float cal_steer_prob(at::TensorAccessor<float, 1> steer_probs_double, int steerID);
public:
	void Load_model(std::string);
	void Load_value_model(std::string);

	void Init();

	void Clear_hist_timestamps();

	void Test_model(std::string);
	void Test_all_srv(int batchsize, int num_guassian_modes, int num_steer_bins);
	void Test_val_srv(int batchsize, int num_guassian_modes, int num_steer_bins);
	void Test_all_libtorch(int batchsize, int num_guassian_modes, int num_steer_bins);
	void Test_val_libtorch(int batchsize, int num_guassian_modes, int num_steer_bins);


public:

	void DebugHistory(string msg);

	VariableActionStateHistory as_history_in_search_recorded;

	void record_cur_history();
	void compare_history_with_recorded();

	void get_history_settings(despot::VNode* cur_node, int mode, int &num_history, int &start_channel);

	void get_history_tensors(int mode, despot::VNode* cur_node);

	void Get_force_steer_action(despot::VNode* vnode, int& opt_act_start, int& opt_act_end);

public:
	int num_hist_channels;
	int num_peds_in_NN;
	nav_msgs::OccupancyGrid raw_map_;
	bool map_received;

	map_properties map_prop_;
	std::string model_file;
	std::string value_model_file;

	std::shared_ptr<torch::jit::script::Module> drive_net;
	std::shared_ptr<torch::jit::script::Module> drive_net_value;

	static ros::ServiceClient nn_client_;
	static ros::ServiceClient nn_client_val_;
};
//#endif



#endif /* NEURAL_PRIOR_H_ */
