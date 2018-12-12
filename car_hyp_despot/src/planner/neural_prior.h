/*
 * neural_prior.h
 *
 *  Created on: Dec 11, 2018
 *      Author: panpan
 */

#ifndef NEURAL_PRIOR_H_
#define NEURAL_PRIOR_H_

#include <despot/core/prior.h>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

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

using namespace cv;

//#ifndef __CUDACC__

class PedNeuralSolverPrior:public SolverPrior{
	WorldModel& world_model;

	COORD point_to_indices(COORD pos, COORD origin, double resolution, int dim) const;
	void add_in_map(cv::Mat map_tensor, COORD indices, double map_intensity, double map_intensity_scale);
	std::vector<COORD> get_transformed_car(CarStruct car, COORD origin, double resolution);
	void Process_states(const vector<PomdpState*>& hist_states, const vector<int> hist_ids);

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

	cv::Mat map_image_;
	cv::Mat rescaled_map_;
	std::vector<cv::Mat> map_hist_images_;
	std::vector<cv::Mat> car_hist_images_;
	cv::Mat goal_image_;

	vector<cv::Point3f> car_shape;
public:

	PedNeuralSolverPrior(const DSPOMDP* model, WorldModel& world);
//	virtual const std::vector<double>& ComputePreference();
//
//	virtual double ComputeValue();

	void Compute(vector<torch::Tensor>& images, map<OBS_TYPE, despot::VNode*>& vnode);

	void Process_history(int);
	std::vector<torch::Tensor> Process_history_input();

	std::vector<torch::Tensor> Process_nodes_input(const std::vector<State*>& vnode_states);
//	at::Tensor Combine_images(const at::Tensor& node_image, const at::Tensor& hist_images){return torch::zeros({1,1,1});}
	torch::Tensor Combine_images();

public:
	void Load_model(std::string);
	void Init();

	static void Test_model(std::string);
public:
	int num_hist_channels;
	int num_peds_in_NN;
	nav_msgs::OccupancyGrid raw_map_;
	bool map_received;

	map_properties map_prop_;
	std::string model_file;
	std::shared_ptr<torch::jit::script::Module> drive_net;

};
//#endif



#endif /* NEURAL_PRIOR_H_ */
