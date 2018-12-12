
#define ONEOVERSQRT2PI 1.0 / sqrt(2.0 * M_PI)

#include <errno.h>
#include <sys/stat.h>

#include "ped_pomdp.h"
#include "neural_prior.h"

#undef LOG
#define LOG(lv) \
if (despot::logging::level() < despot::logging::ERROR || despot::logging::level() < lv) ; \
else despot::logging::stream(lv)
#include <despot/util/logging.h>

const char* window_name = "images";

cv::Mat rescale_image(const cv::Mat& image, double downsample_ratio=0.03125){
	logd << "[rescale_image]" << endl;

	Mat result = image;
	Mat dst;
    for (int i=0; i < (int)log2(1.0 / downsample_ratio); i++){
        pyrDown( result, dst, Size( result.cols/2, result.rows/2 ) );
        result = dst;
    }

    return result;
}
void PedNeuralSolverPrior::Init(){
	// DONE: The environment map will be received via ROS topic as the OccupancyGrid type
	//		 Data will be stored in raw_map_ (class member)
	//       In the current stage, just use a randomized raw_map_ to develop your code.
	//		 Init raw_map_ including its properties here
	//	     Refer to python codes: bag_2_hdf5.parse_map_data_from_dict
	//		 (map_dict_entry is the raw OccupancyGrid data)
	cerr << "DEBUG: Initializing Map" << endl;

	map_prop_.downsample_ratio = 0.03125;
	map_prop_.resolution = raw_map_.info.resolution;
	map_prop_.origin = COORD(raw_map_.info.origin.position.x, raw_map_.info.origin.position.y);
	map_prop_.dim = (int)(raw_map_.info.height);
	map_prop_.new_dim = (int)(map_prop_.dim * map_prop_.downsample_ratio);
	map_prop_.map_intensity_scale = 1500.0;

	// DONE: Convert the data in raw_map to your desired image data structure;
	// 		 (like a torch tensor);

	cerr << "DEBUG: Initializing Map image" << endl;

	map_image_ = cv::Mat(map_prop_.dim, map_prop_.dim, CV_32FC1);

	int index = 0;
	for (std::vector<int8_t>::const_reverse_iterator iterator = raw_map_.data.rbegin();
		  iterator != raw_map_.data.rend(); ++iterator) {
		int x = (size_t)(index / map_prop_.dim);
		int y = (size_t)(index % map_prop_.dim);
		assert(*iterator != -1);
		map_image_.at<float>(x,y) = (float)(*iterator);
		index++;
	}

	double minVal, maxVal;
	minMaxLoc(map_image_, &minVal, &maxVal);
	map_prop_.map_intensity = maxVal;

	logi << "Map properties: " << endl;
	logi << "-dim " << map_prop_.dim << endl;
	logi << "-new_dim " << map_prop_.new_dim << endl;
	logi << "-downsample_ratio " << map_prop_.downsample_ratio << endl;
	logi << "-map_intensity " << map_prop_.map_intensity << endl;
	logi << "-map_intensity_scale " << map_prop_.map_intensity_scale << endl;
	logi << "-resolution " << map_prop_.resolution << endl;

	cerr << "DEBUG: Scaling map" << endl;

	rescaled_map_ = rescale_image(map_image_);

	cerr << "DEBUG: Initializing other images" << endl;

	goal_image_ = cv::Mat(map_prop_.dim, map_prop_.dim, CV_32FC1);

	for( int i=0;i<num_hist_channels;i++){
		map_hist_images_.push_back(cv::Mat(map_prop_.dim, map_prop_.dim, CV_32FC1));
		car_hist_images_.push_back(cv::Mat(map_prop_.dim, map_prop_.dim, CV_32FC1));
	}

	cerr << "DEBUG: Initializing tensors" << endl;

	empty_map_tensor_ = at::zeros({map_prop_.new_dim, map_prop_.new_dim}, at::kFloat);
	map_tensor_ = at::zeros({map_prop_.new_dim, map_prop_.new_dim}, at::kFloat);
	goal_tensor = torch::zeros({map_prop_.new_dim, map_prop_.new_dim}, at::kFloat);

	for( int i=0;i<num_hist_channels;i++){
		map_hist_tensor_.push_back(torch::zeros({map_prop_.new_dim, map_prop_.new_dim}, at::kFloat));
		car_hist_tensor_.push_back(torch::zeros({map_prop_.new_dim, map_prop_.new_dim}, at::kFloat));
	}
}
/*
 * Initialize the prior class and the neural networks
 */
PedNeuralSolverPrior::PedNeuralSolverPrior(const DSPOMDP* model,
		WorldModel& world) :
		SolverPrior(model), world_model(world) {
	cerr << "DEBUG: Initializing PedNeuralSolverPrior" << endl;

	action_probs_.resize(model->NumActions());

	// TODO: get num_peds_in_NN from ROS param
	num_peds_in_NN = 20;
	// TODO: get num_hist_channels from ROS param
	num_hist_channels = 4;

	// DONE Declare the neural network as a class member, and load it here


	cerr << "DEBUG: Initializing car shape" << endl;

	// Car geometry
	car_shape = vector<cv::Point3f>({ Point3f(3.6, 0.95, 1), Point3f(-0.8, 0.95, 1), Point3f(-0.8, -0.95, 1), Point3f(3.6, -0.95, 1)});

	map_received = false;
	drive_net = NULL;
}

void PedNeuralSolverPrior::Load_model(std::string path){
	cerr << "DEBUG: Loading model "<< model_file << endl;

	model_file = path;
	// DONE: Pass the model name through ROS params
	drive_net = torch::jit::load(model_file);
	cerr << "DEBUG: Loaded model "<< model_file << endl;

}

COORD PedNeuralSolverPrior::point_to_indices(COORD pos, COORD origin, double resolution, int dim) const{
	logd << "[point_to_indices] " << endl;

	COORD indices = COORD((pos.x - origin.x) / resolution, (pos.y - origin.y) / resolution);
    if (indices.x < 0 || indices.y < 0 ||
    		indices.x > (dim - 1) || indices.y > (dim - 1))
        return COORD(-1, -1);
    return indices;
}

void PedNeuralSolverPrior::add_in_map(cv::Mat map_image, COORD indices, double map_intensity, double map_intensity_scale){

	if (indices.x == -1 || indices.y == -1)
		return;

	logd << "[add_in_map] " << endl;

	map_image.at<float>((int)round(indices.y), (int)round(indices.x)) = map_intensity * map_intensity_scale;

	logd << "[add_in_map] fill entry " << round(indices.x) << " " << round(indices.y) << endl;

}

float radians(float degrees){
	return (degrees * M_PI)/180.0;
}

std::vector<COORD> PedNeuralSolverPrior::get_transformed_car(CarStruct car, COORD origin, double resolution){
    float theta = car.heading_dir; // TODO: validate that python code is using [0, 2pi] as the range
    float x = car.pos.x - origin.x;
    float y = car.pos.y - origin.x;

    logd << "======================theta: " << theta << endl;
    logd << "======================x: " << x << endl;
    logd << "======================y: " << y << endl;

    logd << "original: \n";
	for (int i=0; i < car_shape.size(); i++){
	  logd << car_shape[i].x << " " << car_shape[i].y << endl;
	}
    // rotate and scale the car
    vector<COORD> car_polygon;
    for (int i=0; i < car_shape.size(); i++){
//    	Point3f& original = car_shape[i];
//    	Point3f rotated;
    	vector<Point3f> original, rotated;
    	original.push_back(car_shape[i]);
    	rotated.resize(1);
    	cv::transform(original, rotated,
    			cv::Matx33f(cos(theta), -sin(theta), x, sin(theta), cos(theta), y, 0, 0, 1));
    	car_polygon.push_back(COORD(rotated[0].x / resolution, rotated[0].y / resolution));
    }

	logd << "transformed: \n";
	for (int i=0; i < car_polygon.size(); i++){
		logd << car_polygon[i].x << " " << car_polygon[i].y << endl;
	}
    // TODO: validate the transformation in test_opencv
    return car_polygon;
}

void fill_car_edges(Mat& image, vector<COORD>& points){
	float default_intensity = 1.0;

	logd << "image size "<< image.size[0]<<","<<image.size[0] << endl;

	for (int i=0;i<points.size();i++){
		int r0, c0, r1, c1;
		r0 = round(points[i].x);
		c0 = round(points[i].y);
		if (i+1 < points.size()){
			r1 = round(points[i+1].x);
			c1 = round(points[i+1].y);
		}
		else{
			r1 = round(points[0].x);
			c1 = round(points[0].y);
		}

		logd << "drawing line from "<< r0<<","<<c0<<
				" to " << r1<<","<<c1 << endl;
		cv::line(image, Point(r0,c0), Point(r1,c1), default_intensity);
	}
}


int img_counter = 0;
std::string img_folder="/home/panpan/Unity/DESPOT-Unity/visualize";

void mkdir_safe(std::string dir){
	if (mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
	{
	    if( errno == EEXIST ) {
	       // alredy exists
	    } else {
	       // something else
	        std::cout << "cannot create sessionnamefolder error:" << strerror(errno) << std::endl;
	        throw std::runtime_error( strerror(errno) );
	    }
	}
}

void export_image(Mat& image, string flag){
	mkdir_safe(img_folder);
	std::ostringstream stringStream;
	stringStream << img_folder << "/" << img_counter << "_" << flag << ".jpg";
	std::string img_name = stringStream.str();

	logi << "saving image " << img_name << endl;

	cv::Mat for_save;

	image.convertTo(for_save, CV_8UC3, 255.0);

	imwrite( img_name , for_save);
}

void inc_counter(){
	img_counter ++;
}

void reset_counter(){
	img_counter = 0;
}


void normalize(cv::Mat& image){
	logd << "[normalize_image]" << endl;

	double image_min, image_max;
	cv::minMaxLoc(image, &image_min, &image_max);
    if (image_max > 0)
    	image = image / image_max;
}

void merge_images(cv::Mat& image_src, cv::Mat& image_des){
	logd << "[merge_images]" << endl;

	image_des = cv::max(image_src, image_des);
}

void copy_to_tensor(cv::Mat& image, at::Tensor& tensor){
	logd << "[copy_to_tensor] copying to tensor" << endl;

    for(int i=0; i<image.rows; i++)
        for(int j=0; j<image.cols; j++)
        	tensor[i][j] = image.at<float>(i,j);
}

void PedNeuralSolverPrior::Process_map(cv::Mat& src_image, at::Tensor& des_tensor, string flag){
	logd << "original "<< src_image.size[0] << src_image.size[1] << endl;

	auto rescaled_image = rescale_image(src_image);

	logd << "rescaled to "<< rescaled_image.size[0] << rescaled_image.size[1] << endl;
	normalize(rescaled_image);
	merge_images(rescaled_map_, rescaled_image);
	logd << "after merge "<< rescaled_image.size[0] << rescaled_image.size[1] << endl;
	copy_to_tensor(rescaled_image, des_tensor);
}

void PedNeuralSolverPrior::Process_states(const vector<PomdpState*>& hist_states, const vector<int> hist_ids) {
	// DONE: Create num_history copies of the map image, each for a frame of dynamic environment
	// DONE: Get peds from hist_states and put them into the dynamic maps
	//		 Do this for all num_history time steps
	// 		 Refer to python codes: bag_2_hdf5.process_peds, bag_2_hdf5.get_map_indices, bag_2_hdf5.add_to_ped_map
	//		 get_map_indices converts positions to pixel indices
	//		 add_to_ped_map fills the corresponding entry (with some intensity value)

	const PedPomdp* pomdp_model = static_cast<const PedPomdp*>(model_);
	logi << "Processing states, len=" << hist_states.size() << endl;

	for (int i = 0; i < hist_states.size(); i++) {

		logd << "[Process_states] reseting image for map hist " << i << endl;

		int hist_channel = hist_ids[i];

		// clear data in the dynamic map
		map_hist_images_[hist_channel].setTo(0.0);

		// get the array of pedestrians (of length ModelParams::N_PED_IN)
		if (hist_states[i]){
			logd << "[Process_states] start processing peds for " << hist_states[i] << endl;
//			pomdp_model->PrintState(*hist_states[i]);
			auto& ped_list = hist_states[i]->peds;
			int num_valid_ped = 0;

			logd << "[Process_states] iterating peds in ped_list=" << &ped_list << endl;

			for (int ped_id = 0; ped_id < ModelParams::N_PED_IN; ped_id++) {
				// Process each pedestrian
				PedStruct ped = ped_list[ped_id];
				// get position of the ped
				COORD ped_indices = point_to_indices(ped.pos, map_prop_.origin, map_prop_.resolution, map_prop_.dim);
				if (ped_indices.x == -1 or ped_indices.y == -1) // ped out of map
					continue;
				// put the point in the dynamic map
				add_in_map(map_hist_images_[hist_channel], ped_indices, map_prop_.map_intensity, map_prop_.map_intensity_scale);
			}
		}
	}
	// DONE: Allocate 1 goal image (a tensor)
	// DONE: Get path and fill into the goal image
	//       Refer to python codes: bag_2_hdf5.construct_path_data, bag_2_hdf5.fill_image_with_points
	//	     construct_path_data only calculates the pixel indices
	//       fill_image_with_points fills the entries in the images (with some intensity value)
	if (hist_states.size()==1 || hist_states.size()==4) { // only for current node states

		goal_image_.setTo(0.0);
		Path& path = world_model.path;

		logd << "[Process_states] processing path of size "<< path.size() << endl;

		for (int i = 0; i < path.size(); i++) {
			COORD point = path[i];
			// process each point
			COORD indices = point_to_indices(point, map_prop_.origin, map_prop_.resolution, map_prop_.dim);
			if (indices.x == -1 or indices.y == -1) // path point out of map
				continue;
			// put the point in the goal map
			add_in_map(goal_image_,indices, map_prop_.map_intensity, map_prop_.map_intensity_scale);
		}
	}
	// DONE: Allocate num_history history images, each for a frame of car state
	//		 Refer to python codes: bag_2_hdf5.get_transformed_car, fill_car_edges, fill_image_with_points
	// DONE: get_transformed_car apply the current transformation to the car bounding box
	//	     fill_car_edges fill edges of the car shape with dense points
	//		 fill_image_with_points fills the corresponding entries in the images (with some intensity value)
	for (int i = 0; i < hist_states.size(); i++) {

		logd << "[Process_states] reseting image for car hist " << i << endl;

		int hist_channel = hist_ids[i];
		car_hist_images_[hist_channel].setTo(0.0);

		if (hist_states[i]){
			logd << "[Process_states] processing car for hist " << i << endl;

			CarStruct& car = hist_states[i]->car;
			//     car vertices in its local frame
			//      (-0.8, 0.95)---(3.6, 0.95)
			//      |                       |
			//      |                       |
			//      |                       |
			//      (-0.8, -0.95)--(3.6, 0.95)
			// ...
			vector<COORD> transformed_car = get_transformed_car(car, map_prop_.origin, map_prop_.resolution);
			fill_car_edges(car_hist_images_[hist_channel],transformed_car);
		}
	}

	// DONE: Now we have all the high definition images, scale them down to 32x32
	//		 Refer to python codes bag_2_hdf5.rescale_image
	//		 Dependency: OpenCV

	logd << "[Process_states] re-scaling images to tensor " << endl;

	if (hist_states.size()==1 || hist_states.size()==4){  // only for current node states
		Process_map(goal_image_, goal_tensor, "path");
	}
	for (int i = 0; i < hist_states.size(); i++) {
		int hist_channel = hist_ids[i];
		Process_map(map_hist_images_[hist_channel], map_hist_tensor_[hist_channel], "map_"+std::to_string(i));
		Process_map(car_hist_images_[hist_channel], car_hist_tensor_[hist_channel], "car_"+std::to_string(i));
	}

	logd << "[Process_states] done " << endl;
}

std::vector<torch::Tensor> PedNeuralSolverPrior::Process_nodes_input(const std::vector<State*>& vnode_states){
	logd << "[Process_nodes_input], num=" << vnode_states.size() << endl;

	vector<PomdpState*> cur_state;
	vector<torch::Tensor> output_images;
	cur_state.resize(1);
	vector<int> hist_ids({0}); // use as the last hist step
	for (int i = 0; i < vnode_states.size(); i++){
		cur_state[0]= static_cast<PomdpState*>(vnode_states[i]);

		Process_states(cur_state, hist_ids);

		auto node_nn_input = Combine_images();

		output_images.push_back(node_nn_input);
	}

	return output_images;
}

torch::Tensor PedNeuralSolverPrior::Combine_images(){
	//DONE: Make a tensor with all 9 channels of nn_input_images_ with the 32x32 images here.
			//		 [IMPORTANT] Be cautious on the order of the channels:
			//			config.channel_map = 0  # current step
			//	    	config.channel_map1 = 1  # history steps t-1
			//	    	config.channel_map2 = 2  # history steps t-2
			//	    	config.channel_map3 = 3  # history steps t-3
			//	    	config.channel_goal = 4
			//	    	config.channel_hist1 = 5  # current step
			//	    	config.channel_hist2 = 6  # history steps
			//	    	config.channel_hist3 = 7  # history steps
			//	    	config.channel_hist4 = 8  # history steps

	// DONE: stack them together and return

	logd << "[Combine_images]" << endl;

	logd << "[Combine_images] map_hist_tensor_ len = " << map_hist_tensor_.size() << endl;

	torch::Tensor result;
	for (int i = 0; i < num_hist_channels; i++) {

		logd << "[Combine_images] map_hist_tensor_[" << i << "].unsqueeze(0) dim = "
				<< map_hist_tensor_[i].unsqueeze(0).sizes()<< endl;

		if (i==0)
			result = map_hist_tensor_[i].unsqueeze(0);
		else{

			result = torch::cat({result, map_hist_tensor_[i].unsqueeze(0)}, 0);
			logd << "[Combine_images] result[" << i << "] dim = "
								<< result.sizes()<< endl;
//			result = result.squeeze(0);
//			logd << "[Combine_images] result[" << i << "] dim = "
//								<< result.sizes()<< endl;
		}
	}

	logd << "[Combine_images] goal_tensor dim = "
				<< goal_tensor.sizes()<< endl;

	result = torch::cat({result, goal_tensor.unsqueeze(0)}, 0);

	for (int i = 0; i < num_hist_channels; i++) {
		logd << "[Combine_images] car_hist_tensor_[" << i << "] dim = "
					<< car_hist_tensor_[i].sizes()<< endl;

		result = torch::cat({result, car_hist_tensor_[i].unsqueeze(0)}, 0);
		logd << "[Combine_images] result[" << i << "] dim = "
										<< result.sizes()<< endl;
	}

	return result;
}

at::Tensor gaussian_probability(at::Tensor &sigma, at::Tensor &mu, at::Tensor &data) {
    // data = data.toType(at::kDouble);
    // sigma = sigma.toType(at::kDouble);
    // mu = mu.toType(at::kDouble);
    // data = data.toType(at::kDouble);
    data = data.unsqueeze(1).expand_as(sigma);
//    std::cout << "data=" << data << std::endl;
//    std::cout << "mu=" << mu  << std::endl;
//    std::cout << "sigma=" << sigma  << std::endl;
//    std::cout << "data - mu=" << data - mu  << std::endl;

    auto exponent = -0.5 * at::pow((data - mu) / sigma, at::Scalar(2));
//    std::cout << "exponent=" << exponent << std::endl;
    auto ret = ONEOVERSQRT2PI * (exponent.exp() / sigma);
//    std::cout << "ret=" << ret << std::endl;
    return at::prod(ret, 2);
}

at::Tensor gm_pdf(at::Tensor &pi, at::Tensor &sigma,
    at::Tensor &mu, at::Tensor &target) {
//    std::cout << "pi=" << pi << std::endl;
//    std::cout << "sigma=" << sigma << std::endl;
//    std::cout << "mu=" << mu << std::endl;
//    std::cout << "target=" << target << std::endl;


    auto prob_double = pi * gaussian_probability(sigma, mu, target);
    auto prob_float = prob_double.toType(at::kFloat);
//    std::cout << "prob_float=" << prob_float << std::endl;
    auto safe_sum = at::add(at::sum(prob_float, at::IntList(1)), at::Scalar(0.000001));
    return safe_sum;
}

void Show_params(std::shared_ptr<torch::jit::script::Module> drive_net){
	auto& model_params = drive_net->get_modules();

	int iter = 0;

	ofstream fout;
	string model_param_file = "/home/panpan/NN_params.txt";
	fout.open(model_param_file, std::ios::trunc);
	assert(fout.is_open());

	for(const auto& module: model_params) {
		std::string module_name = module.key.c_str();
		fout << module.key.c_str() << ": " << std::endl;

		auto& module_params = module.value.module->get_parameters();
		fout << module_params.size() << " params found:" << std::endl;
		for (const auto& param: module_params){
			fout<< *param.value.slot() <<std::endl;
		}

//		if (module_name == "ang_head" ){
		fout << "module_name sub_modules: " << std::endl;

		auto& sub_modules = module.value.module->get_modules();
		for(const auto& sub_module: sub_modules) {
			fout << "sub-module found " << sub_module.key.c_str() << ": " << std::endl;

			auto& sub_module_params = sub_module.value.module->get_parameters();
			fout << sub_module_params.size() << " params found:" << std::endl;
			for (const auto& param: sub_module_params){
				fout<< *param.value.slot() <<std::endl;
			}
		}
//		}
		iter ++;
//		if (iter == 20) break;
	}

	fout.close();
}

void PedNeuralSolverPrior::Compute(vector<torch::Tensor>& input_batch, map<OBS_TYPE, despot::VNode*>& vnodes){
	// DONE: Send nn_input_images_ to drive_net, and get the policy and value output
	const PedPomdp* ped_model = static_cast<const PedPomdp*>(model_);

	std::map<OBS_TYPE, despot::VNode* >::iterator it = vnodes.begin();
	logi << "[Compute] node depth " << it->second->depth() << endl;

	if(logging::level()>=4){
		logd << "[Combine_images] vnodes[0]=" << it->second << endl;
		string level = std::to_string(it->second->depth());
		logd << "[Combine_images] exporting images" << endl;

		export_image(goal_image_, "level" + level + "path");
		for (int i = 0; i < num_hist_channels; i++) {
			int hist_channel = i;
			export_image(map_hist_images_[hist_channel],
					"level" + level + "_map_c" + std::to_string(hist_channel));
			export_image(car_hist_images_[hist_channel],
					"level" + level + "_car_c" + std::to_string(hist_channel));
		}
		inc_counter();
	}

	logd << "[Compute] num_nodes = "
						<< input_batch.size()<< endl;

	std::vector<torch::jit::IValue> inputs;

	torch::Tensor input_tensor = input_batch[0].unsqueeze(0);

	logd << "[Compute] input_tensor dim = \n"
						<< input_tensor.sizes()<< endl;

	for (int node_id = 1; node_id< input_batch.size(); node_id++){
		logd << "here\n";
		input_tensor = torch::cat( {input_tensor,input_batch[node_id].unsqueeze(0)}, 0);

		logd << "[Compute] input_tensor dim = "
							<< input_tensor.sizes()<< endl;
	}

	logd << "[Compute] contiguous \n";

	input_tensor= input_tensor.contiguous();

	inputs.push_back(input_tensor);

//    inputs.push_back(torch::rand({1, 9, 32, 32}));

	logi << "[Compute] Query nn for "<< inputs.size() << " tensors of dim" << inputs[0].toTensor().sizes() << endl;

//	logi << "inputs=\n" << inputs[0] << endl;

	assert(drive_net);

//	std::cout << "displaying params\n";

//	Show_params(drive_net);

	logd << "[Compute] Query " << endl;

	auto drive_net_output = drive_net->forward(inputs).toTuple()->elements();

	logd << "[Compute] Refracting outputs " << endl;

	auto value_batch = drive_net_output[VALUE].toTensor();
	auto acc_pi_batch = drive_net_output[ACC_PI].toTensor();
	auto acc_mu_batch = drive_net_output[ACC_MU].toTensor();
	auto acc_sigma_batch = drive_net_output[ACC_SIGMA].toTensor();
	auto ang_batch = drive_net_output[ANG].toTensor();

    auto value_double = value_batch.accessor<float, 2>();

    logd << "[Compute] Updating prior with nn outputs " << endl;

	int node_id = -1;
	for (std::map<OBS_TYPE, despot::VNode* >::iterator it = vnodes.begin();
		        it != vnodes.end(); it++) {
		despot::VNode* vnode = it->second;
		node_id ++;

		auto acc_pi = acc_pi_batch[node_id];
		auto acc_mu = acc_mu_batch[node_id];
		auto acc_sigma = acc_sigma_batch[node_id];
		auto ang = ang_batch[node_id];

		// TODO: Send nn_input_images_ to drive_net, and get the acc distribution (a guassian mixture (pi, sigma, mu))

		int num_accs = 2*ModelParams::NumAcc+1;
		at::Tensor acc_candiates = torch::ones({num_accs, 1}, at::kFloat);
		for (int acc = 0;  acc < num_accs; acc ++){
			acc_candiates[acc][0] = ped_model->GetAcceleration(acc);
		}

		int num_modes = acc_pi.size(0);

		auto acc_pi_actions = acc_pi.unsqueeze(0).expand({num_accs, num_modes});
		auto acc_mu_actions = acc_mu.unsqueeze(0).expand({num_accs, num_modes, 1});
		auto acc_sigma_actions = acc_sigma.unsqueeze(0).expand({num_accs, num_modes, 1});

		auto acc_probs_Tensor = gm_pdf(acc_pi_actions, acc_sigma_actions, acc_mu_actions, acc_candiates);
		logd << "rescaling acc probs" << endl;

        auto acc_sum = acc_probs_Tensor.sum();
        float acc_total_prob = acc_sum.data<float>()[0];
	    acc_probs_Tensor = acc_probs_Tensor / acc_total_prob;

        auto steer_probs_Tensor = at::_softmax(ang, 0, false);

		logd << "acc probs = " << acc_probs_Tensor << endl;
		logd << "steer probs = " << steer_probs_Tensor << endl;

	    auto acc_probs_double = acc_probs_Tensor.accessor<float, 1>();
	    auto steer_probs_double = steer_probs_Tensor.accessor<float, 1>();

		// Update the values in the vnode

		double accum_prob = 0;
		for (int action = 0;  action < ped_model->NumActions(); action ++){
			int acc_ID=(action%((int)(2*ModelParams::NumAcc+1)));
			int steerID = FloorIntRobust(action/(2*ModelParams::NumAcc+1));

			float acc_prob = acc_probs_double[acc_ID];
			float steer_prob = 0;

			if (steerID == 0){
				steer_prob = 0.5 * steer_probs_double[steerID];
			}
			else if (steerID == 2*ModelParams::NumSteerAngle){
				steer_prob = 0.5 * steer_probs_double[steerID-1];
			}
			else{
				steer_prob = 0.5 * steer_probs_double[steerID-1] + 0.5 * steer_probs_double[steerID];
			}

			float joint_prob = acc_prob * steer_prob;
			vnode->prior_action_probs(action, joint_prob);

			accum_prob += joint_prob;

			logd << "action "<< acc_ID << " " << steerID <<
					" joint_prob = " << joint_prob
					<< " accum_prob = " << accum_prob << endl;

			// get the steering prob from angs
		}

		vnode->prior_value(value_double[node_id][0]);
	}
}


void PedNeuralSolverPrior::Process_history(int mode){

	logd << "Processing history, (FULL/PARTIAL) mode=" << mode << endl;

	int num_history = 0;
	vector<int> hist_ids;
	if (mode == FULL){ // Full update of the input channels
		num_history = num_hist_channels;
		for (int i = 0 ; i<num_history ; i++)
			hist_ids.push_back(i);
	}
	else if (mode == PARTIAL){ // Partial update of input channels and reuse old channels
		num_history = num_hist_channels - 1;

		//[1,2,3]
		// id = 0 will be filled in the nodes
		for (int i = 1 ; i< num_hist_channels ; i++)
			hist_ids.push_back(i);
	}

	// DONE: get the 4 latest history states
	vector<PomdpState*> hist_states;
	int latest=as_history_in_search_.Size()-1;
	for (int t = latest; t > latest - num_history && t>=0 ; t--){// process in reserved time order
		PomdpState* car_peds_state = static_cast<PomdpState*>(as_history_in_search_.state(t));
		hist_states.push_back(car_peds_state);
	}

	// fill non-existing older history with empty states
	while (hist_states.size() < num_history){
		hist_states.push_back(NULL);
	}

	Process_states(hist_states, hist_ids);
}

std::vector<torch::Tensor> PedNeuralSolverPrior::Process_history_input(){

	logi << "[Process_history_input], len=" << num_hist_channels << endl;

	int num_history = num_hist_channels;

	// TODO: get the 4 latest history states
	vector<PomdpState*> hist_states;
	int latest=as_history_in_search_.Size()-1;
	for (int t = latest; t > latest - num_history && t>=0 ; t--){// process in reserved time order
		PomdpState* car_peds_state = static_cast<PomdpState*>(as_history_in_search_.state(t));
		hist_states.push_back(car_peds_state);
	}

	logd << hist_states.size() <<" history states found" << endl;

	// fill older history with empty states
	while (hist_states.size() < num_history){
		hist_states.push_back(NULL);
	}

	vector<int> hist_ids;
	for (int i = 0 ; i<num_history ; i++)
		hist_ids.push_back(i);

	for (int i = 0 ; i<num_history ; i++){
		logd << "hist state " << hist_states[i] << " hist_id " << hist_ids[i]<< endl;
	}
	Process_states(hist_states, hist_ids);

	std::vector<torch::Tensor> nn_input;
	nn_input.push_back(Combine_images());
	return nn_input;
}


void PedNeuralSolverPrior::Test_model(string path){
	cout << "[Test_model] Testing model" << endl;

	auto net = torch::jit::load(path);

	std::vector<torch::jit::IValue> inputs;

    inputs.push_back(torch::rand({1, 9, 32, 32}));

	cout << "[Test_model] Query nn for "<< inputs.size() << " tensors of dim" << inputs[0].toTensor().sizes() << endl;

	assert(net);

	cout << "[Test_model] displaying params\n";

	Show_params(net);

	cout << "[Test_model] Query " << endl;

	auto drive_net_output = net->forward(inputs).toTuple()->elements();

	cout << "[Test_model] Done " << endl;

}

///*
// * query the policy network to provide prior probability for PUCT
// */
//const vector<double>& PedNeuralSolverPrior::ComputePreference(){
//
//	// TODO: remove this when you finish the coding
//	throw std::runtime_error( "PedNeuralSolverPrior::ComputePreference hasn't been implemented!" );
//	cerr << "" << endl;
//
//	// TODO: Construct input images
//	Process_history(FULL);
//
//	auto nn_input = Combine_images();
//
//	// TODO: Send nn_input_images_ to drive_net, and get the acc distribution (a guassian mixture (pi, sigma, mu))
//	const PedPomdp* ped_model = static_cast<const PedPomdp*>(model_);
//	for (int action = 0;  action < ped_model->NumActions(); action ++){
//		double accelaration = ped_model->GetAcceleration(action);
//		// TODO: get the probability of acceleration from the Gaussian mixture, and store in action_probs_
//		// Hint: Need to implement Gaussian pdf and calculate mixture
//		// Hint: Refer to Components/mdn.py in the BTS_RL_NN project
//
//	}
//
//	// return the output as vector<double>
//	return action_probs_;
//}
//
///*
// * query the value network to initialize leaf node values
// */
//double PedNeuralSolverPrior::ComputeValue(){
//
//	// TODO: remove this when you finish the coding
//	throw std::runtime_error( "PedNeuralSolverPrior::ComputeValue hasn't been implemented!" );
//
//	// TODO: Construct input images
//	// 		 Here we can reuse existing channels
//	Process_history(PARTIAL);
//
//	// TODO: Send nn_input_images_ to drive_net, and get the value output
//
//	// TODO: return the output as double
//	return 0;
//}
//


