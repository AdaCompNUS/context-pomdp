
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

double value_normalizer = 10.0;
const char* window_name = "images";

torch::Device device(torch::kCPU);
bool use_gpu_for_nn = true;

int init_hist_len = 0;

ros::ServiceClient PedNeuralSolverPrior::nn_client_;
ros::ServiceClient PedNeuralSolverPrior::nn_client_val_;

std::chrono::time_point<std::chrono::system_clock> SolverPrior::init_time_;


double SolverPrior::get_timestamp(){
	return Globals::ElapsedTime(init_time_);
}

void SolverPrior::record_init_time(){
	SolverPrior::init_time_ = Time::now();

	logd << " SolverPrior init_time recorded" << endl;
}
cv::Mat rescale_image(const cv::Mat& image, double downsample_ratio=0.03125){
	logd << "[rescale_image]" << endl;
	auto start = Time::now();
	Mat result = image;
	Mat dst;
    for (int i=0; i < (int)log2(1.0 / downsample_ratio); i++){
        pyrDown( result, dst, Size( result.cols/2, result.rows/2 ) );
        result = dst;
    }

    logd << __FUNCTION__<<" rescale " << Globals::ElapsedTime(start) << " s" << endl;
    return result;
}


float radians(float degrees){
	return (degrees * M_PI)/180.0;
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

//	double image_min, image_max;
//	cv::minMaxLoc(image, &image_min, &image_max);
//	logd << "saving image " << img_name << " with min-max values: "
//			<< image_min <<", "<< image_max << endl;

//	cv::Mat for_save;
//
//	image.convertTo(for_save, CV_8UC3, 255.0);
//
//	imwrite( img_name , for_save);

//	imshow( img_name, image );
//
//	char c = (char)waitKey(0);
//
//	cvDestroyWindow((img_name).c_str());
}

void inc_counter(){
	img_counter ++;
}

void reset_counter(){
	img_counter = 0;
}


void normalize(cv::Mat& image){
//	auto start = Time::now();
	logd << "[normalize_image]" << endl;

	double image_min, image_max;
	cv::minMaxLoc(image, &image_min, &image_max);
    if (image_max > 0)
    	image = image / image_max;

//    logd << __FUNCTION__ << " " << Globals::ElapsedTime(start) << " s" << endl;

}

void merge_images(cv::Mat& image_src, cv::Mat& image_des){
//	auto start = Time::now();
	logd << "[merge_images]" << endl;

	image_des = cv::max(image_src, image_des);

//    logd << __FUNCTION__ << " " << Globals::ElapsedTime(start) << " s" << endl;
}

void copy_to_tensor(cv::Mat& image, at::Tensor& tensor){
//	auto start = Time::now();
	logd << "[copy_to_tensor] copying to tensor" << endl;

	tensor = torch::from_blob(image.data, {image.rows, image.cols}, at::kFloat).clone();

	//    for(int i=0; i<image.rows; i++)
//        for(int j=0; j<image.cols; j++)
//        	tensor[i][j] = image.at<float>(i,j);

//    logd << __FUNCTION__ << " " << Globals::ElapsedTime(start) << " s" << endl;

}

void print_full(at::Tensor& tensor, std::string msg){
	logd << "Tensor " << msg << endl;
    auto tensor_double = tensor.accessor<float, 2>();
    for (int i =0;i<tensor.size(0);i++){
    	for (int j =0;j<tensor.size(1);j++){
    		logd << tensor_double[i][j] << " ";
    	}
    	logd << endl;
    }
}

bool is_file_exist(string fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

double value_transform_inverse(double value){
    value = value * value_normalizer;
    return value;
}


at::Tensor gaussian_probability(at::Tensor &sigma, at::Tensor &mu, at::Tensor &data) {
    // data = data.toType(at::kDouble);
    // sigma = sigma.toType(at::kDouble);
    // mu = mu.toType(at::kDouble);
    // data = data.toType(at::kDouble);
    data = data.unsqueeze(1).expand_as(sigma);
//    std::logd << "data=" << data << std::endl;
//    std::logd << "mu=" << mu  << std::endl;
//    std::logd << "sigma=" << sigma  << std::endl;
//    std::logd << "data - mu=" << data - mu  << std::endl;

    auto exponent = -0.5 * at::pow((data - mu) / sigma, at::Scalar(2));
//    std::logd << "exponent=" << exponent << std::endl;
    auto ret = ONEOVERSQRT2PI * (exponent.exp() / sigma);
//    std::logd << "ret=" << ret << std::endl;
    return at::prod(ret, 2);
}

at::Tensor gm_pdf(at::Tensor &pi, at::Tensor &sigma,
    at::Tensor &mu, at::Tensor &target) {
//    std::logd << "pi=" << pi << std::endl;
//    std::logd << "sigma=" << sigma << std::endl;
//    std::logd << "mu=" << mu << std::endl;
//    std::logd << "target=" << target << std::endl;


    auto prob_double = pi * gaussian_probability(sigma, mu, target);
    auto prob_float = prob_double.toType(at::kFloat);
//    std::logd << "prob_float=" << prob_float << std::endl;
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

	logd << "Map properties: " << endl;
	logd << "-dim " << map_prop_.dim << endl;
	logd << "-new_dim " << map_prop_.new_dim << endl;
	logd << "-downsample_ratio " << map_prop_.downsample_ratio << endl;
	logd << "-map_intensity " << map_prop_.map_intensity << endl;
	logd << "-map_intensity_scale " << map_prop_.map_intensity_scale << endl;
	logd << "-resolution " << map_prop_.resolution << endl;

	cerr << "DEBUG: Scaling map" << endl;

	rescaled_map_ = rescale_image(map_image_);
	normalize(rescaled_map_);

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

	logd << "[PedNeuralSolverPrior::Init] create tensors of size " <<map_prop_.new_dim<< ","<<map_prop_.new_dim<< endl;
	for( int i=0;i<num_hist_channels;i++){

		map_hist_tensor_.push_back(torch::zeros({map_prop_.new_dim, map_prop_.new_dim}, at::kFloat));
		car_hist_tensor_.push_back(torch::zeros({map_prop_.new_dim, map_prop_.new_dim}, at::kFloat));

		map_hist_links.push_back(NULL);
		car_hist_links.push_back(NULL);
		hist_time_stamps.push_back(-1);
	}

	goal_link = NULL;

	logd << "[PedNeuralSolverPrior::Init] end " << endl;

}

void PedNeuralSolverPrior::Clear_hist_timestamps(){
	for( int i=0;i<num_hist_channels;i++){
		hist_time_stamps[i]=-1;
	}
}
/*
 * Initialize the prior class and the neural networks
 */
PedNeuralSolverPrior::PedNeuralSolverPrior(const DSPOMDP* model,
		WorldModel& world) :
		SolverPrior(model), world_model(world) {

	prior_id_ = 0;
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


//	cerr << "[" << __FUNCTION__<< "] Testing model start" << endl;
//
//	Test_model("");
//
//	cerr << "[" << __FUNCTION__<< "] Testing model end" << endl;

}

void PedNeuralSolverPrior::Load_model(std::string path){
	auto start = Time::now();

	torch::DeviceType device_type;

	if (torch::cuda::is_available()) {
		std::cout << "CUDA available! NN on GPU" << std::endl;
		device_type = torch::kCUDA;
	} else {
		std::cout << "NN on CPU" << std::endl;
		device_type = torch::kCPU;
	}
	device= torch::Device(device_type);

	model_file = path;

	cerr << "DEBUG: Loading model "<< model_file << endl;

	// DONE: Pass the model name through ROS params
	drive_net = torch::jit::load(model_file);

	assert(drive_net);

	if (use_gpu_for_nn)
		drive_net->to(at::kCUDA);

	cerr << "DEBUG: Loaded model "<< model_file << endl;

	logd << __FUNCTION__<<" " << Globals::ElapsedTime(start) << " s" << endl;
}

void PedNeuralSolverPrior::Load_value_model(std::string path){
	auto start = Time::now();

	torch::DeviceType device_type;

	if (torch::cuda::is_available()) {
		std::cout << "CUDA available! NN on GPU" << std::endl;
		device_type = torch::kCUDA;
	} else {
		std::cout << "NN on CPU" << std::endl;
		device_type = torch::kCPU;
	}
	device= torch::Device(device_type);

	value_model_file = path;

	cerr << "DEBUG: Loading model "<< value_model_file << endl;

	// DONE: Pass the model name through ROS params
	drive_net_value = torch::jit::load(value_model_file);

	assert(drive_net_value);

	if (use_gpu_for_nn)
		drive_net_value->to(at::kCUDA);

	cerr << "DEBUG: Loaded model "<< value_model_file << endl;

	logd << __FUNCTION__<<" " << Globals::ElapsedTime(start) << " s" << endl;
}

COORD PedNeuralSolverPrior::point_to_indices(COORD pos, COORD origin, double resolution, int dim) const{
//	logd << "[point_to_indices] " << endl;

	COORD indices = COORD((pos.x - origin.x) / resolution, (pos.y - origin.y) / resolution);
    if (indices.x < 0 || indices.y < 0 ||
    		indices.x > (dim - 1) || indices.y > (dim - 1))
        return COORD(-1, -1);
    return indices;
}

void PedNeuralSolverPrior::add_in_map(cv::Mat map_image, COORD indices, double map_intensity, double map_intensity_scale){

	if (indices.x == -1 || indices.y == -1)
		return;

//	logd << "[add_in_map] " << endl;

	map_image.at<float>((int)round(indices.y), (int)round(indices.x)) = map_intensity * map_intensity_scale;

//	logd << "[add_in_map] fill entry " << round(indices.x) << " " << round(indices.y) << endl;

}



std::vector<COORD> PedNeuralSolverPrior::get_transformed_car(CarStruct car, COORD origin, double resolution){
	auto start = Time::now();

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

	logd << __FUNCTION__<<" " << Globals::ElapsedTime(start) << " s" << endl;

    return car_polygon;
}

void PedNeuralSolverPrior::Process_map(cv::Mat& src_image, at::Tensor& des_tensor, string flag){
	auto start = Time::now();

	logd << "original "<< src_image.size[0] << src_image.size[1] << endl;
	logd << "flag " <<  flag << endl;

	auto rescaled_image = rescale_image(src_image);
	double image_min, image_max;
//	cv::minMaxLoc(rescaled_image, &image_min, &image_max);
//	logd << "rescaled min, max " <<  image_min << " "<<image_max << endl;
//	logd << __FUNCTION__<<" rescale " << Globals::ElapsedTime(start) << " s" << endl;
//
//	logd << "rescaled to "<< rescaled_image.size[0] << rescaled_image.size[1] << endl;

	//if(flag.find("map") != std::string::npos){
	normalize(rescaled_image);
/*		cv::minMaxLoc(rescaled_image, &image_min, &image_max);
	logd << "normalized min, max " <<  image_min << " "<<image_max << endl;
	logd << __FUNCTION__<<" normalize " << Globals::ElapsedTime(start) << " s" << endl;*/
	//}

	if(flag.find("map") != std::string::npos){

//		cv::minMaxLoc(rescaled_map_, &image_min, &image_max);
//		logd << "raw map min, max " <<  image_min << " "<<image_max << endl;

		merge_images(rescaled_map_, rescaled_image);
		logd << __FUNCTION__<<" merge " << Globals::ElapsedTime(start) << " s" << endl;
//		cv::minMaxLoc(rescaled_image, &image_min, &image_max);
//		logd << "merged min, max " <<  image_min << " "<<image_max << endl;
//		logd << "after merge "<< rescaled_image.size[0] << rescaled_image.size[1] << endl;
	}

	copy_to_tensor(rescaled_image, des_tensor);
	logd << __FUNCTION__<<" copy " << Globals::ElapsedTime(start) << " s" << endl;

	logd << __FUNCTION__<<" " << Globals::ElapsedTime(start) << " s" << endl;

	if(logging::level()>=4){
//		cout << flag+"tensor=\n" << des_tensor << endl;

//		logd << "[Process_maps] des_tensor for " + flag + ": "<< endl;
//		print_full( des_tensor);

		logd << "[Process_maps] des_tensor address "<< &des_tensor << endl;

		export_image(rescaled_image, "Process_"+flag);
		inc_counter();
	}
}

void PedNeuralSolverPrior::Reuse_history(int new_channel, int start_channel, int mode){

	logd << "[Reuse_history] copying data to channel "<< new_channel << endl;
	assert(new_channel>=start_channel);
	int old_channel = new_channel - start_channel;

	logd << "[Reuse_history] copying data from "<< old_channel << " to new channel "<< new_channel << endl;

	if(new_channel !=old_channel){
//		if (mode == FULL){
//
//			map_hist_images_[new_channel] = map_hist_images_[old_channel];
//			car_hist_images_[new_channel] = car_hist_images_[old_channel];
//		}

		map_hist_tensor_[new_channel] = map_hist_tensor_[old_channel];
		car_hist_tensor_[new_channel] = car_hist_tensor_[old_channel];

		map_hist_links[new_channel] = map_hist_links[old_channel];
		car_hist_links[new_channel] = car_hist_links[old_channel];

		hist_time_stamps[new_channel] = hist_time_stamps[old_channel];
	}
	else{
		logd << "skipped" << endl;
	}
}

void PedNeuralSolverPrior::Process_states(std::vector<despot::VNode*> nodes, const vector<PomdpState*>& hist_states, const vector<int> hist_ids) {
	// DONE: Create num_history copies of the map image, each for a frame of dynamic environment
	// DONE: Get peds from hist_states and put them into the dynamic maps
	//		 Do this for all num_history time steps
	// 		 Refer to python codes: bag_2_hdf5.process_peds, bag_2_hdf5.get_map_indices, bag_2_hdf5.add_to_ped_map
	//		 get_map_indices converts positions to pixel indices
	//		 add_to_ped_map fills the corresponding entry (with some intensity value)

	if (hist_states.size()==0){
		logd << "[Process_states] skipping empty state list" << endl;
		return;
	}
	auto start_total = Time::now();

	auto start = Time::now();

	const PedPomdp* pomdp_model = static_cast<const PedPomdp*>(model_);
	logd << "Processing states, len=" << hist_states.size() << endl;

	for (int i = 0; i < hist_states.size(); i++) {

		int hist_channel = hist_ids[i];

		logd << "[Process_states] reseting image for map hist " << hist_channel << endl;

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
				map_hist_images_[hist_channel].at<float>((int)round(ped_indices.y), (int)round(ped_indices.x))
						= map_prop_.map_intensity * map_prop_.map_intensity_scale;
			}
		}
	}

	logd << __FUNCTION__<<" Process ped images: " << Globals::ElapsedTime(start) << " s" << endl;
	start = Time::now();

	// DONE: Allocate 1 goal image (a tensor)
	// DONE: Get path and fill into the goal image
	//       Refer to python codes: bag_2_hdf5.construct_path_data, bag_2_hdf5.fill_image_with_points
	//	     construct_path_data only calculates the pixel indices
	//       fill_image_with_points fills the entries in the images (with some intensity value)
	if (hist_states.size()==1 || hist_states.size()==4) { // only for current node states

		goal_image_.setTo(0.0);

		// get distance between cur car pos and car pos at root node
		auto& cur_car_pos = hist_states[0]->car.pos;
		float trav_dist_since_root = (cur_car_pos - root_car_pos_).Length();

		logd << "[Process_states] processing path of size "<< world_model.path.size() << endl;

		// remove points in path according to the car moving distance
		Path path = world_model.path.copy_without_travelled_points(trav_dist_since_root);

		// use the trimmed path for the goal image
		logd << "[Process_states] after processing: path size "<< path.size() << endl;

		for (int i = 0; i < path.size(); i++) {
			COORD point = path[i];
			// process each point
			COORD indices = point_to_indices(point, map_prop_.origin, map_prop_.resolution, map_prop_.dim);
			if (indices.x == -1 or indices.y == -1) // path point out of map
				continue;
			// put the point in the goal map
			goal_image_.at<float>((int)round(indices.y), (int)round(indices.x)) = 1.0 * 1.0;
		}
	}

	logd << __FUNCTION__<<" Process Goal Image: " << Globals::ElapsedTime(start) << " s" << endl;
	start = Time::now();

	// DONE: Allocate num_history history images, each for a frame of car state
	//		 Refer to python codes: bag_2_hdf5.get_transformed_car, fill_car_edges, fill_image_with_points
	// DONE: get_transformed_car apply the current transformation to the car bounding box
	//	     fill_car_edges fill edges of the car shape with dense points
	//		 fill_image_with_points fills the corresponding entries in the images (with some intensity value)
	for (int i = 0; i < hist_states.size(); i++) {
		int hist_channel = hist_ids[i];
		car_hist_images_[hist_channel].setTo(0.0);

		logd << "[Process_states] reseting image for car hist " << hist_channel << endl;

		if (hist_states[i]){
			logd << "[Process_states] processing car for hist " << hist_channel << endl;

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

	logd << __FUNCTION__<<" Process Car Images: " << Globals::ElapsedTime(start) << " s" << endl;
	start = Time::now();

	// DONE: Now we have all the high definition images, scale them down to 32x32
	//		 Refer to python codes bag_2_hdf5.rescale_image
	//		 Dependency: OpenCV

	logd << "[Process_states] re-scaling images to tensor " << endl;

	if (hist_states.size()==1 || hist_states.size()==4){  // only for current node states
		Process_map(goal_image_, goal_tensor, "path");
		goal_link = nodes[0];

//		logd << "[Process_states]  goal_tensor, address " << &goal_tensor<< endl;
//		print_full( goal_tensor);
	}
	for (int i = 0; i < hist_states.size(); i++) {
		int hist_channel = hist_ids[i];

		logd << "[Process_states] create new data for channel " << hist_channel <<" by node "<< nodes[i]<< endl;

		Process_map(map_hist_images_[hist_channel], map_hist_tensor_[hist_channel], "map_"+std::to_string(hist_channel));

//		logd << "[Process_states]  map_hist_tensor_["<<hist_channel<<"], address " << &map_hist_tensor_[hist_channel]<< endl;
//		print_full( map_hist_tensor_[hist_channel]);

		Process_map(car_hist_images_[hist_channel], car_hist_tensor_[hist_channel], "car_"+std::to_string(hist_channel));

//		logd << "[Process_states]  car_hist_tensor_["<<hist_channel<<"], address " << &car_hist_tensor_[hist_channel]<< endl;
//		print_full( car_hist_tensor_[hist_channel]);

		map_hist_links[hist_channel] = nodes[i];
		car_hist_links[hist_channel] = nodes[i];
		hist_time_stamps[hist_channel] = hist_states[i]->time_stamp;


//		if (nodes[i]->depth()==0 && hist_ids[i] > 0){
//			Debug_state(hist_states[i], "Process states", model_);
//		}
	}

//	logd << "[Process_states]  map_hist_tensor_["<<0<<"], address " << &map_hist_tensor_[0]<< endl;
//	print_full( map_hist_tensor_[0]);

	logd << __FUNCTION__<<" Scale Images: " << Globals::ElapsedTime(start) << " s" << endl;

	logd << "[Process_states] done " << endl;

	logd << __FUNCTION__<<" " << Globals::ElapsedTime(start_total) << " s" << endl;

}

std::vector<torch::Tensor> PedNeuralSolverPrior::Process_nodes_input(
		const std::vector<despot::VNode*>& vnodes, const std::vector<State*>& vnode_states){
	auto start = Time::now();

	logd << "[Process_nodes_input], num=" << vnode_states.size() << endl;

	vector<PomdpState*> cur_state;
	vector<torch::Tensor> output_images;
	cur_state.resize(1);
	vector<int> hist_ids({0}); // use as the last hist step
	for (int i = 0; i < vnode_states.size(); i++){

		logd << "[Process_nodes_input] node " << i << endl;
		cur_state[0]= static_cast<PomdpState*>(vnode_states[i]);

		Globals::lock_process();
		logd << "Thread " << prior_id() << " Using current state of node " << i << " as channel 0" << endl;
		Globals::unlock_process();

		Process_states(vector<despot::VNode*>({vnodes[i]}), cur_state, hist_ids);

		auto node_nn_input = Combine_images(vnodes[i]);

		output_images.push_back(node_nn_input);
	}

	logd << __FUNCTION__<<" " << Globals::ElapsedTime(start) << " s" << endl;

	return output_images;
}

torch::Tensor PedNeuralSolverPrior::Combine_images(despot::VNode* cur_node){
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
	auto start = Time::now();

	despot::VNode* parent = cur_node;

	try{
		logd << __FUNCTION__ << " node depth " << cur_node->depth()<< endl;

		for (int i=0;i<num_hist_channels; i++){
			logd << "channel " << i << " ts " << hist_time_stamps[i] <<
					" linked node depth " <<  car_hist_links[i]->depth() << endl;
		}

		for (int i=0;i<num_hist_channels; i++){
			logd << " [Combine_images] hist " << i << " should be depth " << parent->depth() <<
					", get depth "<< car_hist_links[i]->depth()<< endl;

			if(car_hist_links[i] != parent && car_hist_links[i] != cur_node){
				raise(SIGABRT);
			}

			parent = (parent->parent()==NULL)?
					parent: parent->parent()->parent(); // to handle root node
		}
	}catch (Exception e) {
		logd << " [error] !!!!!!!!!!!!!!!!"  << e.what() << endl;
		raise(SIGABRT);
	}

	logd << "[Compute] validating history, node level " << cur_node->depth() << endl;

	logd << "[Combine_images] map_hist_tensor_ len = " << map_hist_tensor_.size() << endl;

//	cout << "[Combine] map channel 0 "<< endl;
//	print_full(map_hist_tensor_[0], "map 0");
//	print_full(map_hist_tensor_[1], "map 1");
//	print_full(goal_tensor, "goal");
//	print_full(car_hist_tensor_[0], "car 0");
//	print_full(car_hist_tensor_[1], "car 1");

	auto combined = map_hist_tensor_;
	combined.push_back(goal_tensor);
	combined.insert(combined.end(), car_hist_tensor_.begin(), car_hist_tensor_.end());

	torch::Tensor result;
	result = torch::stack(combined, 0);

	logd << __FUNCTION__<<" " << Globals::ElapsedTime(start) << " s" << endl;

	return result;
}

int BatchSize = 128;

float PedNeuralSolverPrior::cal_steer_prob(at::TensorAccessor<float, 1> steer_probs_double, int steerID){

	float smoothing = 1.0;
	float steer_prob = smoothing/2.0/ModelParams::NumSteerAngle;

	if (steerID == 0){
		steer_prob += 0.5 * steer_probs_double[steerID];
	}
	else if (steerID == 2*ModelParams::NumSteerAngle){
		steer_prob += 0.5 * steer_probs_double[steerID-1];
	}
	else{
		steer_prob += 0.5 * steer_probs_double[steerID-1] + 0.5 * steer_probs_double[steerID];
	}

	return steer_prob;
}

void PedNeuralSolverPrior::Compute(vector<torch::Tensor>& input_batch, vector<despot::VNode*>& vnodes){
	logd << "[Compute] get " << vnodes.size() << " nodes" << endl;
	auto start = Time::now();

	if (vnodes.size()>BatchSize){
		logd << "Executing " << (vnodes.size()-1)/BatchSize << " batches" << endl;
		for (int batch = 0; batch < (vnodes.size()-1)/BatchSize + 1; batch++) {

			int start = batch*BatchSize;
			int end = ((batch+1) * BatchSize <= vnodes.size()) ?
					(batch+1) * BatchSize : vnodes.size();

			logd << "start end: " << start << " " << end << endl;

			std::vector<torch::Tensor> mini_batch(&input_batch[start],&input_batch[end]);
			std::vector<despot::VNode*> sub_vnodes(&vnodes[batch*BatchSize],&vnodes[end]);
			ComputeMiniBatch(mini_batch, sub_vnodes);
		}
	}
	else
		ComputeMiniBatch(input_batch, vnodes);

	logd << __FUNCTION__<< " " << vnodes.size() << " nodes "  << Globals::ElapsedTime(start) << " s" << endl;
}

#include "GPU_Car_Drive/GPU_Init.h"

void PedNeuralSolverPrior::ComputeMiniBatch(vector<torch::Tensor>& input_batch, vector<despot::VNode*>& vnodes){

	auto start = Time::now();

	torch::NoGradGuard no_grad;
//	drive_net->eval();

	// DONE: Send nn_input_images_ to drive_net, and get the policy and value output
	const PedPomdp* ped_model = static_cast<const PedPomdp*>(model_);

	logd << "[Compute] node depth " << vnodes[0]->depth() << endl;

	if(logging::level()>=4){
		logd << "[Combine_images] vnodes[0]=" << vnodes[0]->depth() << endl;
		string level = std::to_string(vnodes[0]->depth());
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

	torch::Tensor input_tensor;

	input_tensor = torch::stack(input_batch, 0);

	logd << "[Compute] input_tensor dim = \n"
							<< input_tensor.sizes()<< endl;

	input_tensor= input_tensor.contiguous();

	inputs.push_back(input_tensor.to(at::kCUDA));

	logd << "[Compute] contiguous cuda tensor \n" <<
			input_tensor<< endl;

	logd << __FUNCTION__<<" prepare data " << Globals::ElapsedTime(start) << " s" << endl;

	logd << __FUNCTION__<<" query for "<< input_tensor.sizes() << " data " << endl;

	sync_cuda();

	auto start1 = Time::now();

	auto drive_net_output = drive_net->forward(inputs).toTuple()->elements();


	logd << "[Compute] Refracting outputs " << endl;

	auto value_batch = drive_net_output[VALUE].toTensor().cpu();
	auto acc_pi_batch = drive_net_output[ACC_PI].toTensor().cpu();
	auto acc_mu_batch = drive_net_output[ACC_MU].toTensor().cpu();
	auto acc_sigma_batch = drive_net_output[ACC_SIGMA].toTensor().cpu();
	auto ang_batch = drive_net_output[ANG].toTensor().cpu();

	value_batch = value_batch.squeeze(1);
    auto value_double = value_batch.accessor<float, 1>();

	logd << "nn query in " << Globals::ElapsedTime(start1) << " s" << endl;

	logd << "Get value output " << value_batch << endl;

    logd << "[Compute] Updating prior with nn outputs " << endl;

	int node_id = -1;
	for (std::vector<despot::VNode* >::iterator it = vnodes.begin();
		        it != vnodes.end(); it++) {
		despot::VNode* vnode = *it;
		node_id ++;

		auto acc_pi = acc_pi_batch[node_id];
		auto acc_mu = acc_mu_batch[node_id];
		auto acc_sigma = acc_sigma_batch[node_id];
		auto ang = ang_batch[node_id];

		if (logging::level()>=3){
			logd << "large net raw acc output:" << endl;
			auto acc_pi_float = acc_pi.accessor<float, 1>();
			auto acc_mu_float = acc_mu.accessor<float, 2>();

			for (int mode = 0; mode < acc_pi.size(0); mode ++){
				logd << "mu[" << mode <<"]=" << acc_mu_float[mode][0] << endl;
			}
			for (int mode = 0; mode < acc_pi.size(0); mode ++){
				logd << "pi[" << mode <<"]=" << acc_pi_float[mode] << endl;
			}
		}

		// TODO: Send nn_input_images_ to drive_net, and get the acc distribution (a guassian mixture (pi, sigma, mu))

		int num_accs = 2*ModelParams::NumAcc+1;
		at::Tensor acc_candiates = torch::ones({num_accs, 1}, at::kFloat);
		for (int acc_id = 0;  acc_id < num_accs; acc_id ++){
			double query_acc = ped_model->GetAccelerationNoramlized(acc_id);
			acc_candiates[acc_id][0] = query_acc;
			logd << "adding query acc: "<< acc_id <<"="<< query_acc << endl;
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

        assert(2*ModelParams::NumSteerAngle == steer_probs_Tensor.size(0));

		logd << "acc probs = " << acc_probs_Tensor << endl;
		logd << "steer probs = " << steer_probs_Tensor << endl;

	    auto acc_probs_double = acc_probs_Tensor.accessor<float, 1>();

	    if (logging::level()>=3){
	    	logd << "printing acc probs, acc_probs_Tensor dim=" << acc_probs_Tensor.sizes() << endl;
	    	for (int acc_id =0; acc_id < acc_probs_Tensor.sizes()[0]; acc_id++){
				double query_acc = ped_model->GetAccelerationNoramlized(acc_id);
				logd << "query acc "<< acc_id <<"=" <<  query_acc << ", prob=" << acc_probs_double[acc_id] << endl;
			}
	    }

	    auto steer_probs_double = steer_probs_Tensor.accessor<float, 1>();

		// Update the values in the vnode

		double accum_prob = 0;
		for (int action = 0;  action < ped_model->NumActions(); action ++){
			int acc_ID=ped_model->GetAccelerationID(action);
			int steerID = ped_model->GetSteeringID(action);

			float acc_prob = acc_probs_double[acc_ID];
			float steer_prob = cal_steer_prob(steer_probs_double, steerID);

			float joint_prob = acc_prob * steer_prob;
			vnode->prior_action_probs(action, joint_prob);

			accum_prob += joint_prob;

			logd << "action "<< acc_ID << " " << steerID <<
					" joint_prob = " << joint_prob
					<< " accum_prob = " << accum_prob << endl;

			// get the steering prob from angs
		}

		double prior_value = value_transform_inverse(value_double[node_id]);

		logd << "assigning vnode " << vnode << " value " << prior_value << endl;

		vnode->prior_value(prior_value);

		vnode->prior_initialized(true);
	}

	logd << __FUNCTION__<<" " << Globals::ElapsedTime(start) << " s" << endl;
}

void PedNeuralSolverPrior::ComputePreference(vector<torch::Tensor>& input_batch, vector<despot::VNode*>& vnodes){
	logd << "[ComputePreference] get " << vnodes.size() << " nodes" << endl;
	auto start = Time::now();

	if (vnodes.size()>BatchSize){
		logd << "Executing " << (vnodes.size()-1)/BatchSize << " batches" << endl;
		for (int batch = 0; batch < (vnodes.size()-1)/BatchSize + 1; batch++) {

			int start = batch*BatchSize;
			int end = ((batch+1) * BatchSize <= vnodes.size()) ?
					(batch+1) * BatchSize : vnodes.size();

			logd << "start end: " << start << " " << end << endl;

			std::vector<torch::Tensor> mini_batch(&input_batch[start],&input_batch[end]);
			std::vector<despot::VNode*> sub_vnodes(&vnodes[batch*BatchSize],&vnodes[end]);
			ComputeMiniBatchPref(mini_batch, sub_vnodes);
		}
	}
	else
		ComputeMiniBatchPref(input_batch, vnodes);

	logd << __FUNCTION__<< " " << vnodes.size() << " nodes "  << Globals::ElapsedTime(start) << " s" << endl;
}

void PedNeuralSolverPrior::ComputeMiniBatchPref(vector<torch::Tensor>& input_batch, vector<despot::VNode*>& vnodes){

	auto start = Time::now();

	torch::NoGradGuard no_grad;
//	drive_net->eval();

	// DONE: Send nn_input_images_ to drive_net, and get the policy and value output
	const PedPomdp* ped_model = static_cast<const PedPomdp*>(model_);

	logd << "[ComputeMiniBatchPref] node depth " << vnodes[0]->depth() << endl;

	if(logging::level()>=4){
		logd << "[ComputeMiniBatchPref] vnodes[0]=" << vnodes[0]->depth() << endl;
		string level = std::to_string(vnodes[0]->depth());
		logd << "[ComputeMiniBatchPref] exporting images" << endl;

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

	logd << "[ComputeMiniBatchPref] num_nodes = "
						<< input_batch.size()<< endl;

	std::vector<torch::jit::IValue> inputs;

	torch::Tensor input_tensor;

	input_tensor = torch::stack(input_batch, 0);

	logd << "[ComputeMiniBatchPref] input_tensor dim = \n"
							<< input_tensor.sizes()<< endl;

	input_tensor= input_tensor.contiguous();

	inputs.push_back(input_tensor.to(at::kCUDA));

	logd << "[ComputeMiniBatchPref] contiguous cuda tensor \n" <<
			input_tensor<< endl;

	logd << __FUNCTION__<<" prepare data " << Globals::ElapsedTime(start) << " s" << endl;

	logd << __FUNCTION__<<" query for "<< input_tensor.sizes() << " data " << endl;

	sync_cuda();

	auto start1 = Time::now();

	auto drive_net_output = drive_net->forward(inputs).toTuple()->elements();


	logd << "[ComputeMiniBatchPref] Refracting outputs " << endl;

//	auto value_batch = drive_net_output[VALUE].toTensor().cpu();
	auto acc_pi_batch = drive_net_output[ACC_PI].toTensor().cpu();
	auto acc_mu_batch = drive_net_output[ACC_MU].toTensor().cpu();
	auto acc_sigma_batch = drive_net_output[ACC_SIGMA].toTensor().cpu();
	auto ang_batch = drive_net_output[ANG].toTensor().cpu();

//	value_batch = value_batch.squeeze(1);
//    auto value_double = value_batch.accessor<float, 1>();

	logd << "nn query in " << Globals::ElapsedTime(start1) << " s" << endl;

//	logd << "Get value output " << value_batch << endl;

    logd << "[ComputeMiniBatchPref] Updating prior with nn outputs " << endl;

	int node_id = -1;
	for (std::vector<despot::VNode* >::iterator it = vnodes.begin();
		        it != vnodes.end(); it++) {
		despot::VNode* vnode = *it;
		node_id ++;

		auto acc_pi = acc_pi_batch[node_id];
		auto acc_mu = acc_mu_batch[node_id];
		auto acc_sigma = acc_sigma_batch[node_id];
		auto ang = ang_batch[node_id];

		// TODO: Send nn_input_images_ to drive_net, and get the acc distribution (a guassian mixture (pi, sigma, mu))

		int num_accs = 2*ModelParams::NumAcc+1;
		at::Tensor acc_candiates = torch::ones({num_accs, 1}, at::kFloat);
		for (int acc_id = 0;  acc_id < num_accs; acc_id ++){
			acc_candiates[acc_id][0] = ped_model->GetAccelerationNoramlized(acc_id);
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

        assert(2*ModelParams::NumSteerAngle == steer_probs_Tensor.size(0));

		logd << "acc probs = " << acc_probs_Tensor << endl;
		logd << "steer probs = " << steer_probs_Tensor << endl;

	    auto acc_probs_double = acc_probs_Tensor.accessor<float, 1>();
	    auto steer_probs_double = steer_probs_Tensor.accessor<float, 1>();

		// Update the values in the vnode

		double accum_prob = 0;
		for (int action = 0;  action < ped_model->NumActions(); action ++){
			int acc_ID = ped_model->GetAccelerationID(action);
			int steerID = ped_model->GetSteeringID(action);

			float acc_prob = acc_probs_double[acc_ID];

			float steer_prob = cal_steer_prob(steer_probs_double, steerID);

			float joint_prob = acc_prob * steer_prob;
			vnode->prior_action_probs(action, joint_prob);

			accum_prob += joint_prob;

//			logd << "action "<< acc_ID << " " << steerID <<
//					" joint_prob = " << joint_prob
//					<< " accum_prob = " << accum_prob << endl;

			// get the steering prob from angs
		}

//		double prior_value = value_transform_inverse(value_double[node_id]);

//		logd << "assigning vnode " << vnode << " value " << prior_value << endl;

//		vnode->prior_value(prior_value);
	}

	logd << __FUNCTION__<<" " << Globals::ElapsedTime(start) << " s" << endl;
}

void PedNeuralSolverPrior::ComputeValue(vector<torch::Tensor>& input_batch, vector<despot::VNode*>& vnodes){
	logd << "[ComputeValue] get " << vnodes.size() << " nodes" << endl;
	auto start = Time::now();

	if (vnodes.size()>BatchSize){
		logd << "Executing " << (vnodes.size()-1)/BatchSize << " batches" << endl;
		for (int batch = 0; batch < (vnodes.size()-1)/BatchSize + 1; batch++) {

			int start = batch*BatchSize;
			int end = ((batch+1) * BatchSize <= vnodes.size()) ?
					(batch+1) * BatchSize : vnodes.size();

			logd << "start end: " << start << " " << end << endl;

			std::vector<torch::Tensor> mini_batch(&input_batch[start],&input_batch[end]);
			std::vector<despot::VNode*> sub_vnodes(&vnodes[batch*BatchSize],&vnodes[end]);
			ComputeMiniBatchValue(mini_batch, sub_vnodes);
		}
	}
	else
		ComputeMiniBatchValue(input_batch, vnodes);

	logd << __FUNCTION__<< " " << vnodes.size() << " nodes "  << Globals::ElapsedTime(start) << " s" << endl;
}

void PedNeuralSolverPrior::ComputeMiniBatchValue(vector<torch::Tensor>& input_batch, vector<despot::VNode*>& vnodes){

	auto start = Time::now();

	torch::NoGradGuard no_grad;
//	drive_net->eval();

	// DONE: Send nn_input_images_ to drive_net, and get the policy and value output
	const PedPomdp* ped_model = static_cast<const PedPomdp*>(model_);

	logd << "[ComputeValue] node depth " << vnodes[0]->depth() << endl;

	if(logging::level()>=4){
		logd << "[Combine_images] vnodes[0]=" << vnodes[0]->depth() << endl;
		string level = std::to_string(vnodes[0]->depth());
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

	logd << "[ComputeValue] num_nodes = "
						<< input_batch.size()<< endl;

	std::vector<torch::jit::IValue> inputs;

	torch::Tensor input_tensor;

	input_tensor = torch::stack(input_batch, 0);

	logd << "[ComputeValue] input_tensor dim = \n"
							<< input_tensor.sizes()<< endl;

	input_tensor= input_tensor.contiguous();

	inputs.push_back(input_tensor.to(at::kCUDA));

	logd << "[ComputeValue] contiguous cuda tensor \n" <<
			input_tensor<< endl;

	sync_cuda();

	logd << __FUNCTION__<<" prepare data " << Globals::ElapsedTime(start) << " s" << endl;


	logd << __FUNCTION__<<" query for "<< input_tensor.sizes() << " data " << endl;

	auto start1 = Time::now();

	auto drive_net_output = drive_net_value->forward(inputs);

	logd << "[ComputeValue] Refracting outputs " << endl;
	auto value_batch = drive_net_output.toTensor().cpu();

	value_batch = value_batch.squeeze(1);
    auto value_double = value_batch.accessor<float, 1>();

	logd << "nn query in " << Globals::ElapsedTime(start1) << " s" << endl;

	logd << "Get value output " << value_batch << endl;

    logd << "[ComputeValue] Updating prior with nn outputs " << endl;

	int node_id = -1;
	for (std::vector<despot::VNode* >::iterator it = vnodes.begin();
		        it != vnodes.end(); it++) {
		despot::VNode* vnode = *it;
		node_id ++;

		double prior_value = value_transform_inverse(value_double[node_id]);

		logd << "assigning vnode " << vnode << " value " << prior_value << endl;

		vnode->prior_value(prior_value);
	}

	logd << __FUNCTION__<<" " << Globals::ElapsedTime(start) << " s" << endl;
}

void PedNeuralSolverPrior::get_history(int mode, despot::VNode* cur_node, std::vector<despot::VNode*>& parents_to_fix_images,
		vector<PomdpState*>& hist_states, vector<int>& hist_ids){
	int num_history = 0;
	int start_channel = 0;

	if (mode == FULL){ // Full update of the input channels
		num_history = num_hist_channels;
		start_channel = 0;
		logd << "Getting FULL history for node "<< cur_node << " at depth " << cur_node->depth() << endl;

//		for (int i = 0 ; i<num_history ; i++){
//			logd << "add hist id "<< i << endl;
//			hist_ids.push_back(i);
//			parents_to_fix_images.push_back(cur_node);
//		}
	}
	else if (mode == PARTIAL){ // Partial update of input channels and reuse old channels
		num_history = num_hist_channels - 1;
		start_channel = 1;
		logd << "Getting PARTIAL history for node "<< cur_node << " at depth " << cur_node->depth() << endl;

		//[1,2,3]
		// id = 0 will be filled in the nodes
	}

	despot::VNode* parent = cur_node;

	vector<int> reuse_ids;

	for (int i = start_channel ; i< start_channel + num_history ; i++){

		if (i>start_channel)
			parent = (parent->parent()==NULL)?
					parent: parent->parent()->parent(); // to handle root node

		if (mode == FULL && cur_node->depth()==0 && !cur_node->prior_initialized()){
			// root node initialization
//			logd << "Checking reuse" << endl;
//			assert(cur_node->depth()==0);
			double cur_ts = static_cast<PomdpState*>(cur_node->particles()[0])->time_stamp;
			double hist_ts = cur_ts - i*1.0/ModelParams::control_freq;

			logd << "Comparing recorded ts "<< hist_time_stamps[i-start_channel] <<
					" with calculated ts " << hist_ts << endl;
			if(abs(hist_time_stamps[i-start_channel] - hist_ts) <=1e-2) { // can reuse the previous channel
				reuse_ids.push_back(i);

				Globals::lock_process();
				logd << "Thread " << prior_id() << " Reusing channel " << i-start_channel <<
						" to new channel " << i << " for node "<< cur_node << endl;
				Globals::unlock_process();
				continue;
			}
		}
		else{
			if(car_hist_links[i-start_channel] == parent) { // can reuse the previous channel
				reuse_ids.push_back(i);

				Globals::lock_process();
				logd << "Thread " << prior_id() << " Reusing channel " << i-start_channel <<
									" to new channel " << i << " for node "<< cur_node << endl;
				Globals::unlock_process();
				continue;
			}
		}


		logd << "add hist id" << i << ", add parent depth = " << parent->depth() << endl;

		hist_ids.push_back(i);
		parents_to_fix_images.push_back(parent);
	}


	for (int i = reuse_ids.size()-1 ; i>=0 ; i--){
		Reuse_history(reuse_ids[i], start_channel, mode);
	}

//	if (mode == PARTIAL)
//		return; // Debugging

	logd << "hist init len " << init_hist_len<< endl;
	logd << "hist_ids len " << hist_ids.size()<< endl;

	// DONE: get the 4 latest history states
	int latest=as_history_in_search_.Size()-1;
//	for (int t = latest; t > latest - num_history && t>=0 ; t--){// process in reserved time order
	for (int i =0;i<hist_ids.size(); i++){
		int t = latest;
		if (mode == FULL)
			t = latest - hist_ids[i];
		else if (mode == PARTIAL)
			t = latest - hist_ids[i] + 1;

//		if (parents_to_fix_images[i] && mode == PARTIAL){
//
//			if(parents_to_fix_images[i]->depth() != t - init_hist_len + hist_ids[i]){
//
//				logd << "parents_to_fix_images[i]->depth()= " << parents_to_fix_images[i]->depth()
//								<< ", t - init_hist_len  + hist_ids[i] = " << t - init_hist_len + hist_ids[i] << endl;
//
//				logd << "t, init_hist_len, latest, hist_id = " << t <<", "<< init_hist_len << ", "
//						<<  latest << ", " << hist_ids[i] << endl;
//
//				logd.flush();
//				raise(SIGABRT);
//			}
//		}

		if (t >=0){
			PomdpState* car_peds_state = static_cast<PomdpState*>(as_history_in_search_.state(t));
			hist_states.push_back(car_peds_state);

			Globals::lock_process();
			logd << "Thread " << prior_id() << " Using as_history_in_search_ entry " << t <<
					" ts " << car_peds_state->time_stamp <<
					" as new channel " << hist_ids[i] <<
					" node at level " << cur_node ->depth()<< endl;
			Globals::unlock_process();

//			static_cast<const PedPomdp*>(model_)->PrintState(*car_peds_state);

//			if (cur_node->depth()==0 && hist_ids[i] > 0){
//				Debug_state(car_peds_state, "Use history", model_);
//			}
		}
		else{
			hist_states.push_back(NULL);
			logd << " Using NULL state as new channel " << hist_ids[i]
					  << " node at level " << cur_node ->depth()<< endl;
		}
	}

	logd << "[Get_history] validating history, node " << cur_node << endl;

//	if (mode == PARTIAL){

	parent = cur_node;

	try{
		for (int i=start_channel;i<num_hist_channels; i++){
			logd << " [Get_history] hist " << i << " should be depth " << parent->depth() <<
								", get depth "<< (car_hist_links[i]?car_hist_links[i]->depth():-1)
										<< " node depth " << cur_node->depth()<< endl;
			if(car_hist_links[i] != parent)
				logd <<"mismatch!!!!!!!!!!!!!!!!!"<< endl;
			parent = (parent->parent()==NULL)?
					parent: parent->parent()->parent(); // to handle root node
		}
	}catch (Exception e) {
		logd << " [error] !!!!!!!!!!!!!!!!"  << e.what() << endl;
		raise(SIGABRT);
	}

//	}

	logd << "[Get_history] done " << endl;

}

void PedNeuralSolverPrior::Process_history(despot::VNode* cur_node, int mode){
	auto start = Time::now();

	logd << "Processing history, (FULL/PARTIAL) mode=" << mode << endl;

	vector<int> hist_ids;
	vector<PomdpState*> hist_states;
	vector<despot::VNode*> parents_to_fix_images;

	get_history(mode, cur_node, parents_to_fix_images, hist_states, hist_ids);

//	return; // Debugging

	Process_states(parents_to_fix_images, hist_states, hist_ids);

	logd << __FUNCTION__<<" " << Globals::ElapsedTime(start) << " s" << endl;

}

void PedNeuralSolverPrior::Record_hist_len(){
	init_hist_len = as_history_in_search_.Size();
}

void PedNeuralSolverPrior::print_prior_actions(ACT_TYPE action){
	logd << "action " << action << " (acc/steer) = " << static_cast<const PedPomdp*>(model_)->GetAcceleration(action) <<
		"/" << static_cast<const PedPomdp*>(model_)->GetSteering(action)
		<< endl;
}

std::vector<torch::Tensor> PedNeuralSolverPrior::Process_history_input(despot::VNode* cur_node){
	auto start = Time::now();


	logd << "[Process_history_input], len=" << num_hist_channels << endl;

//	int num_history = num_hist_channels;
//
//	// TODO: get the 4 latest history states
//	vector<PomdpState*> hist_states;
//	int latest=as_history_in_search_.Size()-1;
//	for (int t = latest; t > latest - num_history && t>=0 ; t--){// process in reserved time order
//		PomdpState* car_peds_state = static_cast<PomdpState*>(as_history_in_search_.state(t));
//		hist_states.push_back(car_peds_state);
//	}
//
//	logd << hist_states.size() <<" history states found" << endl;
//
//	// fill older history with empty states
//	while (hist_states.size() < num_history){
//		hist_states.push_back(NULL);
//	}
//
//	vector<int> hist_ids;
//	for (int i = 0 ; i<num_history ; i++)
//		hist_ids.push_back(i);
//
//	for (int i = 0 ; i<num_history ; i++){
//		logd << "hist state " << hist_states[i] << " hist_id " << hist_ids[i]<< endl;
//	}
//
//	std::vector<despot::VNode*> cur_nodes_dup;
//
//	int count = 0;
//	while(count < hist_ids.size()){
//		cur_nodes_dup.push_back(cur_node);
//		count++;
//	}

	vector<int> hist_ids;
	vector<PomdpState*> hist_states;
	vector<despot::VNode*> root_to_fix_images;

	get_history(FULL, cur_node, root_to_fix_images, hist_states, hist_ids);

	Process_states(root_to_fix_images, hist_states, hist_ids);

	std::vector<torch::Tensor> nn_input;
	nn_input.push_back(Combine_images(cur_node));

	logd << __FUNCTION__<<" " << Globals::ElapsedTime(start) << " s" << endl;

	return nn_input;
}

//void PedNeuralSolverPrior::OnDataReady(std::string const& name, sio::message::ptr const& data,bool hasAck, sio::message::ptr &ack_resp){
//
//	logd << " [OnDataReady] start" << endl;
//
//	auto value = data->get_vector()[0];
//	auto acc_pi = data->get_vector()[1];
//	auto acc_mu = data->get_vector()[2];
//	auto acc_sigma = data->get_vector()[3];
//
//	logd << " [OnDataReady] end" << endl;
//
//}


void PedNeuralSolverPrior::Test_all_srv(int batchsize, int num_guassian_modes, int num_steer_bins){
	cerr << "Testing all model using ROS service, bs = " << batchsize << endl;

	ros::NodeHandle n("~");

	nn_client_ = n.serviceClient<query_nn::TensorData>("/query");

	logd << "waiting for /query service to be ready" << endl;

	nn_client_.waitForExistence(ros::DURATION_MAX);

	for (int i =0 ;i< 10 ; i++){

		std::vector<torch::jit::IValue> inputs;

		auto images = torch::ones({batchsize, 9, 32, 32});

		auto start1 = Time::now();

		query_nn::TensorData message;

		message.request.tensor = std::vector<float>(images.data<float>(), images.data<float>() + images.numel());

		message.request.batchsize = batchsize;

		message.request.mode = "all";

		logd << "calling service query" << endl;


		if (nn_client_.call(message))
		{
			vector<float> value = message.response.value;
			vector<float> acc_pi = message.response.acc_pi;
			vector<float> acc_mu = message.response.acc_mu;
			vector<float> acc_sigma = message.response.acc_sigma;
			vector<float> ang = message.response.ang;

			logd << "value" << endl;
			for (int i = 0 ; i< value.size(); i++){
				logd << value[i] << " ";
			}
			logd << endl;

			logd << "acc_pi" << endl;
			for (int id = 0 ; id< acc_pi.size(); id++){
				int data_id = id / num_guassian_modes;
				int mode_id = id % num_guassian_modes;
				logd << acc_pi[id] << " ";
				if (mode_id == num_guassian_modes -1){
					logd << endl;
				}
			}

			logd << "acc_mu" << endl;
			for (int id = 0 ; id< acc_mu.size(); id++){
				int data_id = id / num_guassian_modes;
				int mode_id = id % num_guassian_modes;
				logd << acc_mu[id] << " ";
				if (mode_id == num_guassian_modes -1){
					logd << endl;
				}
			}

			logd << "ang" << endl;
			for (int id = 0 ; id< ang.size(); id++){
				int data_id = id / num_steer_bins;
				int bin_id = id % num_steer_bins;
				logd << ang[id] << " ";
				if (bin_id == num_steer_bins -1){
					logd << endl;
				}
			}
			logd << endl;

//			ROS_INFO("value: %f", (long int)message.response.value[0]);
		}
		else
		{
			ROS_ERROR("Failed to call service query");
			raise(SIGABRT);
		}

		logd << "nn query in " << Globals::ElapsedTime(start1) << " s" << endl;
	}
}

void PedNeuralSolverPrior::Test_val_srv(int batchsize, int num_guassian_modes, int num_steer_bins){

	batchsize = 128;
	cerr << "Testing value model via ros service, bs = " << batchsize << endl;

	ros::NodeHandle n("~");

	nn_client_val_ = n.serviceClient<query_nn::TensorData>("/query_val");

	logd << "waiting for /query_val service to be ready" << endl;

	nn_client_val_.waitForExistence(ros::DURATION_MAX);

	for (int i =0 ;i< 10 ; i++){

		std::vector<torch::jit::IValue> inputs;

		auto images = torch::ones({batchsize, 9, 32, 32});

		auto start1 = Time::now();

		query_nn::TensorData message;

		message.request.tensor = std::vector<float>(images.data<float>(), images.data<float>() + images.numel());

		message.request.batchsize = batchsize;

		message.request.mode = "val";


		logd << "calling service query" << endl;

		if (nn_client_val_.call(message)){
			vector<float> value = message.response.value;

			logd << "nn query in " << Globals::ElapsedTime(start1) << " s" << endl;

			logd << "value" << endl;
			for (int i = 0 ; i< value.size(); i++){
				logd << value[i] << " ";
			}
			logd << endl;
		}
		else{
			ROS_ERROR("Failed to call service /query_val");
			raise(SIGABRT);
		}
	}
}

void PedNeuralSolverPrior::Test_all_libtorch(int batchsize, int num_guassian_modes, int num_steer_bins){
	cerr << "Testing all model using libtorch, bs = " << batchsize << endl;
	string path = "/home/yuanfu/Unity/DESPOT-Unity/torchscript_version.pt";
	if( !is_file_exist(path)){
		path = "/home/panpan/Unity/DESPOT-Unity/torchscript_version.pt";
	}
	std::shared_ptr<torch::jit::script::Module> net;

	logd << "[Test_model] Testing model "<< path << endl;
	net = torch::jit::load(path);
	net->to(at::kCUDA);
	assert(net);

	logd << "[Test_model] displaying params\n";
	Show_params(net);
	for (int i =0 ;i< 10 ; i++){

		std::vector<torch::jit::IValue> inputs;

		auto images = torch::ones({batchsize, 9, 32, 32});

		auto start1 = Time::now();

		inputs.push_back(images.to(at::kCUDA));

		logd << "[Test_model] Query nn for "<< inputs.size() << " tensors of dim" << inputs[0].toTensor().sizes() << endl;

		auto drive_net_output = net->forward(inputs).toTuple()->elements();

		auto value_batch = drive_net_output[VALUE].toTensor().cpu();
		auto value_batch_double = value_batch.accessor<float,2>();
		auto acc_pi_batch = drive_net_output[ACC_PI].toTensor().cpu();
		auto acc_pi_batch_double = acc_pi_batch.accessor<float,2>();
		auto acc_mu_batch = drive_net_output[ACC_MU].toTensor().cpu();
		auto acc_mu_batch_double = acc_mu_batch.accessor<float,3>();
		auto acc_sigma_batch = drive_net_output[ACC_SIGMA].toTensor().cpu();
		auto acc_sigma_batch_double = acc_sigma_batch.accessor<float,3>();
		auto ang_batch = drive_net_output[ANG].toTensor().cpu();
		auto ang_batch_double = ang_batch.accessor<float,2>();

		logd << "value" << endl;
		for (int i = 0 ; i< value_batch.size(0); i++){
			logd << value_batch_double[i][0] << " ";
		}
		logd << endl;

		logd << "acc_pi" << endl;
		for (int data_id = 0 ; data_id< acc_pi_batch.size(0); data_id++){
			for (int mode_id = 0 ; mode_id< acc_pi_batch.size(1); mode_id++){
				logd << acc_pi_batch_double[data_id][mode_id] << " ";
				if (mode_id == num_guassian_modes -1){
					logd << endl;
				}
			}
		}

		logd << "acc_mu" << endl;
		for (int data_id = 0 ; data_id< acc_mu_batch.size(0); data_id++){
			for (int mode_id = 0 ; mode_id< acc_mu_batch.size(1); mode_id++){
				logd << acc_mu_batch_double[data_id][mode_id][0] << " ";
				if (mode_id == num_guassian_modes -1){
					logd << endl;
				}
			}
		}

		logd << "ang" << endl;
		for (int data_id = 0 ; data_id< ang_batch.size(0); data_id++){
			for (int bin_id = 0 ; bin_id< ang_batch.size(1); bin_id++){
				logd << ang_batch_double[data_id][bin_id] << " ";
				if (bin_id == num_steer_bins -1){
					logd << endl;
				}
			}
		}
		logd << endl;
		logd << "nn query in " << Globals::ElapsedTime(start1) << " s" << endl;
	}

}

void PedNeuralSolverPrior::Test_val_libtorch(int batchsize, int num_guassian_modes, int num_steer_bins){
	batchsize = 128;
	cerr << "Testing value model via libtorch, bs = " << batchsize << endl;
	string path = "/home/yuanfu/Unity/DESPOT-Unity/jit_val.pt";
	if( !is_file_exist(path)){
		path = "/home/panpan/Unity/DESPOT-Unity/jit_val.pt";
	}

	logd << "[Test_model] Loading value model "<< path << endl;
	auto val_net = torch::jit::load(path);
	val_net->to(at::kCUDA);
	assert(net);
	for (int i =0 ;i< 10 ; i++){

		std::vector<torch::jit::IValue> inputs;

		auto images = torch::ones({batchsize, 9, 32, 32});

		auto start1 = Time::now();

		inputs.push_back(images.to(at::kCUDA));

		logd << "[Test_model] Query nn for "<< inputs.size() << " tensors of dim" << inputs[0].toTensor().sizes() << endl;

		auto drive_net_output = val_net->forward(inputs);

		auto value_batch = drive_net_output.toTensor().cpu();
		auto value_batch_double = value_batch.accessor<float,2>();
		logd << "nn query in " << Globals::ElapsedTime(start1) << " s" << endl;

		logd << "value" << endl;
		for (int i = 0 ; i< value_batch.size(0); i++){
			logd << value_batch_double[i][0] << " ";
		}
		logd << endl;
	}

}

void PedNeuralSolverPrior::Test_model(string path){

//	sio::client query_client;
//	query_client.connect("http://127.0.0.1:8080");


//	if (drive_net){
//		logd << "[Test_model] Testing drive_net loaded in prior " << endl;
//		net = drive_net;
//	}
//	else{
//		logd << "[Test_model]  drive net not ready yet!!!!!!!!!!!!!!!" << endl;
//		exit(1);
//	}

	logd << "[Test_model] Query " << endl;

	torch::NoGradGuard no_grad;

	int batchsize = 1;

	int num_guassian_modes = 5;
	int num_steer_bins = 2*ModelParams::NumSteerAngle;

	Test_all_srv(batchsize, num_guassian_modes, num_steer_bins);

	Test_val_srv(batchsize, num_guassian_modes, num_steer_bins);

	Test_all_libtorch(batchsize, num_guassian_modes, num_steer_bins);

	Test_val_libtorch(batchsize, num_guassian_modes, num_steer_bins);

	logd << "[Test_model] Done " << endl;

}

State* debug_state = NULL;

void PedNeuralSolverPrior::DebugHistory(string msg){
	for (int t = 0; t < as_history_in_search_.Size(); t++){
		auto state = as_history_in_search_.state(t);
		Debug_state(state,  msg + "_t_"+ std::to_string(t), model_);
	}
}

void Debug_state(State* state, string msg, const DSPOMDP* model){
	if (state == debug_state){
		bool mode = DESPOT::Debug_mode;
		DESPOT::Debug_mode = false;

		PomdpState* hist_state = static_cast<PomdpState*>(state);
		static_cast<const PedPomdp*>(model)->PrintState(*hist_state);

		DESPOT::Debug_mode = mode;

		cerr << "=================== " << msg << " breakpoint ==================="<< endl;
		raise(SIGABRT);
	}
}

void Record_debug_state(State* state){
	debug_state = state;
}

bool Compare_states(PomdpState* state1, PomdpState* state2){
	if ((state1->car.pos.x != state2->car.pos.x) ||
			(state1->car.pos.y != state2->car.pos.y) ||
			(state1->car.vel != state2->car.vel) ||
			(state1->car.heading_dir != state2->car.heading_dir)){
		cerr << "!!!!!!! car diff !!!!!!" << endl;
		return true;
	}

	if (state1->num != state2->num){
		cerr << "!!!!!!! ped num diff !!!!!!" << endl;
		return true;
	}

	bool diff= false;
	for (int i =0 ; i< state1->num; i++){
		diff = diff || (state1->peds[i].pos.x != state2->peds[i].pos.x);
		diff = diff || (state1->peds[i].pos.y != state2->peds[i].pos.y);
		diff = diff || (state1->peds[i].goal != state2->peds[i].goal);

		if (diff){
			cerr << "!!!!!!! ped " << i << " diff !!!!!!" << endl;
			return true;
		}
	}

	return false;
}

void PedNeuralSolverPrior::record_cur_history(){
	as_history_in_search_recorded.Truncate(0);
	for (int i = 0 ;i<as_history_in_search_.Size(); i++){
		as_history_in_search_recorded.Add(as_history_in_search_.Action(i), as_history_in_search_.state(i));
	}
}
void PedNeuralSolverPrior::compare_history_with_recorded(){

	if (as_history_in_search_.Size() != as_history_in_search_recorded.Size()){
		cerr << "ERROR: history length changed after search!!!" << endl;
		raise(SIGABRT);
	}
	for (int i = 0 ;i<as_history_in_search_recorded.Size(); i++){
		PomdpState* recorded_hist_state =  static_cast<PomdpState*>(as_history_in_search_recorded.state(i));
		PomdpState* hist_state =  static_cast<PomdpState*>(as_history_in_search_recorded.state(i));

		bool different = Compare_states(recorded_hist_state, hist_state);

		if( different){
			cerr << "ERROR: history "<< i << " changed after search!!!" << endl;
			static_cast<const PedPomdp*>(model_)->PrintState(*recorded_hist_state, "Recorded hist state");
			static_cast<const PedPomdp*>(model_)->PrintState(*hist_state, "Hist state");

			raise(SIGABRT);
		}
	}
}

