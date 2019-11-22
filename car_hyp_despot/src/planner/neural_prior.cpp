
#define ONEOVERSQRT2PI 1.0 / sqrt(2.0 * M_PI)

#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include "ped_pomdp.h"
#include "neural_prior.h"

#undef LOG
#define LOG(lv) \
if (despot::logging::level() < despot::logging::ERROR || despot::logging::level() < lv) ; \
else despot::logging::stream(lv)
#include <despot/util/logging.h>
#include <cmath>

double value_normalizer = 10.0;
const char* window_name = "images";

torch::Device device(torch::kCPU);
bool use_gpu_for_nn = true;

bool do_print = false;

int init_hist_len = 0;
int max_retry_count = 3;

int export_image_level = 3;

ros::ServiceClient PedNeuralSolverPrior::nn_client_;
ros::ServiceClient PedNeuralSolverPrior::nn_client_val_;

std::chrono::time_point<std::chrono::system_clock> SolverPrior::init_time_;

bool delectNAN(double v){
	if (isnan(v))
		return true;
	else if (v != v){
		return true;
	}

	return false;
}

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


void fill_polygon_edges(Mat& image, vector<COORD>& points){
	float default_intensity = 1.0;
//	int thickness = 1; // for debugging

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
std::string img_folder="";

void mkdir_safe(std::string dir){
	if (mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
	{
	    if( errno == EEXIST ) {
	       // alredy exists
	    } else {
	       // something else
	        std::cout << "cannot create folder " << dir << " error:" << strerror(errno) << std::endl;
	        ERR("");
	    }
	}
}

bool dir_exist(string pathname){
	struct stat info;

	if( stat( pathname.c_str(), &info ) != 0 ) {
	    printf( "cannot access %s\n", pathname );
	    return false;
	}
	else if( info.st_mode & S_IFDIR ) {  // S_ISDIR() doesn't exist on my windows
//	    printf( "%s is a directory\n", pathname );
	    return true;
	}
	else {
	    printf( "%s is no directory\n", pathname );
	    return false;
	}
}

void rm_files_in_folder(string folder) {
    // These are data types defined in the "dirent" header
    DIR *theFolder = opendir(folder.c_str());
    struct dirent *next_file;
    char filepath[256];

    while ( (next_file = readdir(theFolder)) != NULL ) {
        // build the path for each file in the folder
        sprintf(filepath, "%s/%s", folder.c_str(), next_file->d_name);
        cout << "Removing file " << filepath << endl;
        remove(filepath);
    }
    closedir(theFolder);
}

void clear_image_folder() {
	std::string homedir = getenv("HOME");
	img_folder = homedir + "/catkin_ws/visualize";
	mkdir_safe(img_folder);
	rm_files_in_folder(img_folder);
}

void export_image(Mat& image, string flag){
	logi << "[export_image] start" << endl;
	std::ostringstream stringStream;
	stringStream << img_folder << "/" << img_counter << "_" << flag << ".jpg";
	std::string img_name = stringStream.str();

    Mat tmp = image.clone();

	double image_min, image_max;
	cv::minMaxLoc(tmp, &image_min, &image_max);
	logi << "saving image " << img_name << " with min-max values: "
			<< image_min <<", "<< image_max << endl;

	cv::Mat for_save;
	tmp.convertTo(for_save, CV_8UC3, 255.0);
	imwrite( img_name , for_save);

//	imshow( img_name, image );
//
//	char c = (char)waitKey(0);
//
//	cvDestroyWindow((img_name).c_str());
	logi << "[export_image] end" << endl;
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

	Globals::lock_process();
//	logi << "Tensor " << msg << endl;
    auto tensor_double = tensor.accessor<float, 2>();
    for (int i =0;i<tensor.size(0);i++){
		logi << msg << " ";

    	for (int j =0;j<tensor.size(1);j++){
    		logi << std::setprecision(2)<< tensor_double[i][j] << " ";
    	}
    	logi << endl;
    }
    Globals::unlock_process();
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
	/*const auto& model_params = drive_net->get_modules();

	int iter = 0;

	ofstream fout;
	string model_param_file = "/home/panpan/NN_params.txt";
	fout.open(model_param_file, std::ios::trunc);
	assert(fout.is_open());

	for(const auto& module: model_params) {
		// fout << module.name() << ": " << std::endl;

		const auto& module_params = module.get_parameters();
		fout << module_params.size() << " params found:" << std::endl;
		for (const auto& param: module_params){
			fout<< param.value() <<std::endl;
		}

//		if (module_name == "ang_head" ){
		fout << "module_name sub_modules: " << std::endl;

		const auto& sub_modules = module.get_modules();
		for(const auto& sub_module: sub_modules) {
			// fout << "sub-module found " << sub_module.name() << ": " << std::endl;

			const auto& sub_module_params = sub_module.get_parameters();
			fout << sub_module_params.size() << " params found:" << std::endl;
			for (const auto& param: sub_module_params){
				fout<< param.value() <<std::endl;
			}
		}
//		}
		iter ++;
//		if (iter == 20) break;
	}

	fout.close();*/
}

void PedNeuralSolverPrior::Init(){
	// DONE: The environment map will be received via ROS topic as the OccupancyGrid type
	//		 Data will be stored in raw_map_ (class member)
	//       In the current stage, just use a randomized raw_map_ to develop your code.
	//		 Init raw_map_ including its properties here
	//	     Refer to python codes: bag_2_hdf5.parse_map_data_from_dict
	//		 (map_dict_entry is the raw OccupancyGrid data)

  if (Globals::config.use_prior){
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

    map_image_ = cv::Mat(map_prop_.dim, map_prop_.dim, CV_32FC1, cv::Scalar(0.0));
    map_prop_.map_intensity = 1.0;
//    int index = 0;
//    for (std::vector<int8_t>::const_reverse_iterator iterator = raw_map_.data.rbegin();
//        iterator != raw_map_.data.rend(); ++iterator) {
//      int x = (map_prop_.dim - 1) - (size_t)(index / map_prop_.dim);
//      int y = (map_prop_.dim - 1) - (size_t)(index % map_prop_.dim);
//      assert(*iterator != -1);
//      map_image_.at<float>(x,y) = (float)(*iterator);
//      index++;
//    }
//
//    double minVal, maxVal;
//    minMaxLoc(map_image_, &minVal, &maxVal);
//    map_prop_.map_intensity = maxVal;

    logd << "Map properties: " << endl;
    logd << "-dim " << map_prop_.dim << endl;
    logd << "-new_dim " << map_prop_.new_dim << endl;
    logd << "-downsample_ratio " << map_prop_.downsample_ratio << endl;
    logd << "-map_intensity " << map_prop_.map_intensity << endl;
    logd << "-map_intensity_scale " << map_prop_.map_intensity_scale << endl;
    logd << "-resolution " << map_prop_.resolution << endl;

    cerr << "DEBUG: Scaling map" << endl;

    rescaled_map_ = cv::Mat(32, 32, CV_32FC1, cv::Scalar(0.0));

//    rescaled_map_ = rescale_image(map_image_);
//    normalize(rescaled_map_);

    cerr << "DEBUG: Initializing other images" << endl;

    path_image_ = cv::Mat(map_prop_.dim, map_prop_.dim, CV_32FC1);
    lane_image_ = cv::Mat(map_prop_.dim, map_prop_.dim, CV_32FC1);

    for( int i=0;i<num_hist_channels;i++){
      map_hist_images_.push_back(cv::Mat(map_prop_.dim, map_prop_.dim, CV_32FC1));
//      car_hist_images_.push_back(cv::Mat(map_prop_.dim, map_prop_.dim, CV_32FC1));
    }

    map_hist_image_ = cv::Mat(map_prop_.dim, map_prop_.dim, CV_32FC1);
//    car_hist_image_ = cv::Mat(map_prop_.dim, map_prop_.dim, CV_32FC1);

    cerr << "DEBUG: Initializing tensors" << endl;

    empty_map_tensor_ = at::zeros({map_prop_.new_dim, map_prop_.new_dim}, at::kFloat);
    map_tensor_ = at::zeros({map_prop_.new_dim, map_prop_.new_dim}, at::kFloat);
    path_tensor = torch::zeros({map_prop_.new_dim, map_prop_.new_dim}, at::kFloat);
    lane_tensor = torch::zeros({map_prop_.new_dim, map_prop_.new_dim}, at::kFloat);

    logd << "[PedNeuralSolverPrior::Init] create tensors of size " <<map_prop_.new_dim<< ","<<map_prop_.new_dim<< endl;
    for( int i=0;i<num_hist_channels;i++){

      map_hist_tensor_.push_back(torch::zeros({map_prop_.new_dim, map_prop_.new_dim}, at::kFloat));
//      car_hist_tensor_.push_back(torch::zeros({map_prop_.new_dim, map_prop_.new_dim}, at::kFloat));

      map_hist_links.push_back(NULL);
      car_hist_links.push_back(NULL);
      hist_time_stamps.push_back(-1);
    }

    goal_link = NULL;
    lane_link = NULL;

    clear_image_folder();
    logd << "[PedNeuralSolverPrior::Init] end " << endl;
  }
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
	logd << "DEBUG: Initializing PedNeuralSolverPrior" << endl;

	action_probs_.resize(model->NumActions());

	// TODO: get num_peds_in_NN from ROS param
	num_peds_in_NN = 20;
	// TODO: get num_hist_channels from ROS param
	num_hist_channels = 4;

	// DONE Declare the neural network as a class member, and load it here


	logd << "DEBUG: Initializing car shape" << endl;

	// Car geometry
	car_shape = vector<cv::Point3f>({ Point3f(3.6, 0.95, 1), Point3f(-0.8, 0.95, 1), Point3f(-0.8, -0.95, 1), Point3f(3.6, -0.95, 1)});

	map_received = false;
	drive_net = NULL;


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
	drive_net = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_file));

	if(drive_net == NULL)
		ERR("");

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
	drive_net_value = std::make_shared<torch::jit::script::Module>(torch::jit::load(value_model_file));

	if(drive_net_value == NULL)
		ERR("");

	if (use_gpu_for_nn)
		drive_net_value->to(at::kCUDA);

	cerr << "DEBUG: Loaded model "<< value_model_file << endl;

	logd << __FUNCTION__<<" " << Globals::ElapsedTime(start) << " s" << endl;

	if (Globals::config.use_prior){
		logd << "[" << __FUNCTION__<< "] Testing model start" << endl;

		Test_model("");

		logd << "[" << __FUNCTION__<< "] Testing model end" << endl;
	}
}

COORD PedNeuralSolverPrior::point_to_indices(COORD pos, COORD origin, double resolution, int dim) const{
//	logd << "[point_to_indices] " << endl;

	COORD indices = COORD((pos.x - origin.x) / resolution, (pos.y - origin.y) / resolution);
    if (indices.x < 0 || indices.y < 0 ||
    		indices.x > (dim - 1) || indices.y > (dim - 1))
        return COORD(-1, -1);
    return indices;
}

COORD PedNeuralSolverPrior::point_to_indices_unbounded(COORD pos, COORD origin, double resolution) const{
//	logd << "[point_to_indices] " << endl;

	COORD indices = COORD((pos.x - origin.x) / resolution,
			(pos.y - origin.y) / resolution);

    return indices;
}

void PedNeuralSolverPrior::add_in_map(cv::Mat map_image, COORD indices, double map_intensity, double map_intensity_scale){

	if (indices.x == -1 || indices.y == -1)
		return;

//	logd << "[add_in_map] " << endl;

	map_image.at<float>((int)round(indices.y), (int)round(indices.x)) = map_intensity * map_intensity_scale;

//	logd << "[add_in_map] fill entry " << round(indices.x) << " " << round(indices.y) << endl;

}

std::vector<COORD> PedNeuralSolverPrior::get_image_space_agent(const AgentStruct agent,
		COORD origin, double resolution, double dim){
	auto start = Time::now();

    float theta = agent.heading_dir;
    float x = agent.pos.x;
    float y = agent.pos.y;

//    agent.text(logd);
//    logd << "=========== transforming agent " << agent.id << "theta: " << theta << " x: " << x << " y: " << y << endl;

    if (agent.bb_extent_y == 0){
    	agent.text(cerr);
    	ERR("agent.bb_extent_y == 0");
    }

    logd << "raw agent shape: \n";
    vector<COORD> agent_shape({COORD(agent.bb_extent_y, agent.bb_extent_x),
    							COORD(-agent.bb_extent_y, agent.bb_extent_x),
    							COORD(-agent.bb_extent_y, -agent.bb_extent_x),
    							COORD(agent.bb_extent_y, -agent.bb_extent_x)});
	for (int i=0; i < agent_shape.size(); i++){
	  logd << agent_shape[i].x << " " << agent_shape[i].y << endl;
	}
    // rotate and scale the agent

	bool out_of_map = false;
    vector<COORD> image_space_polygon;
    for (auto &coord:agent_shape){
    	vector<Point3f> original, rotated;
    	original.push_back(Point3f(coord.x, coord.y, 1.0));
    	rotated.resize(1);
    	// rotate w.r.t its local coordinate system and transform to (x, y)
    	cv::transform(original, rotated,
    			cv::Matx33f(cos(theta), -sin(theta), x, sin(theta), cos(theta), y, 0, 0, 1));

    	COORD image_space_indices = point_to_indices(COORD(rotated[0].x, rotated[0].y), origin, resolution, dim);
    	image_space_polygon.push_back(image_space_indices);
    	if (image_space_indices.x == -1 or image_space_indices.y == -1)
    		out_of_map = true;
    }

	logd << "transformed: \n";
	for (int i=0; i < image_space_polygon.size(); i++){
		logd << image_space_polygon[i].x << " " << image_space_polygon[i].y << endl;
	}

	logd << "image origin " << origin.x << " " << origin.y << endl;

	if (out_of_map) // agent out side of map should not be considered
		image_space_polygon.resize(0);

	logd << __FUNCTION__<<" " << Globals::ElapsedTime(start) << " s" << endl;

    return image_space_polygon;
}

std::vector<COORD> PedNeuralSolverPrior::get_image_space_car(const CarStruct car, COORD origin, double resolution){
	auto start = Time::now();

    logd << "car bb: \n";
	for (int i=0; i < car_shape.size(); i++){
	  logd << car_shape[i].x << " " << car_shape[i].y << endl;
	}
    // rotate and scale the car
    vector<COORD> car_polygon;
    for (auto& point: car_shape){
    	COORD indices = point_to_indices_unbounded(COORD(point.x, point.y),origin, resolution);
    	car_polygon.push_back(indices);
    }

	logd << "image space bb: \n";
	for (int i=0; i < car_polygon.size(); i++){
		logd << car_polygon[i].x << " " << car_polygon[i].y << endl;
	}

	logd << __FUNCTION__<<" " << Globals::ElapsedTime(start) << " s" << endl;

    return car_polygon;
}

void PedNeuralSolverPrior::Process_image_to_tensor(cv::Mat& src_image, at::Tensor& des_tensor, string flag){
	auto start = Time::now();

	logd << "original "<< src_image.size[0] << src_image.size[1] << endl;
	logd << "flag " <<  flag << endl;

	auto rescaled_image = rescale_image(src_image);

	double image_min, image_max;

	normalize(rescaled_image);

	if(flag.find("map") != std::string::npos){

		merge_images(rescaled_map_, rescaled_image);

		logd << __FUNCTION__<<" merge " << Globals::ElapsedTime(start) << " s" << endl;
	}

	copy_to_tensor(rescaled_image, des_tensor);

	logd << __FUNCTION__<<" copy " << Globals::ElapsedTime(start) << " s" << endl;
	logd << __FUNCTION__<<" " << Globals::ElapsedTime(start) << " s" << endl;

	if(logging::level()>=export_image_level + 1){
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
//		car_hist_tensor_[new_channel] = car_hist_tensor_[old_channel];

		map_hist_links[new_channel] = map_hist_links[old_channel];
		car_hist_links[new_channel] = car_hist_links[old_channel];

		hist_time_stamps[new_channel] = hist_time_stamps[old_channel];
	}
	else{
		logd << "skipped" << endl;
	}
}

void PedNeuralSolverPrior::Process_ego_car_images(
		const vector<PomdpState*>& hist_states, const vector<int>& hist_ids) {
	// DONE: Allocate num_history history images, each for a frame of car state
	//		 Refer to python codes: bag_2_hdf5.get_transformed_car, fill_car_edges, fill_image_with_points
	// DONE: get_transformed_car apply the current transformation to the car bounding box
	//	     fill_car_edges fill edges of the car shape with dense points
	//		 fill_image_with_points fills the corresponding entries in the images (with some intensity value)
	for (int i = 0; i < hist_states.size(); i++) {
		int hist_channel = hist_ids[i];
		car_hist_images_[hist_channel].setTo(0.0);

		logd << "[Process_states] reseting image for car hist " << hist_channel
				<< endl;

		if (hist_states[i]) {
			logd << "[Process_states] processing car for hist " << hist_channel
					<< endl;

			CarStruct& car = hist_states[i]->car;

			vector<COORD> transformed_car = get_image_space_car(car,
					map_prop_.origin, map_prop_.resolution);
			fill_polygon_edges(car_hist_images_[hist_channel], transformed_car);
		}
	}
}

void PedNeuralSolverPrior::Process_exo_agent_images(
		const vector<PomdpState*>& hist_states, const vector<int>& hist_ids) {
	for (int i = 0; i < hist_states.size(); i++) {

		int hist_channel = hist_ids[i];

		logd << "[Process_exo_agent_images] reseting image for map hist " << hist_channel
				<< endl;

		// clear data in the dynamic map
		map_hist_images_[hist_channel].setTo(0.0);

		// get the array of pedestrians (of length ModelParams::N_PED_IN)
		if (hist_states[i]) {
			logd << "[Process_exo_agent_images] start processing peds for "
					<< hist_states[i] << endl;
			//			pomdp_model->PrintState(*hist_states[i]);
			auto& agent_list = hist_states[i]->agents;

			logd << "[Process_exo_agent_images] iterating peds in agent_list="
					<< &agent_list << endl;

			for (int agent_id = 0; agent_id < ModelParams::N_PED_IN; agent_id++) {
				// Process each pedestrian
				AgentStruct agent = agent_list[agent_id];

				if (agent.id != -1) {
					// get position of the ped
					auto image_space_coords = get_image_space_agent(agent,
							map_prop_.origin, map_prop_.resolution, map_prop_.dim);

					// put the point in the dynamic map
					fill_polygon_edges(map_hist_images_[hist_channel], image_space_coords);
				}
			}
		}
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

	update_map_origin(hist_states[0]);

	Process_exo_agent_images(hist_states, hist_ids);

	logd << __FUNCTION__<<" Process ped images: " << Globals::ElapsedTime(start) << " s" << endl;
	start = Time::now();

	if (hist_states.size()==1 || hist_states.size()==4) { // only for current node states
		Process_path_image(hist_states[0]);
		Process_lane_image(hist_states[0]);
	}

	logd << __FUNCTION__<<" Process Goal Image: " << Globals::ElapsedTime(start) << " s" << endl;

//	start = Time::now();
//	Process_ego_car_images(hist_states, hist_ids);
//	logd << __FUNCTION__<<" Process Car Images: " << Globals::ElapsedTime(start) << " s" << endl;

	start = Time::now();
	logd << "[Process_states] re-scaling images to tensor " << endl;

	if (hist_states.size()==1 || hist_states.size()==4){  // only for current node states
		Process_image_to_tensor(path_image_, path_tensor, "path");
		Process_image_to_tensor(lane_image_, lane_tensor, "lanes");

		goal_link = nodes[0];
		lane_link = nodes[0];
	}
	for (int i = 0; i < hist_states.size(); i++) {
		int hist_channel = hist_ids[i];

		logd << "[Process_states] create new data for channel " << hist_channel <<" by node "<< nodes[i]<< endl;

		Process_image_to_tensor(map_hist_images_[hist_channel], map_hist_tensor_[hist_channel], "map_"+std::to_string(hist_channel));
//		Process_image_to_tensor(car_hist_images_[hist_channel], car_hist_tensor_[hist_channel], "car_"+std::to_string(hist_channel));

		map_hist_links[hist_channel] = nodes[i];
		car_hist_links[hist_channel] = nodes[i];
		hist_time_stamps[hist_channel] = hist_states[i]->time_stamp;

		if (history_mode == "track"){
//			static_cast<Shared_VNode*>(nodes[i])->car_tensor = car_hist_tensor_[hist_channel];
			static_cast<Shared_VNode*>(nodes[i])->map_tensor = map_hist_tensor_[hist_channel];
		}
//		if (nodes[i]->depth()==0 && hist_ids[i] > 0){
//			Debug_state(hist_states[i], "Process states", model_);
//		}
	}

	logd << __FUNCTION__<<" Scale Images: " << Globals::ElapsedTime(start) << " s" << endl;

	logd << "[Process_states] done " << endl;

	logd << __FUNCTION__<<" " << Globals::ElapsedTime(start_total) << " s" << endl;

}

at::Tensor PedNeuralSolverPrior::Process_tracked_state_to_car_tensor(const State* s){
	const PomdpState* agent_state = static_cast<const PomdpState*>(s);

	car_hist_image_.setTo(0.0);

	if (agent_state){
		logd << "[Process_states] processing car for state " << s << endl;

		const CarStruct& car = agent_state->car;

		vector<COORD> transformed_car = get_image_space_car(car,
							map_prop_.origin, map_prop_.resolution);
		fill_polygon_edges(car_hist_image_, transformed_car);
	}

	at::Tensor car_tensor;
	Process_image_to_tensor(car_hist_image_, car_tensor, "car_state_"+std::to_string(long(s)));

	return car_tensor;
}

at::Tensor PedNeuralSolverPrior::Process_track_state_to_map_tensor(const State* s){
	const PomdpState* agent_state = static_cast<const PomdpState*>(s);

	map_hist_image_.setTo(0.0);

	// get the array of pedestrians (of length ModelParams::N_PED_IN)
	if (agent_state){
		logd << "[Process_states] start processing peds for " << agent_state << endl;
		const auto& agent_list = agent_state->agents;
		int num_valid_ped = 0;

		logd << "[Process_states] iterating peds in agent_list=" << &agent_list << endl;

		for (int agent_id = 0; agent_id < ModelParams::N_PED_IN; agent_id++) {
			// Process each pedestrian
			AgentStruct agent = agent_list[agent_id];

			if (agent.id != -1) {
				auto image_space_coords = get_image_space_agent(agent,
						map_prop_.origin, map_prop_.resolution, map_prop_.dim);

				// put the point in the dynamic map
				fill_polygon_edges(map_hist_image_, image_space_coords);
			}
		}
	}

	at::Tensor map_tensor;
	Process_image_to_tensor(map_hist_image_, map_tensor, "map_state_"+std::to_string(long(s)));

	if(logging::level() >= export_image_level+1) {
		export_image(map_hist_image_, "tracked_map");
		inc_counter();
	}
	return map_tensor;
}

at::Tensor PedNeuralSolverPrior::Process_tracked_state_to_lane_tensor(const State* s){
	const PomdpState* agent_state = static_cast<const PomdpState*>(s);

	map_hist_image_.setTo(0.0);

	// get the array of pedestrians (of length ModelParams::N_PED_IN)
	if (agent_state){
		Process_lane_image(agent_state);
	}

	at::Tensor lane_tensor;
	Process_image_to_tensor(lane_image_, lane_tensor, "lanes_state_"+ std::to_string(long(s)));

	if(logging::level() >= export_image_level + 1) {
		export_image(lane_image_, "tracked_lanes");
	}

	return lane_tensor;
}

void PedNeuralSolverPrior::Process_lane_image(const PomdpState* agent_state){
	lane_image_.setTo(0.0);

	auto& cur_car_pos = agent_state->car.pos;

	logd << "[Process_lane_image] processing lane list of size "<< world_model.local_lane_segments_.size() << endl;

	auto& lanes = world_model.local_lane_segments_;
	for (auto& lane_seg: lanes) {
		COORD image_space_start = point_to_indices_unbounded(
				COORD(lane_seg.start.x, lane_seg.start.y), map_prop_.origin, map_prop_.resolution);
		COORD image_space_end = point_to_indices_unbounded(
				COORD(lane_seg.end.x, lane_seg.end.y), map_prop_.origin, map_prop_.resolution);

		vector<COORD> tmp_polygon({image_space_start, image_space_end});

		//TODO: check whether out-of-bound indices cause unexpected errors!!!!

		fill_polygon_edges(lane_image_, tmp_polygon);
	}
}

at::Tensor PedNeuralSolverPrior::Process_lane_tensor(const State* s){
	const PomdpState* agent_state = static_cast<const PomdpState*>(s);

	Process_lane_image(agent_state);

	Process_image_to_tensor(lane_image_, lane_tensor, "lanes_state_"+ std::to_string(long(s)));

	return lane_tensor;
}

void PedNeuralSolverPrior::Process_path_image(const PomdpState* agent_state){
	path_image_.setTo(0.0);

	// get distance between cur car pos and car pos at root node
	auto& cur_car_pos = agent_state->car.pos;
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
		path_image_.at<float>((int)round(indices.y), (int)round(indices.x)) = 1.0 * 1.0;
	}
}


at::Tensor PedNeuralSolverPrior::Process_path_tensor(const State* s){

	const PomdpState* agent_state = static_cast<const PomdpState*>(s);

	Process_path_image(agent_state);

//	at::Tensor path_tensor;
	Process_image_to_tensor(path_image_, path_tensor, "path_state_"+ std::to_string(long(s)));

	return path_tensor;
}

void PedNeuralSolverPrior::Add_tensor_hist(const State* s){
	update_map_origin(static_cast<const PomdpState*>(s));
//	car_hist_.push_back(Process_state_to_car_tensor(s));
	map_hist_.push_back(Process_track_state_to_map_tensor(s));
	Process_tracked_state_to_lane_tensor(s); //Just for debugging
}

void PedNeuralSolverPrior::Trunc_tensor_hist(int size){
//	car_hist_.resize(size);
	map_hist_.resize(size);
}

int PedNeuralSolverPrior::Tensor_hist_size(){
//	if (car_hist_.size() != map_hist_.size())
//		ERR("");
	return map_hist_.size();
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

		if (do_print){
			Globals::lock_process();
			logi << "Thread " << prior_id() << " Using current state of node " << i << " depth " << vnodes[i]->depth() << " as channel 0" << endl;
			Globals::unlock_process();
		}

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

//	try{
//		logd << __FUNCTION__ << " node depth " << cur_node->depth()<< endl;
//
//		for (int i=0;i<num_hist_channels; i++){
//			logd << "channel " << i << " ts " << hist_time_stamps[i] <<
//					" linked node depth " <<  car_hist_links[i]->depth() << endl;
//		}
//
//		for (int i=0;i<num_hist_channels; i++){
//			logd << " [Combine_images] hist " << i << " should be depth " << parent->depth() <<
//					", get depth "<< car_hist_links[i]->depth()<< endl;
//
//			if(car_hist_links[i] != parent && car_hist_links[i] != cur_node){
//				ERR("");
//			}
//
//			parent = (parent->parent()==NULL)?
//					parent: parent->parent()->parent(); // to handle root node
//		}
//	}catch (Exception e) {
//		logd << " [error] !!!!!!!!!!!!!!!!"  << e.what() << endl;
//		ERR("");
//	}
//
//	logd << "[Compute] validating history, node level " << cur_node->depth() << endl;

	logd << "[Combine_images] map_hist_tensor_ len = " << map_hist_tensor_.size() << endl;

//	cout << "[Combine] map channel 0 "<< endl;
//	print_full(map_hist_tensor_[0], "map 0");
//	print_full(map_hist_tensor_[1], "map 1");
//	print_full(path_tensor, "goal");
//	print_full(car_hist_tensor_[0], "car 0");
//	print_full(car_hist_tensor_[1], "car 1");

	torch::Tensor result;

	if (!path_tensor.defined()){
		cerr << "Empty path tensor" << endl;
		ERR("");
	}

	if (!lane_tensor.defined()){
		cerr << "Empty lane tensor" << endl;
		ERR("");
	}

	int i = 0;
	for (at::Tensor& t: map_hist_tensor_){
		if (!t.defined()){
			cerr << "Empty map tensor channel " << i << endl;
			ERR("");
		}
		i++;
	}

//	i = 0;
//	for (at::Tensor& t: car_hist_tensor_){
//		if (!t.defined()){
//			cerr << "Empty car tensor channel " << i << endl;
//			ERR("");
//		}
//		i++;
//	}

	try{

		auto combined = map_hist_tensor_;
		combined.push_back(lane_tensor);
		combined.push_back(path_tensor);
//		combined.insert(combined.end(), car_hist_tensor_.begin(), car_hist_tensor_.end());

		result = torch::stack(combined, 0);
	} catch (exception &e) {
		cerr << "Combine error: " << e.what() << endl;

		for (int i =0; i< map_hist_tensor_.size(); i++){
			cerr << "Combine info: map_hist_tensor_["<< i << "] dims=" << map_hist_tensor_[i].sizes() << endl;
//			cerr << "Combine info: car_hist_tensor_["<< i << "] dims=" << car_hist_tensor_[i].sizes() << endl;
		}
		cerr << "Combine info: lane_tensor dims=" << lane_tensor.sizes() << endl;
		cerr << "Combine info: path_tensor dims=" << path_tensor.sizes() << endl;

		ERR("");
	}

	logd << __FUNCTION__<<" " << Globals::ElapsedTime(start) << " s" << endl;

	return result;
}

int BatchSize = 128;

float PedNeuralSolverPrior::cal_steer_prob(at::TensorAccessor<float, 1> steer_probs_double, int steerID){

	float smoothing = 0.1;
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

void PedNeuralSolverPrior::Compute_val(torch::Tensor input_tensor,
		const PedPomdp* ped_model, vector<despot::VNode*>& vnodes) {

	auto start1 = Time::now();
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(input_tensor.to(at::kCUDA));

	auto drive_net_output = drive_net_value->forward(inputs);
	logd << "[Compute] Refracting value outputs " << endl;
	auto value_batch = drive_net_output.toTensor().cpu();
	value_batch = value_batch.squeeze(1);

	logd << "[Compute] Refracting value outputs " << endl;

	auto value_double = value_batch.accessor<float, 1>();

	logd << "nn query in " << Globals::ElapsedTime(start1) << " s" << endl;

	logd << "Get value output " << value_batch << endl;

	int node_id = -1;
	for (std::vector<despot::VNode* >::iterator it = vnodes.begin();
				it != vnodes.end(); it++) {
		despot::VNode* vnode = *it;
		node_id ++;

		double prior_value = value_transform_inverse(value_double[node_id]) + ModelParams::GOAL_REWARD;
		logd << "assigning vnode " << vnode << " value " << prior_value << endl;
		vnode->prior_value(prior_value);
	}
}

void PedNeuralSolverPrior::Compute_pref(torch::Tensor input_tensor,
		const PedPomdp* ped_model, vector<despot::VNode*>& vnodes) {

	auto start1 = Time::now();
	at::Tensor value_batch_dummy, acc_batch, ang_batch;
	bool succeed = false;
	int retry_count = 0;
	while (!succeed) {
		succeed = query_srv(vnodes.size(), input_tensor, value_batch_dummy,
				acc_batch, ang_batch);
		retry_count++;
		if (!succeed) {
			logi << "Action model query failed for nodes at level " <<
					vnodes[0]->depth() << "!!!" << endl;
			logi << "retry_count = " << retry_count << ", max_retry="
					<< max_retry_count << endl;
		}
		if (retry_count == max_retry_count)
			break;
	}
	if (!succeed) {
		ERR("ERROR: NN query failure !!!!!");
	}

	logi << "Action model query succeeded for nodes at level " << vnodes[0]->depth() << endl;
	logd << "[Compute] Updating prior with nn outputs " << endl;
	int node_id = -1;
	for (std::vector<despot::VNode*>::iterator it = vnodes.begin();
			it != vnodes.end(); it++) {
		despot::VNode* vnode = *it;
		node_id++;

		auto acc = acc_batch[node_id];
		auto ang = ang_batch[node_id];
		auto acc_float = acc.accessor<float, 1>();

		if (logging::level() >= 3) {
			cout << "large net raw acc output:" << endl;
			for (int bin_idx = 0; bin_idx < acc.size(0); bin_idx++) {
				if(delectNAN(acc_float[bin_idx])){
					cout << "input_tensor: \n" << input_tensor << endl;
					ERR("NAN detected in acc_float");
				}
				cout << "acc[" << bin_idx << "]=" << acc_float[bin_idx] << endl;
			}
		}

		// TODO: Send nn_input_images_ to drive_net, and get the acc distribution (a guassian mixture (pi, sigma, mu))

		int num_accs = 2 * ModelParams::NumAcc + 1;
		at::Tensor acc_logits_tensor = torch::ones( {num_accs}, at::kFloat);
		for (int acc_id = 0; acc_id < num_accs; acc_id++) {
			float query_acc = ped_model->GetAccelerationNoramlized(acc_id);
			float onehot_resolution = 2.0 / float(num_accs);
			int bin_idx = (int)(std::floor((query_acc + 1.0f) / onehot_resolution));
			bin_idx = min(bin_idx, num_accs-1);
			bin_idx = max(bin_idx, 0);
			acc_logits_tensor[acc_id] = acc_float[bin_idx];
			cout << "adding query acc: acc_logits_tensor_" << acc_id << "=acc_float_" << bin_idx
					<< "=" << query_acc << endl;
		}

		auto acc_probs_tensor = at::_softmax(acc_logits_tensor, 0, false);
		auto steer_probs_Tensor = at::_softmax(ang, 0, false);

		Update_prior_probs(acc_probs_tensor, steer_probs_Tensor, vnode);
	}
}

void PedNeuralSolverPrior::Compute_pref_hybrid(torch::Tensor input_tensor,
		const PedPomdp* ped_model, vector<despot::VNode*>& vnodes) {

	auto start1 = Time::now();
	at::Tensor value_batch_dummy, acc_pi_batch, acc_mu_batch, acc_sigma_batch, ang_batch;
	bool succeed = false;
	int retry_count = 0;
	while (!succeed) {
		succeed = query_srv_hybrid(vnodes.size(), input_tensor, value_batch_dummy,
				acc_pi_batch, acc_mu_batch, acc_sigma_batch, ang_batch);
		retry_count++;
		if (!succeed) {
			logi << "Root node action model query failed !!!" << endl;
			logi << "retry_count = " << retry_count << ", max_retry="
					<< max_retry_count << endl;
		}
		if (retry_count == max_retry_count)
			break;
	}
	if (!succeed) {
		cerr << "ERROR: NN query failure !!!!!" << endl;
		raise (SIGABRT);
	}
	logi << "Root node action model query succeeded" << endl;
	logd << "[Compute] Updating prior with nn outputs " << endl;
	int node_id = -1;
	for (std::vector<despot::VNode*>::iterator it = vnodes.begin();
			it != vnodes.end(); it++) {
		despot::VNode* vnode = *it;
		node_id++;

		auto acc_pi = acc_pi_batch[node_id];
		auto acc_mu = acc_mu_batch[node_id];
		auto acc_sigma = acc_sigma_batch[node_id];
		auto ang = ang_batch[node_id];

		if (logging::level() >= 3) {
			logd << "large net raw acc output:" << endl;
			auto acc_pi_float = acc_pi.accessor<float, 1>();
			auto acc_mu_float = acc_mu.accessor<float, 2>();

			for (int mode = 0; mode < acc_pi.size(0); mode++) {
				logd << "mu[" << mode << "]=" << acc_mu_float[mode][0] << endl;
			}
			for (int mode = 0; mode < acc_pi.size(0); mode++) {
				logd << "pi[" << mode << "]=" << acc_pi_float[mode] << endl;
			}
		}

		// TODO: Send nn_input_images_ to drive_net, and get the acc distribution (a guassian mixture (pi, sigma, mu))

		int num_accs = 2 * ModelParams::NumAcc + 1;
		at::Tensor acc_candiates = torch::ones( { num_accs, 1 }, at::kFloat);
		for (int acc_id = 0; acc_id < num_accs; acc_id++) {
			double query_acc = ped_model->GetAccelerationNoramlized(acc_id);
			acc_candiates[acc_id][0] = query_acc;
			logd << "adding query acc: " << acc_id << "=" << query_acc << endl;
		}

		int num_modes = acc_pi.size(0);

		auto acc_pi_actions = acc_pi.unsqueeze(0).expand(
				{ num_accs, num_modes });
		auto acc_mu_actions = acc_mu.unsqueeze(0).expand( { num_accs, num_modes,
				1 });
		auto acc_sigma_actions = acc_sigma.unsqueeze(0).expand( { num_accs,
				num_modes, 1 });

		auto acc_probs_Tensor = gm_pdf(acc_pi_actions, acc_sigma_actions,
				acc_mu_actions, acc_candiates);
		auto steer_probs_Tensor = at::_softmax(ang, 0, false);

		Update_prior_probs(acc_probs_Tensor, steer_probs_Tensor, vnode);
	}
}

void PedNeuralSolverPrior::ComputeMiniBatch(vector<torch::Tensor>& input_batch, vector<despot::VNode*>& vnodes){

	auto start = Time::now();

	torch::NoGradGuard no_grad;

	const PedPomdp* ped_model = static_cast<const PedPomdp*>(model_);
	logd << "[Compute] node depth " << vnodes[0]->depth() << endl;

	if(logging::level()>=export_image_level+1) {
		logd << "[Combine_images] vnodes[0]=" << vnodes[0]->depth() << endl;
		string level = std::to_string(vnodes[0]->depth());
		logd << "[Combine_images] exporting images" << endl;

		export_image(path_image_, "level" + level + "path");
		export_image(lane_image_, "level" + level + "lanes");
		for (int i = 0; i < num_hist_channels; i++) {
			int hist_channel = i;
			export_image(map_hist_images_[hist_channel],
					"level" + level + "_map_c" + std::to_string(hist_channel));
//			export_image(car_hist_images_[hist_channel],
//					"level" + level + "_car_c" + std::to_string(hist_channel));
		}
		inc_counter();
	}

	logd << "[Compute] num_nodes = "
						<< input_batch.size()<< endl;

	torch::Tensor input_tensor;

	input_tensor = torch::stack(input_batch, 0);

	logd << "[Compute] input_tensor dim = \n"
							<< input_tensor.sizes()<< endl;

	input_tensor= input_tensor.contiguous();

	logd << "[Compute] contiguous cuda tensor \n" <<
			input_tensor<< endl;

	logd << __FUNCTION__<<" prepare data " << Globals::ElapsedTime(start) << " s" << endl;

	logd << __FUNCTION__<<" query for "<< input_tensor.sizes() << " data " << endl;

	sync_cuda();

	Compute_pref(input_tensor, ped_model, vnodes);

	Compute_val(input_tensor, ped_model, vnodes);

	for (std::vector<despot::VNode* >::iterator it = vnodes.begin();
					it != vnodes.end(); it++) {
		despot::VNode* vnode = *it;
		vnode->prior_initialized(true);
	}

	logd << __FUNCTION__<<" " << Globals::ElapsedTime(start) << " s" << endl;
}

void PedNeuralSolverPrior::Update_prior_probs(at::Tensor& acc_probs_Tensor, at::Tensor& steer_probs_Tensor, despot::VNode* vnode){
	const PedPomdp* ped_model = static_cast<const PedPomdp*>(model_);
	cout << "acc probs = " << acc_probs_Tensor << endl;

	logd << "normalizing acc probs" << endl;

    auto acc_sum = acc_probs_Tensor.sum();
    float acc_total_prob = acc_sum.data<float>()[0];

    if(acc_total_prob < 0.01){
    	ERR("acc_total_prob < 0.01");
    }
    if( delectNAN(acc_total_prob)){
		ERR("NAN in acc_total_prob");
	}
    acc_probs_Tensor = acc_probs_Tensor / acc_total_prob;

    if(2*ModelParams::NumSteerAngle != steer_probs_Tensor.size(0))
    	ERR("");

    if (vnode->depth()==0) {
    	logd << "acc probs = " << acc_probs_Tensor << endl;
    	logd << "steer probs = " << steer_probs_Tensor << endl;
    }

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
	logd << "sharpening acc probs" << endl;

	double act_prob_total = 0;
	int sharpen_factor = 4;

	if (sharpen_factor != 1){
	//		for (int action = 0;  action < ped_model->NumActions(); action ++){
		for (int acc_ID = 0; acc_ID < 2 * ModelParams::NumAcc; acc_ID ++){
			float acc_prob = acc_probs_double[acc_ID];
			act_prob_total += std::pow(acc_prob, sharpen_factor);
		}
	}

	if(act_prob_total  < 0.01){
		cerr << "act_prob_total=" << act_prob_total << endl;
		ERR("act_prob_total unusual");
	}

	logd << "storing steer probs" << endl;

	for (int steerID = 0; steerID < 2*ModelParams::NumSteerAngle + 1; steerID++){
		float steer_prob = cal_steer_prob(steer_probs_double, steerID);
		vnode->prior_steer_probs(steerID, steer_prob);
	}

	logd << "assigning action probs to vnode " << vnode << " at depth " << vnode->depth() << endl;

	double accum_prob = 0;

//	if (vnode->depth() == 0){
//		logi << "Before assigning root action probs" << endl;
//		vnode->print_action_probs();
//	}
//		for (int action = 0;  action < ped_model->NumActions(); action ++){
	for (auto action: vnode->legal_actions()){
		logd << "legal action " << action << endl;

		int acc_ID=ped_model->GetAccelerationID(action);
		int steerID = ped_model->GetSteeringID(action);

		float acc_prob = acc_probs_double[acc_ID];
		if (sharpen_factor != 1)
			acc_prob = pow(acc_prob, sharpen_factor)/act_prob_total; // sharpen and normalize

		float steer_prob = cal_steer_prob(steer_probs_double, steerID);

		float joint_prob = acc_prob * steer_prob;

		if(delectNAN(acc_prob)){
			cout << "act_prob_total=" << act_prob_total << " acc_prob=" << acc_probs_double[acc_ID] << endl;
			ERR("NAN found in acc_prob");
		}
		if(delectNAN(joint_prob)){
			ERR("NAN found in joint_prob");
		}

		logd << "joint prob " << joint_prob << endl;

		vnode->prior_action_probs(action, joint_prob);

		accum_prob += joint_prob;

		logd << "action "<< acc_ID << " " << steerID <<
				" joint_prob = " << joint_prob
				<< " accum_prob = " << accum_prob << endl;
	}

//	if (vnode->depth() == 0){
//		logi << "Assigning root action probs" << endl;
//		vnode->print_action_probs();
//	}

	logd << "normalizing probs" << endl;

	for (auto action: vnode->legal_actions()){
		// normalize over all legal actions
//		logd << "1" << endl;

		double prob = vnode->prior_action_probs(action);
//		logd << "2" << endl;

		prob = prob / accum_prob;
//		logd << "3" << endl;

		vnode->prior_action_probs(action, prob);

		logd << action_probs_.size() << endl;

		if(delectNAN(vnode->prior_action_probs(action)))
			ERR("");

		if (vnode->depth() == 0)
			action_probs_[action] = vnode->prior_action_probs(action); // store the root action priors
	}

//	if (vnode->depth() == 0){
//		logi << "Assigning root action probs (after normalization)" << endl;
//		vnode->print_action_probs();
//	}

	logd << "done" << endl;
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

	logi << __FUNCTION__<< " " << vnodes.size() << " nodes "  << Globals::ElapsedTime(start) << " s" << endl;
}

void PedNeuralSolverPrior::ComputeMiniBatchPref(vector<torch::Tensor>& input_batch, vector<despot::VNode*>& vnodes){

	auto start = Time::now();

	torch::NoGradGuard no_grad;
//	drive_net->eval();

	// DONE: Send nn_input_images_ to drive_net, and get the policy and value output
	const PedPomdp* ped_model = static_cast<const PedPomdp*>(model_);

	logd << "[ComputeMiniBatchPref] node depth " << vnodes[0]->depth() << endl;
	logd << "[ComputeMiniBatchPref] num_nodes = "
						<< input_batch.size()<< endl;

	torch::Tensor input_tensor;
	input_tensor = torch::stack(input_batch, 0);
	logd << "[ComputeMiniBatchPref] input_tensor dim = \n"
							<< input_tensor.sizes()<< endl;
	input_tensor= input_tensor.contiguous();
	logd << "[ComputeMiniBatchPref] contiguous cuda tensor \n" <<
			input_tensor<< endl;
	logd << __FUNCTION__<<" prepare data " << Globals::ElapsedTime(start) << " s" << endl;
	logd << __FUNCTION__<<" query for "<< input_tensor.sizes() << " data " << endl;

	sync_cuda();

	auto start1 = Time::now();

	Compute_pref(input_tensor, ped_model, vnodes);

	if(logging::level()>=export_image_level){
		if (/*vnodes[0]->depth()==1*/true){
			logd << "[ComputeMiniBatchPref] vnodes[0]=" << vnodes[0]->depth() << endl;
			string level = std::to_string(vnodes[0]->depth());
			logd << "[ComputeMiniBatchPref] exporting images" << endl;

//			export_image(path_image_, "level" + level + "path");
			export_image(lane_image_, "level" + level + "lanes");

//			if (num_hist_channels>0)
//				export_image(map_hist_images_[0], "level" + level + "_map_c" + std::to_string(0));

//			for (int i = 0; i < num_hist_channels; i++) {
//				int hist_channel = i;
//				export_image(map_hist_images_[hist_channel],
//						"level" + level + "_map_c" + std::to_string(hist_channel));
////				export_image(car_hist_images_[hist_channel],
////						"level" + level + "_car_c" + std::to_string(hist_channel));
//			}
			inc_counter();
		}
	}

	logi << __FUNCTION__<<" " << Globals::ElapsedTime(start) << " s" << endl;
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

	if(logging::level()>=export_image_level + 1){
		logd << "[Combine_images] vnodes[0]=" << vnodes[0]->depth() << endl;
		string level = std::to_string(vnodes[0]->depth());
		logd << "[Combine_images] exporting images" << endl;

		export_image(path_image_, "level" + level + "path");
		export_image(lane_image_, "level" + level + "lanes");
		for (int i = 0; i < num_hist_channels; i++) {
			int hist_channel = i;
			export_image(map_hist_images_[hist_channel],
					"level" + level + "_map_c" + std::to_string(hist_channel));
//			export_image(car_hist_images_[hist_channel],
//					"level" + level + "_car_c" + std::to_string(hist_channel));
		}
		inc_counter();
	}

	logd << "[ComputeValue] num_nodes = "
						<< input_batch.size()<< endl;

	torch::Tensor input_tensor;

	input_tensor = torch::stack(input_batch, 0);

	logd << "[ComputeValue] input_tensor dim = \n"
							<< input_tensor.sizes()<< endl;

	input_tensor= input_tensor.contiguous();

	logd << "[ComputeValue] contiguous cuda tensor \n" <<
			input_tensor<< endl;

	sync_cuda();

	logd << __FUNCTION__<<" prepare data " << Globals::ElapsedTime(start) << " s" << endl;
	logd << __FUNCTION__<<" query for "<< input_tensor.sizes() << " data " << endl;

	Compute_val(input_tensor, ped_model, vnodes);
	logd << __FUNCTION__<<" " << Globals::ElapsedTime(start) << " s" << endl;
}

std::vector<ACT_TYPE> PedNeuralSolverPrior::ComputeLegalActions(const State* state, const DSPOMDP* model){
  const PomdpState* pomdp_state = static_cast<const PomdpState*>(state);
  const PedPomdp* pomdp_model = static_cast<const PedPomdp*>(model_);

  ACT_TYPE act_start, act_end;

  if (Globals::config.use_prior){
    int steer_code = world_model.hasMinSteerPath(*pomdp_state);

    if (steer_code == 1){
      int steerID = pomdp_model->GetSteerIDfromSteering(ModelParams::MaxSteerAngle);
      act_start = pomdp_model->GetActionID(steerID, 0);
      act_end = act_start + 2 * ModelParams::NumAcc + 1;
      printf("Limited legal actions: %d - %d (dir %d)\n", act_start, act_end, steer_code);
    }
    else if (steer_code == -1){
      int steerID = pomdp_model->GetSteerIDfromSteering(-ModelParams::MaxSteerAngle);
      act_start = pomdp_model->GetActionID(steerID, 0);
      act_end = act_start + 2 * ModelParams::NumAcc + 1;
      printf("Limited legal actions: %d - %d (dir %d)\n", act_start, act_end, steer_code);
    }
    else if (steer_code == 2){
      int steerID = pomdp_model->GetSteerIDfromSteering(0);
      act_start = pomdp_model->GetActionID(steerID, 0);
      act_end = act_start + 1;
      printf("Limited legal actions: %d - %d (dir %d)\n", act_start, act_end, steer_code);
    }
    else{
      act_start = 0;
      act_end = model->NumActions();
    }
  } else { // decoupled POMDP
    double steer_to_path = pomdp_model->world_model->GetSteerToPath<PomdpState>(*pomdp_state);
   
    // steer_to_path = 0; //debugging 

    act_start = pomdp_model->GetActionID(pomdp_model->GetSteerIDfromSteering(steer_to_path), 0);

    act_end = act_start + 2 * ModelParams::NumAcc + 1;

    // printf("Limited legal actions: %d - %d \n", act_start, act_end);
  }

	std::vector<ACT_TYPE> legal_actions;
  for (ACT_TYPE action = act_start; action < act_end; action++) {
    legal_actions.push_back(action);
  }

  return legal_actions;
}

void PedNeuralSolverPrior::get_history_settings(despot::VNode* cur_node, int mode, int &num_history, int &start_channel){
	if (mode == FULL){ // Full update of the input channels
		num_history = num_hist_channels;
		start_channel = 0;
		logd << "Getting FULL history for node "<< cur_node << " at depth " << cur_node->depth() << endl;
	}
	else if (mode == PARTIAL){ // Partial update of input channels and reuse old channels
		//[1,2,3], id = 0 will be filled in the nodes
		num_history = num_hist_channels - 1;
		start_channel = 1;
		logd << "Getting PARTIAL history for node "<< cur_node << " at depth " << cur_node->depth() << endl;
	}
}

void PedNeuralSolverPrior::get_history_tensors(int mode, despot::VNode* cur_node) {
	int num_history = 0;
	int start_channel = 0;

	get_history_settings(cur_node, mode, num_history, start_channel);

	if (do_print){
		Globals::lock_process();
		logi << "num_history=" << num_history << ", start_channel=" << start_channel << endl;
		Globals::unlock_process();
	}

	despot::VNode* parent = cur_node;

	int t = map_hist_.size()-1; // latest pos in tensor history

	if (do_print){
		Globals::lock_process();
		logi << "cur tensor list len " << t+1 << endl;
		Globals::unlock_process();
	}

	for (int i = start_channel ; i< start_channel + num_history ; i++){
		if (parent->parent()==NULL){ // root

//			if (!car_hist_[t].defined()){
//				cerr << "Empty car tensor hist, slot "<< t << endl;
//				ERR("");
//			}

			if (!map_hist_[t].defined()){
				cerr << "Empty map tensor hist, slot "<< t << endl;
				ERR("");
			}

//			car_hist_tensor_[i] = car_hist_[t];
			map_hist_tensor_[i] = map_hist_[t];

			if (do_print){
				Globals::lock_process();
				logi << "Using tensor hist " << t << " of len " << map_hist_.size() << " as channel " << i << endl;
				Globals::unlock_process();
			}

			t--;
		}else{

//			if (!static_cast<Shared_VNode*>(parent)->car_tensor.defined()){
//				cerr << "Empty car tensor hist, node "<< parent << endl;
//				ERR("");
//			}

			if (!static_cast<Shared_VNode*>(parent)->map_tensor.defined()){
				cerr << "Empty map tensor hist, node "<< parent << endl;
				ERR("");
			}

//			car_hist_tensor_[i] = static_cast<Shared_VNode*>(parent)->car_tensor;
			map_hist_tensor_[i] = static_cast<Shared_VNode*>(parent)->map_tensor;

			if (do_print){
				Globals::lock_process();
				logi << "Using tensors in vnode depth " << parent->depth() << " as channel " << i << endl;
				Globals::unlock_process();
			}

			parent = parent->parent()->parent();
		}
	}

	for (int i = start_channel ; i< start_channel + num_history ; i++){
//		if (!car_hist_tensor_[i].defined()){
//			cerr << "Empty car tensor, channel "<< i << endl;
//			ERR("");
//		}

		if (!map_hist_tensor_[i].defined()){
			cerr << "Empty map tensor, channel "<< i << endl;
			ERR("");
		}
	}

//	if (cur_node->depth()==0){
//		for (int i = start_channel ; i< start_channel + num_history ; i++){
//			print_full(car_hist_tensor_[i], "get_history_tensor_car_" + std::to_string(i));
//			print_full(map_hist_tensor_[i], "get_history_tensor_map_" + std::to_string(i));
//		}
//		print_full(path_tensor, "get_history_tensor_goal");
//	}
}

void PedNeuralSolverPrior::get_history(int mode, despot::VNode* cur_node, std::vector<despot::VNode*>& parents_to_fix_images,
		vector<PomdpState*>& hist_states, vector<int>& hist_ids){
	int num_history = 0;
	int start_channel = 0;

	get_history_settings(cur_node, mode, num_history, start_channel);

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

				if (do_print){
					Globals::lock_process();
					logd << "Thread " << prior_id() << " Reusing channel " << i-start_channel <<
							" to new channel " << i << " for node "<< cur_node << endl;
					Globals::unlock_process();
				}

				continue;
			}
		}
		else{
			if(car_hist_links[i-start_channel] == parent) { // can reuse the previous channel
				reuse_ids.push_back(i);

				if (do_print){
					Globals::lock_process();
					logd << "Thread " << prior_id() << " Reusing channel " << i-start_channel <<
										" to new channel " << i << " for node "<< cur_node << endl;
					Globals::unlock_process();
				}
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
//				ERR("");
//			}
//		}

		if (t >=0){
			PomdpState* car_peds_state = static_cast<PomdpState*>(as_history_in_search_.state(t));
			hist_states.push_back(car_peds_state);

			if (do_print){
				Globals::lock_process();
				logd << "Thread " << prior_id() << " Using as_history_in_search_ entry " << t <<
						" ts " << car_peds_state->time_stamp <<
						" as new channel " << hist_ids[i] <<
						" node at level " << cur_node ->depth()<< endl;
				Globals::unlock_process();
			}

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
		ERR("");
	}

//	}

	logd << "[Get_history] done " << endl;

}

void PedNeuralSolverPrior::Process_history(despot::VNode* cur_node, int mode){
	auto start = Time::now();

	logd << "Processing history, (FULL/PARTIAL) mode=" << mode << endl;

	if (history_mode == "re-process"){

		vector<int> hist_ids;
		vector<PomdpState*> hist_states;
		vector<despot::VNode*> parents_to_fix_images;

		get_history(mode, cur_node, parents_to_fix_images, hist_states, hist_ids);

		Process_states(parents_to_fix_images, hist_states, hist_ids);

	} else if (history_mode == "track"){

		get_history_tensors(mode, cur_node);
	}

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

	if (history_mode == "track"){

		if (do_print){
			Globals::lock_process();
			logi << " Processing goal image for node in depth " << cur_node->depth() << endl;
			Globals::unlock_process();
		}

		Process_path_tensor(cur_node->particles()[0]);

		get_history_tensors(FULL, cur_node); // for map and car

	} else if (history_mode == "re-process") {

		vector<int> hist_ids;
		vector<PomdpState*> hist_states;
		vector<despot::VNode*> root_to_fix_images;

		get_history(FULL, cur_node, root_to_fix_images, hist_states, hist_ids);

		Process_states(root_to_fix_images, hist_states, hist_ids);

	}

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

bool PedNeuralSolverPrior::query_srv(int batchsize, at::Tensor images, at::Tensor& t_value,
		at::Tensor& t_acc, at::Tensor& t_ang){
	int num_acc_bins = 2*ModelParams::NumAcc + 1;
	int num_steer_bins = 2*ModelParams::NumSteerAngle;

	t_value=torch::zeros({batchsize});
	t_acc=torch::zeros({batchsize, num_acc_bins});
	t_ang=torch::zeros({batchsize, num_steer_bins});

	msg_builder::TensorData message;

	message.request.tensor = std::vector<float>(images.data<float>(), images.data<float>() + images.numel());
	message.request.batchsize = batchsize;
	message.request.mode = "all";

	logd << "calling service query" << endl;

	if (nn_client_.call(message))
	{
		vector<float> value = message.response.value;
		vector<float> acc = message.response.acc;
		vector<float> ang = message.response.ang;

		logd << "acc" << endl;
		for (int id = 0 ; id< acc.size(); id++){
			if (delectNAN(acc[id]))
				return false;

			int data_id = id / num_acc_bins;
			int mode_id = id % num_acc_bins;
			logd << acc[id] << " ";
			if (mode_id == num_acc_bins -1){
				logd << endl;
			}

			t_acc[data_id][mode_id]= acc[id];
		}

		logd << "ang" << endl;
		for (int id = 0 ; id< ang.size(); id++){
			if (delectNAN(ang[id]))
				return false;

			int data_id = id / num_steer_bins;
			int bin_id = id % num_steer_bins;
			logd << ang[id] << " ";
			if (bin_id == num_steer_bins -1){
				logd << endl;
			}

			t_ang[data_id][bin_id] = ang[id];
		}

		return true;
	} else {
		return false;
	}
}

bool PedNeuralSolverPrior::query_srv_hybrid(int batchsize, at::Tensor images, at::Tensor& t_value, at::Tensor& t_acc_pi,
		at::Tensor& t_acc_mu, at::Tensor& t_acc_sigma, at::Tensor& t_ang){
	int num_guassian_modes = 5;
	int num_steer_bins = 2*ModelParams::NumSteerAngle;

	t_value=torch::zeros({batchsize});
	t_acc_pi=torch::zeros({batchsize, num_guassian_modes});
	t_acc_mu=torch::zeros({batchsize, num_guassian_modes, 1});
	t_acc_sigma=torch::zeros({batchsize, num_guassian_modes, 1});
	t_ang=torch::zeros({batchsize, num_steer_bins});

	msg_builder::TensorDataHybrid message;

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

//		logd << "value" << endl;
//		for (int i = 0 ; i< value.size(); i++){
//			logd << value[i] << " ";
//			t_value[i]= -3.4; //value[i];
//		}
//		logd << endl;

		logd << "acc_pi" << endl;
		for (int id = 0 ; id< acc_pi.size(); id++){
			if (delectNAN(acc_pi[id]))
				return false;

			int data_id = id / num_guassian_modes;
			int mode_id = id % num_guassian_modes;
			logd << acc_pi[id] << " ";
			if (mode_id == num_guassian_modes -1){
				logd << endl;
			}

			t_acc_pi[data_id][mode_id]= acc_pi[id];
		}

		logd << "acc_mu" << endl;
		for (int id = 0 ; id< acc_mu.size(); id++){
			if (delectNAN(acc_mu[id]))
				return false;

			int data_id = id / num_guassian_modes;
			int mode_id = id % num_guassian_modes;
			logd << acc_mu[id] << " ";
			if (mode_id == num_guassian_modes -1){
				logd << endl;
			}

			t_acc_mu[data_id][mode_id][0]= acc_mu[id];
		}

		logd << "acc_sigma" << endl;
		for (int id = 0 ; id< acc_sigma.size(); id++){
			if (delectNAN(acc_sigma[id]))
				return false;

			int data_id = id / num_guassian_modes;
			int mode_id = id % num_guassian_modes;
			logd << acc_sigma[id] << " ";
			if (mode_id == num_guassian_modes -1){
				logd << endl;
			}

			t_acc_sigma[data_id][mode_id][0]= acc_sigma[id];
		}

		logd << "ang" << endl;
		for (int id = 0 ; id< ang.size(); id++){
			if (delectNAN(ang[id]))
				return false;

			int data_id = id / num_steer_bins;
			int bin_id = id % num_steer_bins;
			logd << ang[id] << " ";
			if (bin_id == num_steer_bins -1){
				logd << endl;
			}

			t_ang[data_id][bin_id] = ang[id];
		}

		return true;
	} else {
		return false;
	}
}

void PedNeuralSolverPrior::Test_all_srv_hybrid(int batchsize, int num_guassian_modes, int num_steer_bins){
	cerr << "Testing all model using ROS service, bs = " << batchsize << "..." << endl;

	ros::NodeHandle n("~");

	nn_client_ = n.serviceClient<msg_builder::TensorDataHybrid>("/query");

	logd << "waiting for /query service to be ready" << endl;

	nn_client_.waitForExistence(ros::DURATION_MAX);

	for (int i =0 ;i< 2 ; i++){

		std::vector<torch::jit::IValue> inputs;

		auto images = torch::ones({batchsize, 6, 32, 32});

		auto start1 = Time::now();

		msg_builder::TensorDataHybrid message;

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
			ERR("");
		}

		logd << "nn query in " << Globals::ElapsedTime(start1) << " s" << endl;
	}
	cerr << "NN ROS servie test done." << endl;
}

void PedNeuralSolverPrior::Test_all_srv(int batchsize, int num_acc_bins, int num_steer_bins){
	cerr << "Testing all model using ROS service, bs = " << batchsize << "..." << endl;

	ros::NodeHandle n("~");

	nn_client_ = n.serviceClient<msg_builder::TensorData>("/query");

	logd << "waiting for /query service to be ready" << endl;

	nn_client_.waitForExistence(ros::DURATION_MAX);

	for (int i =0 ;i< 2 ; i++){

		std::vector<torch::jit::IValue> inputs;

		auto images = torch::ones({batchsize, 6, 32, 32});

		auto start1 = Time::now();

		msg_builder::TensorData message;

		message.request.tensor = std::vector<float>(images.data<float>(), images.data<float>() + images.numel());
		message.request.batchsize = batchsize;
		message.request.mode = "all";

		logd << "calling service query" << endl;

		if (nn_client_.call(message))
		{
			vector<float> value = message.response.value;
			vector<float> acc = message.response.acc;
			vector<float> ang = message.response.ang;

			logd << "value" << endl;
			for (int i = 0 ; i< value.size(); i++){
				logd << value[i] << " ";
			}
			logd << endl;

			logd << "acc" << endl;
			for (int id = 0 ; id< acc.size(); id++){
				int data_id = id / num_acc_bins;
				int mode_id = id % num_acc_bins;
				logd << acc[id] << " ";
				if (mode_id == num_acc_bins -1){
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
			ERR("");
		}

		logd << "nn query in " << Globals::ElapsedTime(start1) << " s" << endl;
	}
	cerr << "NN ROS servie test done." << endl;
}

void PedNeuralSolverPrior::Test_val_srv(int batchsize, int num_guassian_modes, int num_steer_bins){

	batchsize = 128;
	cerr << "Testing value model via ros service, bs = " << batchsize << endl;

	ros::NodeHandle n("~");

	nn_client_val_ = n.serviceClient<msg_builder::TensorData>("/query_val");

	logd << "waiting for /query_val service to be ready" << endl;

	nn_client_val_.waitForExistence(ros::DURATION_MAX);

	for (int i =0 ;i< 10 ; i++){

		std::vector<torch::jit::IValue> inputs;

		auto images = torch::ones({batchsize, 6, 32, 32});

		auto start1 = Time::now();

		msg_builder::TensorData message;

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
			ERR("");
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
	net = std::make_shared<torch::jit::script::Module>(torch::jit::load(path));
	net->to(at::kCUDA);
	assert(net);

	logd << "[Test_model] displaying params\n";
	Show_params(net);

	Globals::lock_process();

	for (int i =0 ;i< 1 ; i++){

		std::vector<torch::jit::IValue> inputs;

		auto images = torch::ones({batchsize, 6, 32, 32});

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

		logi << "value 0: " << value_batch_double[0][0] << endl;

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
		logi << "nn query in " << Globals::ElapsedTime(start1) << " s" << endl;
	}
	Globals::unlock_process();

}

void PedNeuralSolverPrior::Test_val_libtorch(int batchsize, int num_guassian_modes, int num_steer_bins){

	logi << "Testing libtorch value model" << endl;
//	logd << "[Test_model] Loading value model "<< path << endl;
//	auto drive_net_value = std::make_shared<torch::jit::script::Module>(torch::jit::load(path));
	drive_net_value->to(at::kCUDA);
	assert(drive_net_value);

	Globals::lock_process();
	for (int i =0 ;i< 1 ; i++){

		std::vector<torch::jit::IValue> inputs;

		auto images = torch::ones({batchsize, 6, 32, 32});

		auto start1 = Time::now();

		inputs.push_back(images.to(at::kCUDA));

		logd << "[Test_model] Query nn for "<< inputs.size() << " tensors of dim" << inputs[0].toTensor().sizes() << endl;

		auto drive_net_output = drive_net_value->forward(inputs);

		auto value_batch = drive_net_output.toTensor().cpu();
		auto value_batch_double = value_batch.accessor<float,2>();

		logi << "value 0: " << value_batch_double[0][0] << endl;
		logi << "nn query in " << Globals::ElapsedTime(start1) << " s" << endl;

		logd << "value" << endl;
		for (int i = 0 ; i< value_batch.size(0); i++){
			logd << value_batch_double[i][0] << " ";
		}
		logd << endl;
	}
	Globals::unlock_process();
}

void PedNeuralSolverPrior::Test_model(string path){

	logd << "[Test_model] Query " << endl;

	torch::NoGradGuard no_grad;

	int batchsize = 1;

	int num_guassian_modes = 5;
	int num_acc_bins = 2*ModelParams::NumAcc + 1;
	int num_steer_bins = 2*ModelParams::NumSteerAngle;

	Test_all_srv(batchsize, num_acc_bins, num_steer_bins);

//	Test_val_srv_hybrid(batchsize, num_guassian_modes, num_steer_bins);

//	Test_all_libtorch(batchsize, num_guassian_modes, num_steer_bins);

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
		ERR("");
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
		diff = diff || (state1->agents[i].pos.x != state2->agents[i].pos.x);
		diff = diff || (state1->agents[i].pos.y != state2->agents[i].pos.y);
		diff = diff || (state1->agents[i].intention != state2->agents[i].intention);

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
		ERR("");
	}
	for (int i = 0 ;i<as_history_in_search_recorded.Size(); i++){
		PomdpState* recorded_hist_state =  static_cast<PomdpState*>(as_history_in_search_recorded.state(i));
		PomdpState* hist_state =  static_cast<PomdpState*>(as_history_in_search_recorded.state(i));

		bool different = Compare_states(recorded_hist_state, hist_state);

		if( different){
			cerr << "ERROR: history "<< i << " changed after search!!!" << endl;
			static_cast<const PedPomdp*>(model_)->PrintState(*recorded_hist_state, "Recorded hist state");
			static_cast<const PedPomdp*>(model_)->PrintState(*hist_state, "Hist state");

			ERR("");
		}
	}
}

void PedNeuralSolverPrior::Get_force_steer_action(despot::VNode* vnode, int& opt_act_start, int& opt_act_end){

	if (vnode->children().size() == 0)
		return;

	try{

		// if(prior_id_ == 0)
		// 	cout  << "1" << endl;

		int opt_steer_id = -1;
		float max_prob = Globals::NEG_INFTY;

		for (int steer_id = 0; steer_id < vnode->prior_steer_probs().size(); steer_id++){
			if (vnode->prior_steer_probs(steer_id) > max_prob){

				int dec_action = 
						static_cast<const PedPomdp*>(model_)->GetActionID(steer_id, 2);

				if(vnode->children()[dec_action]->upper_bound() > -100){ // no collision 
					max_prob = vnode->prior_steer_probs(steer_id);
					opt_steer_id = steer_id;
				}
				else if(vnode->depth() == 0){
					cout << "[Get_force_steer_action] skipping collision steering id "
					 	<< steer_id << endl;
				}
			}
		}

		// if(prior_id_ == 0)
		// 	cout  << "2" << endl;

		if (opt_steer_id == -1){
			if(vnode->depth() == 0)
				cerr << "[Get_force_steer_action] all steering colliding, use default" << endl;

			// all steering colliding, use default steering
			opt_steer_id = static_cast<const PedPomdp*>(model_)->GetSteeringID(
				vnode->default_move().action);
			// ERR("");
		}

		// if(prior_id_ == 0)
		// cout  << "3" << endl;

		PomdpState * cur_state = static_cast<PomdpState*>(vnode->particles()[0]);
		double car_vel = cur_state->car.vel;

		if(car_vel >0){
			opt_act_start = static_cast<const PedPomdp*>(model_)->GetActionID(opt_steer_id, 0);
			opt_act_end = static_cast<const PedPomdp*>(model_)->GetActionID(opt_steer_id + 1, 0);
		}
		else{ // exclude dece;aration
			opt_act_start = static_cast<const PedPomdp*>(model_)->GetActionID(opt_steer_id, 0);
			opt_act_end = static_cast<const PedPomdp*>(model_)->GetActionID(opt_steer_id, 2);
		}

		// if(prior_id_ == 0)
		// 	cout  << "4" << endl;
		
		// cout << "Vnode steering prob size: " << vnode->prior_steer_probs().size() << endl;

		// cout << "Optimal prior steering: " << opt_steer_id << "(" <<
		// 		static_cast<const PedPomdp*>(model_)->GetSteering(opt_act_start)
		// 		<< ")" << endl;

		// cout << "Optimal steering id range: " << opt_act_start << "-" << opt_act_end << endl;
	
		// cout << "returning" << endl;
	}catch(std::exception e){
		cerr << "Error: " << e.what() << endl;
		ERR("");
	}
}

void PedNeuralSolverPrior::Check_force_steer(int action, int default_action){

	return; // disable this functionality

	if (abs(action - default_action) > 3 * 100){ // steering to different direction
		cout << "Searched steering too far away from default" << endl;
		prior_force_steer = true; 
	}
}

bool PedNeuralSolverPrior::Check_high_uncertainty(despot::VNode* vnode){

	return false; // disable this functionality

	double max_prob = Globals::NEG_INFTY;
	double prob2 = 0.0;
	for (int steer_id = 0; steer_id < vnode->prior_steer_probs().size(); steer_id++){
		prob2 = vnode->prior_steer_probs(steer_id);
		if (steer_id + 1 < vnode->prior_steer_probs().size())
			prob2 += vnode->prior_steer_probs(steer_id+1);
		if ( prob2 > max_prob){
			max_prob = prob2;
		}
	}
	if (max_prob < 0.4)
		return true; // uncertainty too high

//	if (abs(static_cast<const PedPomdp*>(model_)->GetSteering(vnode->default_move().action)) > 15)
//		return true;

	return false;
}

int keep_count = 0;

void PedNeuralSolverPrior::Check_force_steer(double car_heading, double path_heading, double car_vel){

	return; // disable this functionality

	double dir_diff = abs(car_heading - path_heading);

	if (dir_diff > 2* M_PI){
		cerr << "[update_cmds_naive] Angle error" << endl;
		ERR("");
	} else if (dir_diff > M_PI){
		dir_diff = 2 * M_PI - dir_diff;
	}

	bool keep_count_modified = false;

	if ( dir_diff > M_PI / 2.0){ // 30 degree difference with path
		// cout << "resetting to default action: " << default_action << ", ";
		// SolverPrior::nn_priors[0]->print_prior_actions(default_action);
		// action = default_action;

		cout << "car_heading and path_heading diff > 30 degree: resetting to default steering: " << endl;
		SolverPrior::prior_force_steer = true;
		SolverPrior::prior_force_acc = true;
		keep_count = 0;

		cout << "car dir: "<< car_heading  << endl;
		cout << "path dir: "<< path_heading << endl;
	}
	else{
		keep_count ++;
		keep_count_modified = true;
		
		if (keep_count == 3){
			SolverPrior::prior_force_steer = false;
			SolverPrior::prior_force_acc = false;
			keep_count = 0;
		}
	}

	if (car_vel < 0.01){
		SolverPrior::prior_force_steer = true;
		SolverPrior::prior_force_acc = false;

		keep_count = 0;
		SolverPrior::prior_discount_optact = 0.0;
	}
	else{

		if (!keep_count_modified)
			keep_count ++;
		if (keep_count == 3){
			SolverPrior::prior_force_steer = false;
			SolverPrior::prior_force_acc = false;
			SolverPrior::prior_discount_optact = 1.0;
			keep_count = 0;
		}
	}
}

void PedNeuralSolverPrior::update_ego_car_shape(vector<geometry_msgs::Point32> points) {
	car_shape.resize(0);
	for (auto &point: points){
		car_shape.push_back(cv::Point3f(point.x, point.y, 1.0));
	}
}

void PedNeuralSolverPrior::update_map_origin(const PomdpState* agent_state){
	map_prop_.origin = agent_state->car.pos - COORD(20.0, 20.0);

	logd << "Resetting image origin to car position: " << map_prop_.origin.x << " " <<
			map_prop_.origin.y << endl;
}

