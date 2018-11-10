/*
 * cluster_assoc.h
 *
 *  Created on: Sep 15, 2011
 *      Author: golfcar
 */

#ifndef cluster_assoc_H_
#define cluster_assoc_H_
#include <ros/ros.h>
#include <vector>
#include <sensor_msgs/PointCloud.h>
#include <cluster_assoc/pedestrian_array.h>
#include <cluster_extraction/clusters.h>
#include <cluster_assoc/Frame.h>
#include <geometry_msgs/Point.h>
#include <tf/transform_listener.h>
#include <tf/message_filter.h>
#include <message_filters/subscriber.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <dynamic_reconfigure/server.h>
#include "../cfg/cpp/cluster_assoc/ClusterAssocConfig.h"

using namespace message_filters;
using namespace std;

#define NN_MATCH_THRESHOLD 1.0
#define NN_ANG_MATCH_THRESHOLD 0.0873 //5 degree

class ClusterAssoc
{
public:
    ClusterAssoc(ros::NodeHandle nh, ros::NodeHandle nh_private);
    ~ClusterAssoc();

private:
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    
    message_filters::Subscriber<cluster_extraction::clusters> pedClusterSub_;
    message_filters::Subscriber<cluster_assoc::Frame> pedImageSub_;
    typedef message_filters::sync_policies::ApproximateTime<cluster_extraction::clusters, cluster_assoc::Frame> ApproxPolicy;
    message_filters::Synchronizer<ApproxPolicy> sync;

    ros::Publisher pedVisualizerPub_;
    ros::Publisher pedPub_;

    dynamic_reconfigure::Server<cluster_assoc::ClusterAssocConfig>server;

    string frame_id_, global_frame_;
    bool use_sim_time_;
    tf::TransformListener *listener_;
	
    /* cluster_assoc::pedestrian_array ped_array; */
    int latest_id;

    double dist_cost_, cost_threshold_;
    double time_out_;
    bool first_cluster;    
    
    void pedCallback(cluster_extraction::clustersConstPtr cluster_vector, cluster_assoc::FrameConstPtr ped_image_ptr);
    void publishPed(cluster_assoc::pedestrian_array& ped_array);
    bool transformPointToGlobal(std_msgs::Header header, geometry_msgs::Point32 input_point, geometry_msgs::Point32& output_point);
    bool transformGlobalToLocal(geometry_msgs::PointStamped& global_point, geometry_msgs::PointStamped& local_point);
    void transformCameraCoordinate(cluster_assoc::Frame& frame);
    void filterCluster(cluster_extraction::clusters& cluster_vector, cluster_assoc::Frame& frame, cluster_assoc::pedestrian_array& ped_array);
    void cleanUp();
    void updatePedArrayWithNewCluster(cluster_extraction::clusters& cluster_vector);
    void dynamicReconfigureCallback(cluster_assoc::ClusterAssocConfig &config, uint32_t level); 
};

#endif /* cluster_assoc_H_ */
