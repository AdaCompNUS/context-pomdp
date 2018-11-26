/*
 * ped_clustering.h
 *
 *  Created on: Jun 1, 2012
 *      Author: demian
 */

#include <ros/ros.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
//#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/octree/octree_search.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
//#include <fmutil/fm_math.h>
#include <std_msgs/Int64.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/search/impl/search.hpp>

#include <geometry_msgs/PolygonStamped.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <laser_geometry/laser_geometry.h>
#include <tf/message_filter.h>
#include <message_filters/subscriber.h>
#include <cluster_extraction/cluster.h>
#include <cluster_extraction/clusters.h>


using namespace std;


class ped_clustering
{
public:
    ped_clustering();
private:
    void scanCallback(const sensor_msgs::PointCloud2ConstPtr& pc2);
    void clustering(const sensor_msgs::PointCloud2& pc2, sensor_msgs::PointCloud &ped_poi,
                    double tolerance, int minSize, int maxSize, bool publish);
    void filterLines(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>& cloud_out);
    void extractCluster(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered, std::vector<pcl::PointIndices>& cluster_indices,
                        double clusterTolerance, int minSize,int maxSize);
    void laserCallback(const sensor_msgs::LaserScanConstPtr& scan_in);
    message_filters::Subscriber<sensor_msgs::LaserScan> laser_sub_;
    ros::Publisher cloud_pub_, filter_pub_, clusters_pub_;

    tf::TransformListener tf_;
    tf::MessageFilter<sensor_msgs::LaserScan> * tf_filter_;
    laser_geometry::LaserProjection projector_;
    string laser_frame_id_, global_frame_;
    ros::NodeHandle nh;
    sensor_msgs::PointCloud laser_global_pc_;
};



