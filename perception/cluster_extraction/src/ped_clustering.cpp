/*
 * pcl_clustering.cpp
 *
 *  Created on: Jan 15, 2012
 *      Author: golfcar
 */
#include "ped_clustering.h"

ped_clustering::ped_clustering()
{
    ros::NodeHandle private_nh("~");

    private_nh.param("global_frame", global_frame_, string("map"));
    private_nh.param("laser_frame", laser_frame_id_, string("laser"));

    cloud_pub_ = nh.advertise<sensor_msgs::PointCloud>("pedestrian_cluster", 1);
    filter_pub_ = nh.advertise<sensor_msgs::PointCloud>("cloud_filtered",1);
    clusters_pub_ = nh.advertise<cluster_extraction::clusters>("pedestrian_clusters", 1);

    laser_sub_.subscribe(nh, "scan", 10);
    tf_filter_ = new tf::MessageFilter<sensor_msgs::LaserScan>(laser_sub_, tf_, global_frame_, 10);
    tf_filter_->registerCallback(boost::bind(&ped_clustering::laserCallback, this, _1));
    tf_filter_->setTolerance(ros::Duration(0.05));

    ros::spin();
}

//core function of ped_clustering, use PCL for pointcloud clustering;
void ped_clustering::clustering(const sensor_msgs::PointCloud2 &pc, sensor_msgs::PointCloud &ped_poi,
                                double tolerance, int minSize, int maxSize, bool publish)
{
    sensor_msgs::PointCloud2 pc_temp;
    pcl::PointCloud<pcl::PointXYZ> cloud_outremoved, cloud_without_line;
    pcl::PointCloud<pcl::PointXYZ> cloud_temp;
    pcl::fromROSMsg(pc, cloud_temp);

    for(size_t i=0; i<cloud_temp.points.size(); i++) cloud_temp.points[i].z = 0;

    //perform outlier filtering. Final output: cloud_outremoved
    if(cloud_temp.points.size()>0)
    {
        pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
        outrem.setInputCloud(cloud_temp.makeShared());
	    outrem.setRadiusSearch(1.0);
	    ros::NodeHandle private_nh("~");
	    int radius = 8;
	    private_nh.param("radius", radius, 8);
        outrem.setMinNeighborsInRadius(radius);
        outrem.filter(cloud_outremoved);
        pcl::toROSMsg(cloud_outremoved, pc_temp);
        ROS_DEBUG("Radius filter %d", cloud_outremoved.points.size());
    }

    //Then segmentation

    sensor_msgs::PointCloud total_clusters;
    sensor_msgs::PointCloud filtered_clusters;
    cluster_extraction::clusters clusters;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
    if(cloud_outremoved.points.size()>0)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered = cloud_outremoved.makeShared();
        int cluster_number = 0;

        std::vector<pcl::PointIndices> cluster_indices;
        extractCluster(cloud_filtered, cluster_indices, tolerance, minSize, maxSize);

        ped_poi.points.clear();

        pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
        feature_extractor.setInputCloud (cloud_filtered);


        for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
        {
            cluster_extraction::cluster cluster;
            bool is_ped = false;

            //find OBB
            // Eigen::Vector4f pcaCentroid;
            // Eigen::Matrix3f covariance;
            // pcl::compute3DCentroid(*cloud_filtered, it->indices, pcaCentroid);
            // pcl::computeCovarianceMatrixNormalized(*cloud_filtered, it->indices, pcaCentroid, covariance);
            // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
            // Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
            // eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));

            // Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
            // projectionTransform.block<3,3>(0,0) = eigenVectorsPCA.transpose();
            // projectionTransform.block<3,1>(0,3) = -1.f * (projectionTransform.block<3,3>(0,0) * pcaCentroid.head<3>());
            // pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPointsProjected (new pcl::PointCloud<pcl::PointXYZ>);
            // pcl::transformPointCloud(*cloud_filtered, *cloudPointsProjected, projectionTransform);

            // pcl::PointXYZ minPoint, maxPoint;
            // pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);

            // std::cout << maxPoint.x - minPoint.x << " " << maxPoint.y - minPoint.y << " " << maxPoint.z - minPoint.z << std::endl;

            // cluster.centroid.x = pcaCentroid[0];
            // cluster.centroid.y = pcaCentroid[1];
            // cluster.centroid.z = pcaCentroid[2];

            // cluster.width = maxPoint.x - minPoint.x;
            // cluster.height = maxPoint.y - minPoint.y;
            // cluster.depth = maxPoint.z - minPoint.z;
            // double dist = sqrt(cluster.centroid.x * cluster.centroid.x + cluster.centroid.y * cluster.centroid.y);

            pcl::PointXYZ min_point_OBB, max_point_OBB, position_OBB;
            Eigen::Matrix3f rotational_matrix_OBB;

            Eigen::Vector4f pcaCentroid;
            pcl::compute3DCentroid(*cloud_filtered, it->indices, pcaCentroid);

            feature_extractor.setIndices(boost::make_shared<std::vector<int> >(it->indices));
            feature_extractor.compute();
            feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);

            cluster.centroid.x = pcaCentroid[0];
            cluster.centroid.y = pcaCentroid[1];
            cluster.centroid.z = pcaCentroid[2];

            cluster.width = max_point_OBB.x - min_point_OBB.x;
            cluster.height = max_point_OBB.y - min_point_OBB.y;
            cluster.depth = max_point_OBB.z - min_point_OBB.z;
            double dist = sqrt(cluster.centroid.x * cluster.centroid.x + cluster.centroid.y * cluster.centroid.y);

            /* std::cout << cluster.centroid.x << " " << cluster.centroid.y << std::endl;; */
            /* std::cout << cluster.width << " " << cluster.height << " " << cluster.depth << std::endl << std::endl; */

            if(cluster.width < 3.0 && cluster.height < 3.0 && dist < 5)
            	// ped_poi.points.push_back(p);
                is_ped = true;

            std::vector<geometry_msgs::Point32> cluster_points;
            for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
            {
            	pcl::PointXYZ pclpt_temp;
            	pclpt_temp = cloud_filtered->points[*pit];
            	geometry_msgs::Point32 p_temp;
            	p_temp.x = pclpt_temp.x;
            	p_temp.y = pclpt_temp.y;
            	p_temp.z = pclpt_temp.z;
            	cluster_points.push_back(p_temp);
            }
            total_clusters.points.insert(total_clusters.points.end(), cluster_points.begin(),cluster_points.end());
            if(is_ped)
            {
                cluster.points = cluster_points;
                clusters.clusters.push_back(cluster);
                filtered_clusters.points.push_back(cluster.centroid);
            }
        }
    }
    total_clusters.header = pc.header;
    clusters.header = pc.header;
    filtered_clusters.header = pc.header;
    cloud_pub_.publish(total_clusters);
    clusters_pub_.publish(clusters);
    filter_pub_.publish(filtered_clusters);

}

void ped_clustering::scanCallback(const sensor_msgs::PointCloud2ConstPtr& pc2)
{
    double clusterTolerance = 0.7;
    int minClusterSize = 1;
    int maxClusterSize = 1000;
    sensor_msgs::PointCloud2 final_pc2 = *pc2;
    sensor_msgs::PointCloud temp;
    clustering(final_pc2, temp, clusterTolerance, minClusterSize, maxClusterSize, true);
}

inline geometry_msgs::Point32 getCenterPoint(geometry_msgs::Point32 start_pt, geometry_msgs::Point32 end_pt)
{
    geometry_msgs::Point32 center;
    center.x = (start_pt.x + end_pt.x)/2;
    center.y = (start_pt.y + end_pt.y)/2;
    center.z = (start_pt.z + end_pt.z)/2;
    return center;
}

inline double distance(double x1, double y1, double x2, double y2)
{
    double a = x1 - x2;
    double b = y1 - y2;
    return sqrt(a*a + b*b);
}

void ped_clustering::laserCallback(const sensor_msgs::LaserScanConstPtr& scan_in)
{
    sensor_msgs::PointCloud pc, global_pc;

    try{projector_.transformLaserScanToPointCloud(laser_frame_id_, *scan_in, pc, tf_);}
    catch (tf::TransformException& e){ROS_INFO_STREAM(e.what());return;}

    //interleave based on map frame
    //based on initial impression, it is quite good. Points appears to be more stable

    if(scan_in->header.seq%2)
    {
    	try{tf_.transformPointCloud(global_frame_, pc, global_pc);}
        catch(tf::TransformException& e){ROS_INFO_STREAM(e.what());return;}
        laser_global_pc_.points.clear();
        laser_global_pc_ = global_pc;
        return;
    }
    else
    {
    	laser_global_pc_.header.stamp = scan_in->header.stamp;
    	sensor_msgs::PointCloud pc_temp;
    	try{tf_.transformPointCloud(scan_in->header.frame_id, laser_global_pc_, pc_temp);}
    	catch(tf::TransformException& e){ROS_INFO_STREAM(e.what());return;}
    	pc.points.insert(pc.points.end(), pc_temp.points.begin(), pc_temp.points.end());
    }
    //cout<<"laser_scan seq "<<scan_in->header.seq<<endl;


    //A workaround with an apparent bug where sometimes It shows a different
    //cluster each with single point although there are separated by less than the threshold value
    //perform interleave action by default
    //pc.points.insert(pc.points.end(),pc.points.begin(),pc.points.end());
    sensor_msgs::PointCloud2 pc2;
    sensor_msgs::PointCloud output;
    // try{tf_.transformPointCloud(global_frame_, pc, temp);}
    // catch(tf::TransformException& e){ROS_INFO_STREAM(e.what());return;}
    sensor_msgs::convertPointCloudToPointCloud2(pc,pc2);
    clustering(pc2, output, 0.2, 2, 500, true);

}

void ped_clustering::extractCluster(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered, std::vector<pcl::PointIndices>& cluster_indices,
                                    double clusterTolerance, int minSize,int maxSize)
{
    /* ROS_INFO("cloud_filter size(); cloud_filtered->points.size() %d, %d", (int)cloud_filtered->size(), (int)cloud_filtered->points.size()); */
    if(cloud_filtered->points.size()<1) return;
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud_filtered);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (clusterTolerance);//0.5);
    ec.setMinClusterSize (minSize);//3);
    ec.setMaxClusterSize (maxSize);//100);
    ec.setSearchMethod (tree);
    ec.setInputCloud( cloud_filtered);

    cluster_indices.clear();
    ec.extract (cluster_indices);
}

int main (int argc, char** argv)
{
    ros::init(argc, argv, "ped_clustering");
    ped_clustering ped_cluster;
    return (0);
}
