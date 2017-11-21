#include "cluster_assoc.h"

using namespace std;

double dist(geometry_msgs::Point32 A, geometry_msgs::Point32 B)
{
    double distance = numeric_limits<double>::max();
    //if(A.x || B.x || A.y || B.y)
    distance = sqrt( (A.x -B.x) *(A.x-B.x) + (A.y -B.y) *(A.y-B.y));

    return distance;
}

ClusterAssoc::ClusterAssoc(ros::NodeHandle nh, ros::NodeHandle nh_private) : 
    nh_(nh),
    nh_private_(nh_private),
    pedClusterSub_(nh, "pedestrian_clusters", 1),
    pedImageSub_(nh, "person_tracking_output", 1),
    sync(ApproxPolicy(10), pedClusterSub_, pedImageSub_)
{
    ROS_DEBUG("Starting Pedestrian Avoidance ...");

   
    sync.registerCallback(boost::bind(&ClusterAssoc::pedCallback, this, _1, _2));

    pedPub_ = nh_.advertise<cluster_assoc::pedestrian_array>("pedestrian_array",1);
    pedVisualizerPub_ = nh_.advertise<sensor_msgs::PointCloud>("ped_visual",1);    
    
    server.setCallback(boost::bind(&ClusterAssoc::dynamicReconfigureCallback,this,_1,_2));

    nh_private_.param("global_frame", global_frame_, string("map"));
    nh_private_.param("time_out", time_out_, 2.0);

    listener_ = new tf::TransformListener(ros::Duration(10));

    dist_cost_ = 20;
    cost_threshold_ = 10;

    latest_id = 0;
    first_cluster = true;
}

ClusterAssoc::~ClusterAssoc()
{

}

void ClusterAssoc::dynamicReconfigureCallback(cluster_assoc::ClusterAssocConfig &config, uint32_t level)
{
    dist_cost_ = config.dist_cost;
    cost_threshold_ = config.cost_threhold;
}


bool ClusterAssoc::transformGlobalToLocal(geometry_msgs::PointStamped& global_point, geometry_msgs::PointStamped& local_point)
{
    /* ROS_INFO("%s",local_point.header.frame_id.c_str()); */
    try{

        listener_->transformPoint(local_point.header.frame_id, global_point, local_point);
    }
    catch(tf::TransformException& ex){
        ROS_ERROR("Received an exception trying to transform point: %s", ex.what());
        return false;
    }

    return true;
}

bool ClusterAssoc::transformPointToGlobal(std_msgs::Header header, geometry_msgs::Point32 input_point, geometry_msgs::Point32& output_point)
{
    try{
        geometry_msgs::PointStamped global_point, local_point;
        local_point.header = header;
        local_point.point.x = input_point.x;
        local_point.point.y = input_point.y;
        local_point.point.z = input_point.z;
        listener_->transformPoint(global_frame_, local_point, global_point);
        output_point.x = global_point.point.x; 
        output_point.y = global_point.point.y; 
        output_point.z = global_point.point.z;
    }
    catch(tf::TransformException& ex){
        ROS_ERROR("Received an exception trying to transform point: %s", ex.what());
        return false;
    }

    return true;
}

void ClusterAssoc::pedCallback(cluster_extraction::clustersConstPtr cluster_vector_ptr, realsense_ros_person::FrameConstPtr ped_image_ptr)
{
    ROS_INFO("entering pedestrian callback");
    cluster_extraction::clusters cluster_vector = *cluster_vector_ptr;
    realsense_ros_person::Frame frame = *ped_image_ptr;


    /*     ped_array.header = cluster_vector.header; */
    /*     ped_array.header.frame_id = global_frame_; */
    /*     first_cluster = false; */
    /* } */

    transformCameraCoordinate(frame);
    
    //print ped coordinate in camera frame
    for(int i=0; i<frame.numberOfUsers; i++)
    {
        geometry_msgs::Point32 point = frame.usersData[i].centerOfMassWorld;
        cout << "ped " << frame.usersData[i].userInfo.Id << " is at location " << point.x << ", " << point.y << endl; 
    }

    cout << endl;
    //print cluster coordinate in lidar frame
    for(int i=0; i<cluster_vector.clusters.size(); i++)
    {
        geometry_msgs::Point32 point = cluster_vector.clusters[i].centroid;
        cout << "cluster location: " << point.x << ", " << point.y << endl;
    }


    //filter clusters based on image
    cluster_assoc::pedestrian_array ped_array;
    ped_array.header.frame_id = global_frame_;
    filterCluster(cluster_vector, frame, ped_array);

    publishPed(ped_array);

    /* //update the new cluster with the existing ped_array */
    /* updatePedArrayWithNewCluster(cluster_vector); */

    /* // Add remaining clusters as new pedestrian cluster */

    /* for(size_t i=0; i<cluster_vector.clusters.size(); i++) */
    /* { */
    /*     geometry_msgs::Point32 global_point; */
    /*     bool transformed = transformPointToGlobal(cluster_vector.header, cluster_vector.clusters[i].centroid, global_point); */
    /*     if(!transformed) continue; */
    /*     cluster_assoc::pedestrian newPed; */
    /*     newPed.object_label = latest_id++; */
    /*     newPed.cluster = cluster_vector.clusters[i]; */
    /*     newPed.cluster.centroid = global_point; */
    /*     newPed.cluster.last_update = ros::Time::now(); */
    /*     newPed.local_centroid = cluster_vector.clusters[i].centroid; */
    /*     newPed.global_centroid = global_point; */

    /*     cout<< "Creating new pedestrian with id #" << latest_id << " at x:" << newPed.cluster.centroid.x << " y:" << newPed.cluster.centroid.y<<endl; */
    /*     ped_array.pd_vector.push_back(newPed); */
    /* } */

    /* cleanUp(); */
    /* publishPed(); */
    /* ROS_DEBUG_STREAM("PedCluster callback end"); */
}

/* void ClusterAssoc::updatePedArrayWithNewCluster(cluster_extraction::clusters& cluster_vector) */
/* { */
/*     if(ped_array.pd_vector.size() == 0) return; */

/*     /1* cout<<"Current ped list: "; *1/ */
/*     /1* for(size_t i = 0; i < ped_array.pd_vector.size(); i++) *1/ */
/*     /1* { *1/ */
/*     /1*     cout << ped_array.pd_vector[i].object_label << " " << *1/ */ 
/*     /1*         ped_array.pd_vector[i].cluster.centroid.x << " " << *1/ */ 
/*     /1*         ped_array.pd_vector[i].cluster.centroid.y << endl; *1/ */
/*     /1* } *1/ */

/*     for(size_t i = 0; i < cluster_vector.clusters.size(); ) */
/*     { */

/*         // Get the cluster's centroid in global position */
/*         geometry_msgs::Point32 global_point; */
/*         bool transformed = transformPointToGlobal(cluster_vector.header, cluster_vector.clusters[i].centroid, global_point); */
/*         //if(!transformed) return; */
/*         if(!transformed) continue; */

/*         cluster_assoc::pedestrian ped; */
/*         ped.cluster = cluster_vector.clusters[i]; */
/*         std_msgs::Header ped_header = cluster_vector.header; */

/*         double minCost = numeric_limits<double>::max(); */
/*         int minID = -1; */
/*         double img_score = 0, dist_score = 0; */

/*         for( size_t j = 0; j < ped_array.pd_vector.size(); j++) */
/*         { */
/*             double currDist; */
/*             currDist = dist(ped_array.pd_vector[j].cluster.centroid, global_point); */

/*             // Get the total cost with a simple linear function */
/*             double currCost = dist_cost_ * currDist; */
/*             if( (currCost < minCost)) */
/*             { */
/*                 minCost = currCost; */
/*                 minID = j; */
/*                 dist_score = dist_cost_ * currDist; */
/*             } */
/*         } */

/*         if(minID > -1) */
/*         { */
/*             /1* cout << "New cluster matched ped ID " << ped_array.pd_vector[minID].cluster.id << endl; *1/ */
/*             /1* cout << "Score details: " << dist_score << " = " << minCost<< endl; *1/ */
/*             /1* cout << "At location xy " << global_point.x << " " << global_point.y << endl; *1/ */
/*         } */

/*         if(minCost < cost_threshold_ && minID > -1) */
/*         { */
/*             // Found the matching pedestrian, update the lPedInView accordingly */
/*             ped_array.pd_vector[minID].cluster = cluster_vector.clusters[i]; */
/*             ped_array.pd_vector[minID].cluster.id = ped_array.pd_vector[minID].object_label; */
/*             ped_array.pd_vector[minID].cluster.centroid = global_point; */
/*             ped_array.pd_vector[minID].cluster.last_update = ros::Time::now(); */
/*             //keep a record of centroid position in the original LIDAR frame; */
/*             ped_array.pd_vector[minID].local_centroid = cluster_vector.clusters[i].centroid; */
/*             ped_array.pd_vector[minID].global_centroid = global_point; */
/*             cluster_vector.clusters.erase(cluster_vector.clusters.begin()+i); */
/*         } */
/*         else */
/*         { */
/*             // No match found, just increment the count */
/*             i++; */
/*         } */


/*     } */

/* } */

/* void ClusterAssoc::cleanUp() */
/* { */
/*     ROS_DEBUG_STREAM("cleanUp start"); */
/*     for(int jj=0; jj< ped_array.pd_vector.size(); ) */
/*     { */
/*         ros::Duration unseen_time = ros::Time::now() - ped_array.pd_vector[jj].cluster.last_update; */
/*         /1* ROS_DEBUG("ped with ID %d unseen time = %lf", ped_array.pd_vector[jj].cluster.id, unseen_time.toSec()); *1/ */
/*         if(unseen_time.toSec() > time_out_) */
/*         { */
/*             /1* printf("Erase ped with ID %d due to time out", ped_array.pd_vector[jj].object_label); *1/ */
/*             ped_array.pd_vector.erase(ped_array.pd_vector.begin()+jj); */
/*         } */
/*         else */
/*             jj++; */
/*     } */
/*     ROS_DEBUG_STREAM("cleanup end"); */
/* } */

void ClusterAssoc::transformCameraCoordinate(realsense_ros_person::Frame& frame)
{
    //change from camera coordinate system to ros coordinate system
    for(int i=0; i<frame.numberOfUsers; i++)
    {
        geometry_msgs::Point32 point;
        point.x = frame.usersData[i].centerOfMassWorld.z - 0.15;
        point.y = -frame.usersData[i].centerOfMassWorld.x;
        point.z = 0;
        frame.usersData[i].centerOfMassWorld = point;
    }
}

void ClusterAssoc::filterCluster(cluster_extraction::clusters& cluster_vector, realsense_ros_person::Frame& frame, cluster_assoc::pedestrian_array& ped_array)
{
    for(int i=0; i<frame.numberOfUsers; i++)
    {
        geometry_msgs::Point32 cam_center = frame.usersData[i].centerOfMassWorld;
        double min_dist = 2;
        int min_index = -1;
        geometry_msgs::Point32 min_center;

        for(int j=0; j<cluster_vector.clusters.size(); j++)
        {
            geometry_msgs::Point32 cluster_center = cluster_vector.clusters[j].centroid;
            double distance = dist(cam_center, cluster_center);

            if(distance < 0.25 && distance < min_dist)
            {
                min_dist = distance;
                min_index = j;
                min_center = cluster_center;
            }
            
           
        }
        
        if(min_index != -1)
        {
            //transform the centroid point into global coordinate frame
            geometry_msgs::Point32 global_point;
            bool transformed = transformPointToGlobal(cluster_vector.header, min_center, global_point);

            cout << "match found at: " << min_center.x << ", " << min_center.y << endl;
            cout << "match distance: " << min_dist << endl;

            //construct pedestrian message
            cluster_vector.clusters.erase(cluster_vector.clusters.begin() + min_index);
            cluster_assoc::pedestrian ped;
            ped.object_label = frame.usersData[i].userInfo.Id;
            ped.local_centroid = min_center;
            ped.global_centroid = global_point;
            ped.last_update = ros::Time::now();                
            ped_array.pd_vector.push_back(ped); 
        }

    }
}

void ClusterAssoc::publishPed(cluster_assoc::pedestrian_array& ped_array)
{
    sensor_msgs::PointCloud pc;
    pc.header.frame_id = global_frame_;
    pc.header.stamp = ros::Time::now();
    ROS_DEBUG_STREAM("publishPed start");
    for(int ii=0; ii < ped_array.pd_vector.size(); ii++)
    {
        geometry_msgs::Point32 p;
        p = ped_array.pd_vector[ii].global_centroid;
        //p.z = lPedInView.pd_vector[ii].object_label;
        p.z = 0.0;
        pc.points.push_back(p);
    }

    pedPub_.publish(ped_array);
    pedVisualizerPub_.publish(pc);

    ROS_DEBUG_STREAM("publishPed end");
}

int main(int argc, char** argv)
{
    ros::init(argc,argv,"cluster_assoc");
    ros::NodeHandle nh, nh_private;
    ClusterAssoc node(nh,nh_private);
    ros::spin();
}
