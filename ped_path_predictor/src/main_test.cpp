#include <ped_path_predictor/ped_path_predictor.h>
#include "RVO.h"
#include <iostream>


int main(int argc, char** argv)
{
  ros::init(argc, argv, "ped_path_predictor");
  PedPathPredictor pred;

  ros::spin();
}