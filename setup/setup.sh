# Pre requisites:
# * CUDA 10.0
# * CUDNN 7
# This setu has only be testes for ubuntu18.04
##

echo "========================Installing ROS========================="
bash install_ros_melodic.sh
echo "========================Installing Torch========================="
bash install_torch.sh
echo "========================Installing OpenCV========================="
bash install_opencv4.sh

if [ -d "~/catkin_ws/src/car_hyp_despot" ] 
then
    echo "Directory car_hyp_despot exists, not cloning LeTS-Drive-SUMMIT repository." 
else
    mkdir -p catkin_ws/src
    # cd catkin_ws && catkin_make

    cd src
    git clone https://github.com/cindycia/LeTS-Drive-SUMMIT.git    
    mv LeTS-Drive-SUMMIT/* .
    mv LeTS-Drive-SUMMIT/.git .
    rm -r LeTS-Drive-SUMMIT
fi

cd catkin_ws/src/IL_contoller && pip install -r requirements.txt
