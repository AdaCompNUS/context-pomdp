# Pre requisites:
# * CUDA 10.0
# * CUDNN 7
# This setu has only be testes for ubuntu18.04
##

print("========================Installing ROS=========================")
bash install_ros_melodic.sh
print("========================Installing Torch=========================")
bash install_torch.sh
print("========================Installing OpenCV=========================")
bash install_opencv4.sh

mkdir -p catkin_ws/src
cd catkin_ws && catkin_make

if [ -d "~/catkin_ws/src/car_hyp_despot" ] 
then
    echo "Directory car_hyp_despot exists, not cloning LeTS-Drive-SUMMIT repository." 
else
    cd src
    git clone https://github.com/cindycia/LeTS-Drive-SUMMIT.git    
    mv LeTS-Drive-SUMMIT/* .
    mv LeTS-Drive-SUMMIT/.git .
fi

cd catkin_ws/src/IL_contoller && pip install -r requirements.txt
