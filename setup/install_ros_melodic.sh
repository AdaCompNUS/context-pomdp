sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    dirmngr \
    gnupg2

# setup sources.list
sudo echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros-latest.list

# setup keys
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup environment
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# install ros packages
export ROS_DISTRO=melodic
sudo apt-get update && sudo apt-get install -y \
    ros-melodic-desktop-full ros-melodic-navigation

# bootstrap rosdep
rosdep init \
    && rosdep update

# install bootstrap tools
sudo apt-get update && sudo apt-get install --no-install-recommends -y \
    python-rosdep \
    python-rosinstall \
    python-catkin-tools \
    python-vcstools
