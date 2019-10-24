#
# LeTS-Drive with SUMMIT simulator integration

Simulation and driving in SUMMIT (click to see video): 

[![Watch the driving video](https://img.youtube.com/vi/bQjcd-NBdIg/0.jpg)](https://youtu.be/wrR1VQUTUEE "Watch the driving video")

## Overview
Existing driving simulators do not capture the full complexity of real-world, unregulated, densely-crowded urban environments, such as complex road structures and traffic behaviors, and are thus insufficient for testing or training robust driving algorithms. We aim to fill this gap.  SUMMIT is a high-fidelity simulator that facilitates the development and testing of crowd-driving algorithm extending CARLA to support the following additional features:

1. _Real-World Maps:_ generates real-world maps from online open sources (e.g. OpenStreetMap) to provide a virtually unlimited source of complex environments. 

2. _Unregulated Behaviors:_ agents may demonstrate variable behavioral types (e.g. demonstrating aggressive or distracted driving) and violate simple rule-based traffic assumptions (e.g. stopping at a stop sign). 

3. _Dense Traffic:_  controllable parameter for the density of heterogenous agents such as pedestrians, buses, bicycles, and motorcycles.

4. _Realistic Visuals and Sensors:_ extending off CARLA there is support for a rich set of sensors such as cameras, Lidar, depth cameras, semantic segmentation etc. 

This repository contains all algorithmic elements for reproducing **LeTS-Drive** [(paper)](https://arxiv.org/abs/1905.12197) in heterogenous traffic simulated by the **SUMMIT simulator** [(paper)](https://www.dropbox.com/s/fs0e9j4o0r80e82/SUMMIT.pdf?dl=0). The repository structure has the following conceptual architecture:

<a href="https://docs.google.com/drawings/d/e/2PACX-1vR__3TWU8FzVXUJf2J8QxnrqaTkhlEjEd9OMxWbRAwE37swNKLNegU3CaTXAZFK7Uar2qOdDDdnYqv_/pub?w=900&h=360"><img src="https://docs.google.com/drawings/d/e/2PACX-1vR__3TWU8FzVXUJf2J8QxnrqaTkhlEjEd9OMxWbRAwE37swNKLNegU3CaTXAZFK7Uar2qOdDDdnYqv_/pub?w=900&h=360" style="width: 500px; max-width: 100%; height: auto" title="SUMMIT Architecture" /></a>

To briefly explain the core sub-systems: 

* **Summit Server** SUMMIT server for rendering environment.

* **Summit Connector** A python package for communicating with SUMMIT, constructing the scene, controlling the traffic, and processing state and context information.

* **Crowd Pomdp Controller** A wrapping over the POMDP planner. It receives information from the simulator and run belief tracking and POMDP planning.

* **IL Controller** The neural network learner for imitation learning (dashed lines denote that once trained these networks can be used to control the driver).

**Information on the Installation Steps or Technical User Guide of SUMMIT can be located on our [wiki](https://github.com/AdaCompNUS/LeTS-Drive-SUMMIT/wiki).**
