#
# Context-POMDP: a planner for autonomous driving in the SUMMIT simulator

Simulation and driving with Context-POMDP in SUMMIT (click to see video): 

[![Watch the driving video](http://img.youtube.com/vi/wrR1VQUTUEE/maxresdefault.jpg)](https://youtu.be/dNiR0z2dROg "Watch the driving video")

## Getting Started
**Information on the Installation Steps and Technical User Guides of Context-PODMP can be located on our [wiki](https://github.com/AdaCompNUS/Context-POMDP/wiki).**

## Overview
This repository contains the code for the Context-POMDP planner for driving among dense urban traffic simulated by the [**SUMMIT simulator**](https://github.com/AdaCompNUS/summit).

The algorithm was initially published in our ICRA [(paper)](https://arxiv.org/abs/1911.04074):

Cai, P., Lee, Y., Luo, Y. and Hsu, D.,. SUMMIT: A Simulator for Urban Driving in Massive Mixed Traffic. International Conference on Robotics and Automation (ICRA) 2020.

### The SUMMIT Simulator
Existing driving simulators do not capture the full complexity of real-world, unregulated, densely-crowded urban environments, such as complex road structures and traffic behaviors, and are thus insufficient for testing or training robust driving algorithms. SUMMIT aim to fill this gap. It is a high-fidelity simulator that facilitates the development and testing of crowd-driving algorithm extending CARLA to support the following additional features:

1. _Real-World Maps:_ generates real-world maps from online open sources (e.g. OpenStreetMap) to provide a virtually unlimited source of complex environments. 

2. _Unregulated Behaviors:_ agents may demonstrate variable behavioral types (e.g. demonstrating aggressive or distracted driving) and violate simple rule-based traffic assumptions (e.g. stopping at a stop sign). 

3. _Dense Traffic:_  controllable parameter for the density of heterogenous agents such as pedestrians, buses, bicycles, and motorcycles.

4. _Realistic Visuals and Sensors:_ extending off CARLA there is support for a rich set of sensors such as cameras, Lidar, depth cameras, semantic segmentation etc. 

### Context-POMDP
Context-POMDP is an expert planner in SUMMIT that explicitly reasons about interactions among traffic agents and the uncertainty on human driver intentions and types. The core is a POMDP model conditioned on humman didden states and urban road contexts. The model is solved using an efficient parallel planner, [HyP-DESPOT](https://github.com/AdaCompNUS/HyP-DESPOT). A detailed description of the model can be found in our [paper](https://arxiv.org/abs/1911.04074).

### Architecture and Components

The repository structure has the following conceptual architecture:

<a href="https://docs.google.com/drawings/d/e/2PACX-1vT_cr2QI6jZfuuwVb2UEAgqSMjvl2T1eHsO6zEiG5mp8P5byTr7OEs4BbMGw4EnIOhS-juI72XR4ZHI/pub?w=1441&amp;h=450"><img src="https://docs.google.com/drawings/d/e/2PACX-1vT_cr2QI6jZfuuwVb2UEAgqSMjvl2T1eHsO6zEiG5mp8P5byTr7OEs4BbMGw4EnIOhS-juI72XR4ZHI/pub?w=1441&amp;h=450" style="width: 500px; max-width: 100%; height: auto" title="SUMMIT Architecture" /></a>

To briefly explain the core sub-systems: 

* **Summit** the SUMMIT simulator.

* [**Summit Connector**](summit_connector/) A python package for communicating with SUMMIT. It publishes ROS topics on state and context information.

* [**Crowd Pomdp Planner**](crowd_pomdp_planner) The POMDP planner package. It receives ROS topics from the Summit_connector package and executes the belief tracking and POMDP planning loop.

* [**Car Hyp Despot**](car_hyp_despot) A static library package that implements the context-based POMDP model and the HyP-DESPOT solver. It exposes planning and belief tracking functions to be called in crowd_pomdp_planner.

