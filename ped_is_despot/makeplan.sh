#!/bin/sh
#rosservice call /scooter/ped_path_planner/planner/make_plan "start:
rosservice call /ped_path_planner/planner/make_plan "start:
  header:
    seq: 0
    stamp:
      secs: 0
      nsecs: 0
    frame_id: '/scooter/map'
  pose:
    position:
      x: 50
      y: 19
      z: 0.0
    orientation:
      x: 0.0
      y: 0.0
      z: 1.0
      w: 0.0
goal:
  header:
    seq: 0
    stamp:
      secs: 0
      nsecs: 0
    frame_id: '/scooter/map'
  pose:
    position:
      x: 18.0
      y: 49.0
      z: 0.0
    orientation:
      x: 0.0
      y: 0.0
      z: 0.6396
      w: 0.7687"
#tolerance: 1.0"
