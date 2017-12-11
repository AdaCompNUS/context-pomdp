#!/bin/bash

for i in $(seq 0 3)
do
	/home/yuanfu/Unity/DESPOT-Unity/car.x86_64 &
	roslaunch ped_is_despot is_despot.launch
	sleep 2
	pkill car.x86_64
	sleep 10
done
