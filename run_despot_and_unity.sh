#!/bin/bash

for i in $(seq 2 100)
do
	#/home/yuanfu/Unity/DESPOT-Unity/car_scenario1.x86_64 &
	#/home/yuanfu/Unity/DESPOT-Unity/car.x86_64 &
	#/home/yuanfu/Unity/DESPOT-Unity/car_scenario0.x86_64 &
	/home/yuanfu/Unity/DESPOT-Unity/car_scenario3.x86_64 &
	#time timeout 200 roslaunch ped_is_despot is_despot.launch &> result/trial-$i.txt
	timeout 200 roslaunch ped_is_despot is_despot.launch &> result/trial-$i.txt
	sleep 1
	#pkill car_scenario1.x
	#pkill car.x86_64
	#pkill car_scenario0.x
	pkill car_scenario3.x
	sleep 1
done

