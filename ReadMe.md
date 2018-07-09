[TOC]

# Introduction
Deep reinforcement learning for UAV in Gazebo simulation environment

## Environment:
Gazebo & pixhawk & ROS SITL(software in the loop) simulation:

## DRL:
+ state = [Pos_z, Vel_z, Thrust]

+ action = {0, 1, 200} 

	// 0: decrease thrust; 

	// 1: increase thrust; 
    
	// 200: environment needs to be restarted(manually selected)!

+ reward:

	if(14.7 < Pos_z < 15.3)	reward = 1

	else reward = 0

+ Deep Network: 4 full connection layers

## Destination:
UAV hovering at the altitude of 15m.


# How to run UAV_DRL(talker.py) in Gazebo environment:
## 1) run pixhawk connection:
```
source ~/src/mavros_ws/devel/setup.bash
roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"
```


## 2) run pixhawk&gazebo:
```
cd ~/src/Firmware
make posix_sitl_default gazebo
```

## 3) run state_machine: in branch flight_test(for PID fine-tunning!)
```
source ~/src/attitude_controller/devel/setup.bash
rosrun state_machine offb_simulation_test
```


## 4) switch pixhawk to proper mode:
```
source ~/src/mavros_ws/devel/setup.bash
rosrun mavros mavsafety arm
rosrun mavros mavsys mode -c OFFBOARD
```

## 5) run DRL:
```
source ~/src/DRL_node_ROS/devel/setup.bash
rosrun beginner_tutorials talker.py
```




# Constraints:
## 1) Gazebo environment:
Thrust: [0.40, 0.78]

Vel_z: [-3.0, 3.0]

## 2) UAV_DRL(talker.py):
if vel_z > 3.0 => force action=0(increase thrust)

if vel_z < -3.0 => force action=1(decrease thrust)

# Others:
## 1) time delay of restart between pixhawk and DRL:

~1.14s
