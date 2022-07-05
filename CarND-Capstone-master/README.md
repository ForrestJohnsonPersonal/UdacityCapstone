[//]: # (Image References)

[image1]: ./imgs/carlaAnimation.gif "Simulation Animation"

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

# Udacity Self-Driving Car Capstone Project
The capstone project for the final term of the Udacity self-driving car nanodegree is a system integration project that puts together perception, planning and control modules. The Robot Operating System (ROS) middleware is used as the platform for integration.

## Team: BluePillRedPill
This project is a group effort - the 6 of us are part of the __BluePillRedPill__ team. The group effort was both rewarding and challenging. While it was interesting to meet (virtually) with people from different parts of the world interested in self-driving cars, it was also challenging to maintain communication across different time zones. The team members are listed below:

* __Team Lead:__ Swarooph Seshadri (swarooph.nirmal@gmail.com)
* Anand Raja (araja@mathworks.com)
* Neel Kowdley (nkowdley@gmail.com)
* Ahmed Madbouly (ahmedmadbouly88@yahoo.com)
* Soumen De (soumende@yahoo.com)
* Zhi Chai (zhichaivt@gmail.com)

In order to efficiently tackle the project, we divided ourselves into sub-teams as follows:
* _Perception:_ Anand and Zhi
* _Planning:_ Swarooph and Soumen
* _Control:_ Neel and Ahmed

Once each of the modules were separately implemented by the sub-teams, multiple people ended up working on integrating the modules, which is where a significant portion of time was spent.

## Project Details
### Perception
The perception module consists of two components.
1. Using the current pose of the car and prior knowledge of stop lines for traffic lights, determine the closest stop line ahead of the car. This is mostly contained in `tl_detector.py`.
2. Using the image captured by on-board camera to detect and classify the traffic light. This information is then published for the planning module to react. Two trained Single-Shot Detector (SSD) detectors (one for simulation and one for real-world) are used in the perception module. This is contained in `tl_classifier.py`. The trained model was obtained from [https://github.com/iburris/Traffic-Light-Classification].

We encountered several performance issues with running the classifier on the provided _Workspaces_. Some of these were resolved by running the perception loop at a slower rate (10Hz) than the camera rate (30 Hz). We measured the performance of the classifier at around 0.07s, which informed our choice for the perception rate. Still however, at different times, turning on the camera would cause a lag in the response of the system, leading the car to veer off the road. This was also reported by several others on Udacity forums. We later also tried docker runs, where the issue showed up.

### Planning
The goal of the planning module is to process the incoming waypoint list to adjust the velocity at each waypoint. Each waypoint is initially associated with a linear velocity (the speed limit) and an angular velocity for the lane. The detected traffic light from the perception module is then processed to change the desired linear velocity profile of the car. The velocity profile is modulated to decelerate when a red light is ahead. This has the effect of stopping the car at or before the stop line for the traffic light. See `waypoint_updater.py` for more details.

### Control
The goal of the controls module is to  realize the velocity profile provided by the planning module. This is implemented inside `dbw_node.py` and `twist_controller.py`. To do this, we subscribe to the dbw_enabled topic which turns on and off the controller. As part of this controller, we enable our yaw and PID controller from the dbw_node.py. Steering is obtained using the Yaw Controller, where we pass in the linear, angular and current velocities. Throttle is obtained by using our PID controller.  For more information on this, please look at our `twist_controller.py`. In addition, an out of the box AutoWare based path follower code that implemented the pure-pursuit algorithm was used unaltered based on the project instructions.

Animation of the simulation can be seen below:
![alt text][image1]

## Installation and Running Instructions:

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
