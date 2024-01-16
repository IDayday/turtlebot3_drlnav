#!/usr/bin/env bash

source ./install/setup.bash

ros2 run turtlebot3_drl environment  &

sleep 3

ros2 run turtlebot3_drl test_agent ddpg 'ddpg_100_stage_4' 2500  &

sleep 3

ros2 run turtlebot3_drl gazebo_goals_test & 

sleep 3

wait 

killall -9 test_agent & killall -9 environment & killall -9 gazebo_goals_test