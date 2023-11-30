#!/bin/bash
ros2 topic pub --once /$3/goal_pose geometry_msgs/msg/Pose "{position: {x: $1, y: $2, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 0.0}}"
