#
#!/usr/bin/env python3
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Ryan Shim, Gilbert, Tomas
import os
import math
import numpy as np
import random
import torch
import sys
import copy
from numpy.core.numeric import Infinity

from geometry_msgs.msg import Pose, Twist
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from turtlebot3_msgs.srv import DrlStep, Goal, RingGoal

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

from . import reward as rw
from ..common import utilities as util
from ..common.settings import ENABLE_BACKWARD, EPISODE_TIMEOUT_SECONDS, ENABLE_MOTOR_NOISE, UNKNOWN, SUCCESS, COLLISION_WALL, COLLISION_OBSTACLE, TIMEOUT, TUMBLE, \
                                TOPIC_SCAN, TOPIC_VELO, TOPIC_ODOM, ARENA_LENGTH, ARENA_WIDTH, MAX_NUMBER_OBSTACLES, OBSTACLE_RADIUS, LIDAR_DISTANCE_CAP, \
                                    SPEED_LINEAR_X_BOUND, SPEED_LINEAR_Y_BOUND, SPEED_ANGULAR_BOUND, THRESHOLD_COLLISION, THREHSOLD_GOAL, THREHSOLD_GOALHEADING, ENABLE_DYNAMIC_GOALS,\
                                    STEP_TIME

NUM_SCAN_SAMPLES = util.get_scan_count()
LINEAR_X = 0
LINEAR_Y = 1
ANGULAR = 2
# MAX_GOAL_DISTANCE = math.sqrt(ARENA_LENGTH**2 + ARENA_WIDTH**2)
MAX_GOAL_DISTANCE = 20.0
class DRLEnvironment(Node):
    def __init__(self):
        super().__init__('drl_environment')
        with open(os.getenv('DRLNAV_BASE_PATH') +'/tmp/drlnav_current_stage.txt', 'r') as f:
            self.stage = int(f.read())
        print(f"running on stage: {self.stage}")
        self.episode_timeout = EPISODE_TIMEOUT_SECONDS

        self.scan_topic = TOPIC_SCAN
        self.velo_topic = TOPIC_VELO
        self.odom_topic = TOPIC_ODOM
        self.goal_topic = 'goal_pose'

        self.goal_x, self.goal_y = 0.0, 0.0
        self.subgoal_distance, self.subgoal_angle = 0.0, 0.0
        # self.subgoal = [0.1, math.pi*10/180, 0.1, 0.05, 0.05]
        self.subgoal = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.robot_x, self.robot_y = 0.0, 0.0
        self.robot_x_prev, self.robot_y_prev = 0.0, 0.0
        self.robot_x_tmp, self.robot_y_tmp = 0.0, 0.0
        self.robot_heading = 0.0
        self.total_distance = 0.0
        self.robot_tilt = 0.0


        self.done = False
        self.succeed = UNKNOWN
        self.episode_deadline = Infinity
        self.reset_deadline = False
        self.clock_msgs_skipped = 0

        self.obstacle_distances = [10000] * MAX_NUMBER_OBSTACLES

        self.new_goal = False
        self.goal_angle = 0.0
        self.goal_distance = MAX_GOAL_DISTANCE
        self.initial_distance_to_goal = MAX_GOAL_DISTANCE

        self.scan_ranges = [1.0] * NUM_SCAN_SAMPLES
        self.obstacle_distance = LIDAR_DISTANCE_CAP

        self.difficulty_radius = 1
        self.local_step = 0
        self.time_sec = 0

        self.state_tmp_list = [[], [], []]

        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        qos = QoSProfile(depth=10)
        qos_clock = QoSProfile(depth=1)
        # publishers
        self.cmd_vel_pub = self.create_publisher(Twist, self.velo_topic, qos)
        # subscribers
        self.goal_pose_sub = self.create_subscription(Pose, self.goal_topic, self.goal_pose_callback, qos)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, qos)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, qos_profile=qos_profile_sensor_data)
        self.clock_sub = self.create_subscription(Clock, '/clock', self.clock_callback, qos_profile=qos_clock)
        self.obstacle_odom_sub = self.create_subscription(Odometry, 'obstacle/odom', self.obstacle_odom_callback, qos)
        # clients
        self.task_succeed_client = self.create_client(RingGoal, 'task_succeed')
        self.task_fail_client = self.create_client(RingGoal, 'task_fail')
        # servers
        self.step_comm_server = self.create_service(DrlStep, 'step_comm', self.step_comm_callback)
        self.goal_comm_server = self.create_service(Goal, 'goal_comm', self.goal_comm_callback)

    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""
    def state_queue(self, state):
        self.state_tmp_list.append(state)
        del self.state_tmp_list[0]

    def goal_pose_callback(self, msg):
        self.goal_x = msg.position.x
        self.goal_y = msg.position.y
        self.new_goal = True
        print(f"new goal! x: {self.goal_x} y: {self.goal_y}")

    def goal_comm_callback(self, request, response):
        response.new_goal = self.new_goal
        return response

    def obstacle_odom_callback(self, msg):
        # TODO: 计算障碍物距离
        if 'obstacle' in msg.child_frame_id:
            robot_pos = msg.pose.pose.position
            obstacle_id = int(msg.child_frame_id[-1]) - 1
            diff_x = self.robot_x - robot_pos.x
            diff_y = self.robot_y - robot_pos.y
            # print("move distance", math.sqrt(diff_y**2 + diff_x**2))
            self.obstacle_distances[obstacle_id] = math.sqrt(diff_y**2 + diff_x**2)
        else:
            print("ERROR: received odom was not from obstacle!")

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x - self.robot_x_tmp
        self.robot_y = msg.pose.pose.position.y - self.robot_y_tmp
        _, _, self.robot_heading = util.euler_from_quaternion(msg.pose.pose.orientation)
        self.robot_tilt = msg.pose.pose.orientation.y
        # print("robot_tilt", self.robot_tilt)

        # calculate traveled distance for logging
        if self.local_step % 32 == 0:
            self.total_distance += math.sqrt(
                (self.robot_x_prev - self.robot_x)**2 +
                (self.robot_y_prev - self.robot_y)**2)
            self.robot_x_prev = self.robot_x
            self.robot_y_prev = self.robot_y

        diff_y = self.goal_y - self.robot_y
        diff_x = self.goal_x - self.robot_x
        distance_to_goal = math.sqrt(diff_x**2 + diff_y**2)
        heading_to_goal = math.atan2(diff_y, diff_x)
        goal_angle = heading_to_goal - self.robot_heading

        while goal_angle > math.pi:
            goal_angle -= 2 * math.pi
        while goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = distance_to_goal
        # print("goal_x, goal_y", [self.goal_x, self.goal_y])
        # print("robot_x, robot_y", [self.robot_x, self.robot_y])
        # print("distance_to_goal", distance_to_goal)
        self.goal_angle = goal_angle

    # TODO: 传感器数据处理
    def scan_callback(self, msg):
        if len(msg.ranges) != NUM_SCAN_SAMPLES:
            print(f"more or less scans than expected! check model.sdf, got: {len(msg.ranges)}, expected: {NUM_SCAN_SAMPLES}")
        msg.ranges = [float('inf') if x<0.1 else x for x in msg.ranges]
        # print("scan_ranges", self.scan_ranges)
        # normalize laser values
        self.obstacle_distance = 1
        for i in range(NUM_SCAN_SAMPLES):
                self.scan_ranges[i] = np.clip(float(msg.ranges[i]) / LIDAR_DISTANCE_CAP, 0, 1)
                if self.scan_ranges[i] < self.obstacle_distance:
                    self.obstacle_distance = self.scan_ranges[i]
        # print("min obstacle_distance", self.obstacle_distance)
        self.obstacle_distance *= LIDAR_DISTANCE_CAP
        # print("scan call back obstacle_distance", self.obstacle_distance)

    def clock_callback(self, msg):
        self.time_sec = msg.clock.sec
        if not self.reset_deadline:
            return
        self.clock_msgs_skipped += 1
        if self.clock_msgs_skipped <= 10: # Wait a few message for simulation to reset clock
            return
        episode_time = self.episode_timeout
        if ENABLE_DYNAMIC_GOALS:
            episode_time = np.clip(episode_time * self.difficulty_radius, 10, 50)
        self.episode_deadline = self.time_sec + episode_time
        self.reset_deadline = False
        self.clock_msgs_skipped = 0
    
    # 重启环境
    def stop_reset_robot(self, success):
        self.cmd_vel_pub.publish(Twist()) # stop robot
        self.episode_deadline = Infinity
        self.done = True
        req = RingGoal.Request()
        # TODO: 修正里程计
        req.robot_pose_x = self.robot_x
        req.robot_pose_y = self.robot_y
        if success != 1:
            self.robot_x_tmp += self.robot_x
            self.robot_y_tmp += self.robot_y
        req.radius = np.clip(self.difficulty_radius, 0.5, 4)
        if success:
            self.difficulty_radius *= 1.01
            while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('success service not available, waiting again...')
            self.task_succeed_client.call_async(req)
        else:
            self.difficulty_radius *= 0.99
            while not self.task_fail_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('fail service not available, waiting again...')
            self.task_fail_client.call_async(req)
    # TODO: acc bound
    def calculate_acc_bound(self, spped_bound, spped):
        acc_up = float((spped_bound[1] - spped))/2
        acc_down = float((spped_bound[0] - spped))/2
        acc_b = [acc_down, acc_up]
        return acc_b

    # TODO: state
    def get_state(self, action_linear_previous, action_angular_previous, acc):
        # TODO: 状态定义
        state = copy.deepcopy(self.scan_ranges)                                             # range: [ 0, 1]
        state.append(float(np.clip((self.goal_distance / MAX_GOAL_DISTANCE), 0, 1)))     # range: [ 0, 1]
        state.append(float(self.goal_angle) / math.pi)                                      # range: [-1, 1]
        v_x = float(action_linear_previous[LINEAR_X])
        v_y = float(action_linear_previous[LINEAR_Y])
        v_w = float(action_angular_previous)
        state.append(v_x)                         
        state.append(v_y)                             
        state.append(v_w)
        state.append(float(acc[0]))
        state.append(float(acc[1]))
        state.append(float(acc[2]))                              

        # calculate acc bound
        acc_x_b = self.calculate_acc_bound(SPEED_LINEAR_X_BOUND, v_x)
        acc_y_b = self.calculate_acc_bound(SPEED_LINEAR_Y_BOUND, v_y)
        acc_w_b = self.calculate_acc_bound(SPEED_ANGULAR_BOUND, v_w)
        state.extend(acc_x_b)
        state.extend(acc_y_b)
        state.extend(acc_w_b)

        state.append(self.robot_x)
        state.append(self.robot_y)
        state.append(self.robot_heading)
        state.append(self.goal_x)
        state.append(self.goal_y)

        self.local_step += 1
        self.state_queue(state)
        if self.local_step <= 30: # Grace period to wait for simulation reset
            return state
        # Success
        # TODO: 增加成功判定
        # X__X = 0.187*(self.goal_distance - 0.5) + 0.1
        # v_x_real = v_x * X__X
        # v_y_real = v_y * 0.1
        # if self.goal_distance < THREHSOLD_GOAL and math.sqrt(pow(v_x_real,2)+pow(v_y_real,2))<=0.2:
        if self.goal_distance < THREHSOLD_GOAL:
            self.succeed = SUCCESS
        # Collision
        elif self.obstacle_distance < THRESHOLD_COLLISION:
            # print("obstacle_distance",self.obstacle_distance)
            # print("THRESHOLD_COLLISION",THRESHOLD_COLLISION)
            dynamic_collision = False
            for obstacle_distance in self.obstacle_distances:
                if obstacle_distance < (THRESHOLD_COLLISION + OBSTACLE_RADIUS + 0.05):
                    dynamic_collision = True
            if dynamic_collision:
                self.succeed = COLLISION_OBSTACLE
            else:
                self.succeed = COLLISION_WALL
        # Timeout
        elif self.time_sec >= self.episode_deadline:
            self.succeed = TIMEOUT
        # Tumble
        elif self.robot_tilt > 0.06 or self.robot_tilt < -0.06:
            self.succeed = TUMBLE
        if self.succeed is not UNKNOWN:
            print("succeed", self.succeed)
            self.stop_reset_robot(self.succeed == SUCCESS)
        return state

    def initalize_episode(self, response):
        self.initial_distance_to_goal = self.goal_distance
        response.state = self.get_state([0.0,0.0], 0.0, [0.0,0.0,0.0])
        for i in range(3):
            self.state_queue(copy.deepcopy(response.state))
        response.goal = self.subgoal
        response.reward = 0.0
        response.done = False
        response.distance_traveled = 0.0
        rw.reward_initalize(self.initial_distance_to_goal)
        return response

    # TODO: step interaction
    def step_comm_callback(self, request, response):
        if len(request.action) == 0:
            return self.initalize_episode(response)
        action = np.array(request.action)
        acc = np.array(request.acc)
        # if ENABLE_MOTOR_NOISE:
        #     request.action[LINEAR_X] += np.clip(np.random.normal(0, 0.05), -0.1, 0.1)
        #     request.action[LINEAR_Y] += np.clip(np.random.normal(0, 0.05), -0.1, 0.1)
            # request.action[ANGULAR] += numpy.clip(numpy.random.normal(0, 0.05), -0.1, 0.1)

        # TODO: actual vel constraint
        # c = 0.311*self.goal_distance - 0.0556

        action_linear_x = action[LINEAR_X]
        action_linear_y = action[LINEAR_Y]
        action_angular = action[ANGULAR]

        # Publish action cmd
        twist = Twist()
        twist.linear.x = float(action_linear_x)
        twist.linear.y = float(action_linear_y)
        twist.angular.z = float(action_angular)
        self.cmd_vel_pub.publish(twist)

        # Prepare repsonse
        response.state = self.get_state([action_linear_x,action_linear_y], action_angular, acc)
        # action_linear = math.sqrt(pow(action_linear_x,2)+pow(action_linear_y,2))
        # response.reward = rw.get_reward(self.succeed, action_linear, action_angular, self.goal_distance,
        #                                     self.goal_angle, self.obstacle_distance)
        response.goal = self.subgoal
        response.reward = rw.get_reward(self.succeed, self.state_tmp_list, self.goal_distance,
                                            self.goal_angle, self.obstacle_distance)
        response.done = self.done
        response.success = self.succeed
        response.distance_traveled = 0.0
        if self.done:
            response.distance_traveled = self.total_distance
            # Reset variables
            self.succeed = UNKNOWN
            self.total_distance = 0.0
            self.local_step = 0
            self.done = False
            self.reset_deadline = True
        if self.local_step % 200 == 0:
            print(f"Rtot: {response.reward:<8.2f}GD: {self.goal_distance:<8.2f}GA: {math.degrees(self.goal_angle):.1f}°\t", end='')
            print(f"MinD: {self.obstacle_distance:<8.2f}AlinX: {request.action[LINEAR_X]:<7.1f}AlinY: {request.action[LINEAR_Y]:<7.1f}Aturn: {request.action[ANGULAR]:<7.1f}")
        return response

def main(args=sys.argv[1:]):
    # Set seed
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    rclpy.init(args=args)
    if len(args) == 0:
        drl_environment = DRLEnvironment()
    else:
        rclpy.shutdown()
        quit("ERROR: wrong number of arguments!")
    rclpy.spin(drl_environment)
    drl_environment.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
