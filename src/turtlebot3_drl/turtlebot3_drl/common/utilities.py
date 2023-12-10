from queue import Empty
from turtlebot3_msgs.srv import DrlStep
from turtlebot3_msgs.srv import Goal
from std_srvs.srv import Empty
import os
import time
import rclpy
import torch
import numpy
from ..common.settings import REWARD_FUNCTION, COLLISION_OBSTACLE, COLLISION_WALL, TUMBLE, SUCCESS, TIMEOUT, RESULTS_NUM

import xml.etree.ElementTree as ET

try:
    with open(os.getenv('DRLNAV_BASE_PATH') +'/tmp/drlnav_current_stage.txt', 'r') as f:
        stage = int(f.read())
except FileNotFoundError:
    print("\033[1m" + "\033[93m" + "Make sure to launch the gazebo simulation node first!" + "\033[0m}")

def check_gpu():
    print("gpu torch available: ", torch.cuda.is_available())
    if (torch.cuda.is_available()):
        print("device name: ", torch.cuda.get_device_name(0))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def step(agent_self, action, previous_action, acc):
    req = DrlStep.Request()
    req.action = action
    req.previous_action = previous_action
    req.acc = acc

    while not agent_self.step_comm_client.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info('env step service not available, waiting again...')
    future = agent_self.step_comm_client.call_async(req)

    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            if future.result() is not None:
                res = future.result()
                return res.state, res.goal, res.reward, res.done, res.success, res.distance_traveled
            else:
                agent_self.get_logger().error(
                    'Exception while calling service: {0}'.format(future.exception()))
                print("ERROR getting step service response!")

def init_episode(agent_self):
    state, goal, _, _, _, _ = step(agent_self, [], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    return state, goal

def get_goal_status(agent_self):
    req = Goal.Request()
    while not agent_self.goal_comm_client.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info('new goal service not available, waiting again...')
    future = agent_self.goal_comm_client.call_async(req)

    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            if future.result() is not None:
                res = future.result()
                return res.new_goal
            else:
                agent_self.get_logger().error(
                    'Exception while calling service: {0}'.format(future.exception()))
                print("ERROR getting   service response!")

def wait_new_goal(agent_self):
    while(get_goal_status(agent_self) == False):
        print("Waiting for new goal... (if persists: reset gazebo_goals node)")
        time.sleep(1.0)

def pause_simulation(agent_self, real_robot):
    if real_robot:
        return
    while not agent_self.gazebo_pause.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info('pause gazebo service not available, waiting again...')
    future = agent_self.gazebo_pause.call_async(Empty.Request())
    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            return

def unpause_simulation(agent_self, real_robot):
    if real_robot:
        return
    while not agent_self.gazebo_unpause.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info('unpause gazebo service not available, waiting again...')
    future = agent_self.gazebo_unpause.call_async(Empty.Request())
    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            return

def translate_outcome(outcome):
    if outcome == SUCCESS:
        return "SUCCESS"
    elif outcome == COLLISION_WALL:
        return "COLL_WALL"
    elif outcome == COLLISION_OBSTACLE:
        return "COLL_OBST"
    elif outcome == TIMEOUT:
        return "TIMEOUT"
    elif outcome == TUMBLE:
        return "TUMBLE"
    else:
        return f"UNKNOWN: {outcome}"

# --- Environment ---

def euler_from_quaternion(quat):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quat = [x, y, z, w]
    """
    x = quat.x
    y = quat.y
    z = quat.z
    w = quat.w

    sinr_cosp = 2 * (w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = numpy.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w*y - z*x)
    if sinp < -1:
        sinp = -1
    if sinp > 1:
        sinp = 1
    pitch = numpy.arcsin(sinp)

    siny_cosp = 2 * (w*z + x*y)
    cosy_cosp = 1 - 2 * (y*y + z*z)
    yaw = numpy.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def get_scan_count():
    # TODO: 需要重定位到dog模型,这里写的绝对路径
    # tree = ET.parse(os.getenv('DRLNAV_BASE_PATH') + '/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf')

    # root = tree.getroot()
    # for link in root.find('model').findall('link'):
    #     if link.get('name') == 'base_scan':
    #         return int(link.find('sensor').find('ray').find('scan').find('horizontal').find('samples').text)

    # export DOG_BASE_PATH=/home/dayday/project/cyberdog_sim
    # tree = ET.parse(os.getenv('DOG_BASE_PATH')+'/src/cyberdog_ros2/cyberdog_robot/cyberdog_description/xacro/gazebo.xacro')
    tree = ET.parse('src/turtlebot3_simulations/turtlebot3_gazebo/xacro/gazebo.xacro')
    root = tree.getroot()
    for link in root.findall('gazebo'):
        if link.get('reference') == 'lidar_link':
            print("find lidar_link")
            scan_num = int(link.find('sensor').find('ray').find('scan').find('horizontal').find('samples').text)
            print("scan num",scan_num)
            return int(link.find('sensor').find('ray').find('scan').find('horizontal').find('samples').text)
    # return int(360)    


def get_simulation_speed(stage):
    tree = ET.parse(os.getenv('DRLNAV_BASE_PATH') + '/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_drl_stage' + str(stage) + '/burger.model')
    root = tree.getroot()
    return int(root.find('world').find('physics').find('real_time_factor').text)


