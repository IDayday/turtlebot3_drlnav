from ..common.settings import REWARD_FUNCTION, COLLISION_OBSTACLE, COLLISION_WALL, TUMBLE, SUCCESS, TIMEOUT, RESULTS_NUM
import numpy as np
goal_dist_initial = 0

reward_function_internal = None

# def get_reward(succeed, action_linear, action_angular, distance_to_goal, goal_angle, min_obstacle_distance):
#     return reward_function_internal(succeed, action_linear, action_angular, distance_to_goal, goal_angle, min_obstacle_distance)

def get_reward_A(succeed, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist):
        # [-3.14, 0]
        r_yaw = -1 * abs(goal_angle)

        # [-4, 0]
        r_vangular = -1 * (action_angular**2)

        # [-1, 1]
        r_distance = (2 * goal_dist_initial) / (goal_dist_initial + goal_dist) - 1

        # [-20, 0]
        if min_obstacle_dist < 0.25:
            r_obstacle = -20
        else:
            r_obstacle = 0

        # [-2 * (2.5^2), 0]
        r_vlinear = -1 * (((0.25 - action_linear) * 10) ** 2)

        reward = r_yaw + r_distance + r_obstacle + r_vlinear + r_vangular - 1

        if succeed == SUCCESS:
            reward += 5000
        elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL:
            reward -= 4000
        return float(reward)

# Define your own reward function by defining a new function: 'get_reward_X'
# Replace X with your reward function name and configure it in settings.py
# def get_reward(succeed, action_linear, action_angular, distance_to_goal, goal_angle, min_obstacle_distance):
#     return reward_function_internal(succeed, action_linear, action_angular, distance_to_goal, goal_angle, min_obstacle_distance)

# TODO: 自定义奖励函数
def get_reward_B(succeed, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist):


        # [-1, 1]
        r_distance = (2 * goal_dist_initial) / (goal_dist_initial + goal_dist) - 1

        # [-20, 0]
        if min_obstacle_dist < 0.25:
            r_obstacle = -20
        else:
            r_obstacle = 0


        reward = r_distance + r_obstacle

        if succeed == SUCCESS:
            reward += 500
        elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL or succeed == TUMBLE:
            reward -= 500
        elif succeed == TIMEOUT :
             reward -= 300
        return float(reward)


# def get_reward(succeed, action_linear_x, action_linear_y, action_angular, distance_to_goal, goal_angle, min_obstacle_distance):
#     return reward_function_internal(succeed, action_linear_x, action_linear_y, action_angular, distance_to_goal, goal_angle, min_obstacle_distance)

def get_reward_C(succeed, action_linear_x, action_linear_y, action_angular, goal_dist, goal_angle, min_obstacle_dist):
        # [-3.14, 0]
        r_yaw = -1 * abs(goal_angle)

        # [-4, 0]
        r_vangular = -1 * (action_angular**2)

        # [-1, 1]
        r_distance = (2 * goal_dist_initial) / (goal_dist_initial + goal_dist) - 1

        # [-20, 0]
        if min_obstacle_dist < 0.25:
            r_obstacle = -20
        else:
            r_obstacle = 0
        
        if action_linear_x < 0:
             r_vlinear_x = -2 * abs(action_linear_x)
        else:
             r_vlinear_x = -0.5 * action_linear_x
        r_vlinear_y = -2 * abs(action_linear_y)
        r_vlinear = r_vlinear_x + r_vlinear_y

        reward = 0.1*r_yaw + 0.5*r_distance + 0.3*r_obstacle + 0.01*r_vlinear + 0.19*r_vangular

        if succeed == SUCCESS:
            reward += 500
        elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL or succeed == TUMBLE:
            reward -= 500
        elif succeed == TIMEOUT :
             reward -= 300
        return float(reward)

def get_reward(succeed, state_tmp_list, distance_to_goal, goal_angle, min_obstacle_distance, obstacle_index):
    return reward_function_internal(succeed, state_tmp_list, distance_to_goal, goal_angle, min_obstacle_distance, obstacle_index)

def get_reward_D(succeed, state_tmp_list, goal_dist, goal_angle, min_obstacle_dist):
        pp_state = state_tmp_list[0]
        p_state = state_tmp_list[1]
        state = state_tmp_list[2]

        pp_vx = pp_state[-3]
        pp_vy = pp_state[-2]
        pp_vang = pp_state[-1]

        p_vx = p_state[-3]
        p_vy = p_state[-2]
        p_vang = p_state[-1]

        vx = state[-3]
        vy = state[-2]
        vang = state[-1]

        acc_x = (vx - p_vx)/0.1
        acc_y = (vy - p_vy)/0.1
        acc_w = (vang - p_vang)/0.1

        p_acc_x = (p_vx - pp_vx)/0.1
        p_acc_y = (p_vy - pp_vy)/0.1
        p_acc_w = (p_vang - pp_vang)/0.1

        r_acc = - 2*abs(acc_x - p_acc_x) - 1*abs(acc_y - p_acc_y) - 1*abs(acc_w - p_acc_w)

        # scan [-5, 0]
        if isinstance(state, list):
             state = np.array(state)
        else:
             state = np.array(state.tolist())
        scan = state[0:-5]
        r_scan = -(sum(1*(1-scan[:120]))/120 + sum(3*(1-scan[120:240]))/120 + sum(1*(1-scan[240:]))/120)

        # [-3.14, 0]
        r_yaw = -1 * abs(goal_angle)

        # [-4, 0]
        r_vangular = -1 * (vang**2)

        # [-1, 1]
        r_distance = (2 * goal_dist_initial) / (goal_dist_initial + goal_dist) - 1

        # [-20, 0]
        if min_obstacle_dist < 0.25:
            r_obstacle = -20
        else:
            r_obstacle = 0
        
        if vx < 0:
             r_vlinear_x = -2 * abs(vx)
        else:
             r_vlinear_x = -0.5 * vx
        r_vlinear_y = -2 * abs(vy)
        r_vlinear = r_vlinear_x + r_vlinear_y

        reward = 0.2*r_yaw + 0.5*r_distance + 0.2*r_obstacle + 0.01*r_vlinear + 0.05*r_vangular + 0.2*r_acc + 0.05*r_scan - 0.1

        if succeed == SUCCESS:
            reward += 500
        elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL or succeed == TUMBLE:
            reward -= 500
        elif succeed == TIMEOUT :
             reward -= 300
        return float(reward)

def get_reward_E(succeed, state_tmp_list, goal_dist, goal_angle, min_obstacle_dist, obstacle_index):
        pp_state = state_tmp_list[0]
        p_state = state_tmp_list[1]
        state = state_tmp_list[2]

        pp_vx = pp_state[-3]
        pp_vy = pp_state[-2]
        pp_vang = pp_state[-1]

        p_vx = p_state[-3]
        p_vy = p_state[-2]
        p_vang = p_state[-1]

        vx = state[-3]
        vy = state[-2]
        vang = state[-1]

        acc_x = (vx - p_vx)/0.1
        acc_y = (vy - p_vy)/0.1
        acc_w = (vang - p_vang)/0.1

        p_acc_x = (p_vx - pp_vx)/0.1
        p_acc_y = (p_vy - pp_vy)/0.1
        p_acc_w = (p_vang - pp_vang)/0.1

        r_acc = - 2*abs(acc_x - p_acc_x) - 1*abs(acc_y - p_acc_y) - 1*abs(acc_w - p_acc_w)

        # scan [-5, 0]
        if isinstance(state, list):
             state = np.array(state)
        else:
             state = np.array(state.tolist())
        scan = state[0:-5]
        r_scan = -(sum(1*(1-scan[:120]))/120 + sum(3*(1-scan[120:240]))/120 + sum(1*(1-scan[240:]))/120)

        # [-3.14, 0]
        r_yaw = -1 * abs(goal_angle)

        # [-4, 0]
        r_vangular = -1 * (vang**2)

        # [-1, 1]
        r_distance = (2 * goal_dist_initial) / (goal_dist_initial + goal_dist) - 1

        # # [-20, 0]
        # if min_obstacle_dist < 0.25:
        #     r_obstacle = -20
        # else:
        #     r_obstacle = 0

        # [-20, 0]
        if min_obstacle_dist < 0.25:
            r_obstacle = -20
        elif min_obstacle_dist >= 0.25 and min_obstacle_dist < 1.25:
            r_obstacle = -20 * min_obstacle_dist**2 + 50*min_obstacle_dist - 31.25
        else:
            r_obstacle = 0

        r_obstale_index = 0
        if min_obstacle_dist < 1.25:
            # if obstacle_index < 180 and obstacle_index >= 90:
            #     r_obstale_index = -10 * obstacle_index / 180
            # elif obstacle_index < 270 and obstacle_index >= 180:
            #     r_obstale_index = -10 * (360 - obstacle_index) / 180
            if obstacle_index < 180:
                r_obstale_index = -10 * obstacle_index / 180
            else:
                r_obstale_index = -10 * (360 - obstacle_index) / 180
        
        if vx < 0:
             r_vlinear_x = -2 * abs(vx)
        else:
             r_vlinear_x = -0.5 * vx
        r_vlinear_y = -2 * abs(vy)
        r_vlinear = r_vlinear_x + r_vlinear_y

        # reward = 0.2*r_yaw + 0.5*r_distance + 0.2*r_obstacle + 0.2*r_obstale_index + 0.01*r_vlinear + 0.05*r_vangular + 0.2*r_acc + 0.05*r_scan - 0.1
        reward = 0.2*r_yaw + 0.5*r_distance + 0.2*r_obstacle + 0.1*r_obstale_index + 0.01*r_vlinear + 0.05*r_vangular - 0.1

        if succeed == SUCCESS:
            reward += 500
        elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL or succeed == TUMBLE:
            reward -= 500
        elif succeed == TIMEOUT :
             reward -= 300
        return float(reward)

def reward_initalize(init_distance_to_goal):
    global goal_dist_initial
    goal_dist_initial = init_distance_to_goal

function_name = "get_reward_" + REWARD_FUNCTION
reward_function_internal = globals()[function_name]
if reward_function_internal == None:
    quit(f"Error: reward function {function_name} does not exist")
