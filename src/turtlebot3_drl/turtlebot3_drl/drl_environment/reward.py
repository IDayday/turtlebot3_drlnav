from ..common.settings import REWARD_FUNCTION, COLLISION_OBSTACLE, COLLISION_WALL, TUMBLE, SUCCESS, TIMEOUT, RESULTS_NUM

goal_dist_initial = 0

reward_function_internal = None

def get_reward(succeed, action_linear_x, action_linear_y, action_angular, distance_to_goal, goal_angle, min_obstacle_distance):
    return reward_function_internal(succeed, action_linear_x, action_linear_y, action_angular, distance_to_goal, goal_angle, min_obstacle_distance)

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


        r_vlinear = -1 * action_linear_x - 2*action_linear_y

        reward = 0.1*r_yaw + 0.5*r_distance + 0.3*r_obstacle + 0.01*r_vlinear + 0.19*r_vangular

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
