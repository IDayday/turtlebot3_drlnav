# ===================================================================== #
#                           GENERAL SETTINGS                            #
# ===================================================================== #

ENABLE_BACKWARD          = False    # Enable backward movement of the robot
ENABLE_STACKING          = False    # Enable processing multiple consecutive scan frames at every observation step
ENABLE_VISUAL            = False    # Meant to be used only during evaluation/testing phase
ENABLE_TRUE_RANDOM_GOALS = False    # If false, goals are selected semi-randomly from a list of known valid goal positions
ENABLE_DYNAMIC_GOALS     = False    # If true, goal difficulty (distance) is adapted according to current success rate
MODEL_STORE_INTERVAL     = 100      # Store the model weights every N episodes
GRAPH_DRAW_INTERVAL      = 10       # Draw the graph every N episodes (drawing too often will slow down training)
GRAPH_AVERAGE_REWARD     = 10       # Average the reward graph over every N episodes


# ===================================================================== #
#                         ENVIRONMENT SETTINGS                          #
# ===================================================================== #

# --- SIMULATION ENVIRONMENT SETTINGS ---
REWARD_FUNCTION = "A"           # Defined in reward.py
EPISODE_TIMEOUT_SECONDS = 50    # Number of seconds after which episode timeout occurs

TOPIC_SCAN = 'scan'
TOPIC_VELO = 'cmd_vel'
TOPIC_ODOM = 'odom_out'

EPISODE_TIMEOUT_SECONDS     = 50    # Number of seconds after which episode timeout occurs
ARENA_LENGTH                = 16   # meters 地图大小/希望设置的目标范围大小
ARENA_WIDTH                 = 16   # meters 地图大小
SPEED_LINEAR_MAX            = 1.5  # m/s 线速度最大值
SPEED_ANGULAR_MAX           = 1.0   # rad/s 角速度最大值

LIDAR_DISTANCE_CAP          = 3.5   # meters 雷达探测范围
THRESHOLD_COLLISION         = 0.15  # meters 障碍物碰撞判定距离
THREHSOLD_GOAL              = 0.20  # meters 目标抵达判定距离

OBSTACLE_RADIUS             = 0.5  # meters 障碍物半径（圆柱体）
MAX_NUMBER_OBSTACLES        = 6     # 最多障碍物数量
ENABLE_MOTOR_NOISE          = False # Add normally distributed noise to motor output to simulate hardware imperfections 电机噪声

# --- REAL ROBOT ENVIRONMENT SETTINGS ---  真实物理环境参数
REAL_TOPIC_SCAN  = 'scan'
REAL_TOPIC_VELO  = 'cmd_vel'
REAL_TOPIC_ODOM  = 'odom_out'

REAL_N_SCAN_SAMPLES         = 360    # LiDAR density count your robot is providing
REAL_ARENA_LENGTH           = 50   # meters
REAL_ARENA_WIDTH            = 50   # meters
REAL_SPEED_LINEAR_MAX       = 1.5  # in m/s
REAL_SPEED_ANGULAR_MAX      = 1.0   # in rad/s

REAL_LIDAR_CORRECTION       = 0.40  # meters, subtracted from the real LiDAR values 激光雷达盲区？修正值？
REAL_LIDAR_DISTANCE_CAP     = 3.5   # meters, scan distances are capped this value
REAL_THRESHOLD_COLLISION    = 0.15  # meters, minimum distance to an object that counts as a collision
REAL_THRESHOLD_GOAL         = 0.20  # meters, minimum distance to goal that counts as reaching the goal


# ===================================================================== #
#                       DRL ALGORITHM SETTINGS                          #
# ===================================================================== #

# DRL parameters
REWARD_FUNCTION = "A"       # Defined in reward.py
ACTION_SIZE     = 3         # Not used for DQN, see DQN_ACTION_SIZE 三个动作 x方向线速度，y方向线速度,z轴旋转角速度
HIDDEN_SIZE     = 512       # Number of neurons in hidden layers

BATCH_SIZE      = 128       # Number of samples per training batch
BUFFER_SIZE     = 1000000   # Number of samples stored in replay buffer before FIFO
DISCOUNT_FACTOR = 0.99
LEARNING_RATE   = 0.003
TAU             = 0.003

OBSERVE_STEPS   = 25000     # At training start random actions are taken for N steps for better exploration
STEP_TIME       = 0.01      # Delay between steps, can be set to 0
EPSILON_DECAY   = 0.9995    # Epsilon decay per step
EPSILON_MINIMUM = 0.05

# DQN parameters
DQN_ACTION_SIZE = 5
TARGET_UPDATE_FREQUENCY = 1000

# DDPG parameters

# TD3 parameters
POLICY_NOISE            = 0.2
POLICY_NOISE_CLIP       = 0.5
POLICY_UPDATE_FREQUENCY = 2

# Stacking
STACK_DEPTH = 3             # Number of subsequent frames processed per step
FRAME_SKIP  = 4             # Number of frames skipped in between subsequent frames

# Episode outcome enumeration
UNKNOWN = 0
SUCCESS = 1
COLLISION_WALL = 2
COLLISION_OBSTACLE = 3
TIMEOUT = 4
TUMBLE = 5
RESULTS_NUM = 6