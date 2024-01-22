import matplotlib.pyplot as plt
import numpy
import os
import glob
import sys
import pandas as pd
import socket
from datetime import datetime
import argparse

TOP_EPISODES = 4
PLOT_INTERVAL = 100
# Episode outcome enumeration
UNKNOWN = 0
SUCCESS = 1
COLLISION_WALL = 2
COLLISION_OBSTACLE = 3
TIMEOUT = 4
TUMBLE = 5
RESULTS_NUM = 6


def main(args):


    base_path = os.getenv('DRLNAV_BASE_PATH') + f"/src/{args.agent_name}/model/{args.file_path}/"
    logfile = glob.glob(base_path + '/*_speed_*.txt')
    # logfile = glob.glob(base_path + '/_test_*.txt')
    # if len(logfile) != 1:
    #     print(f"ERROR: found less or more than 1 logfile for: {base_path}")
    for i in range(len(logfile)):
        plt.figure(figsize=(10,10))
        df = pd.read_csv(logfile[i])
        robot_pose_x_column = df[' robot_pose_x']
        robot_pose_y_column = df[' robot_pose_y']
        goal_pose_x_column = df[' goal_pose_x']
        goal_pose_y_column = df[' goal_pose_y']
        # steps_column   = df['step']
        robot_pose_x = robot_pose_x_column.tolist()
        robot_pose_y = robot_pose_y_column.tolist()
        goal_pose_x = goal_pose_x_column.tolist()
        goal_pose_y = goal_pose_y_column.tolist()
        # steps = steps_column.tolist()


        # plt.plot(xaxis, average_rewards, label="avg_rewards")
        plt.plot(0.0, 0.0, "o", lw=3, c='black')
        plt.plot(robot_pose_x, robot_pose_y, label="trajectory", lw=1, c='black')
        plt.plot(goal_pose_x[0], goal_pose_y[0], "o",  label="goal", lw=3, c='red')

        # plt.xlabel('Episode', fontsize=24, fontweight='bold')
        # plt.ylabel('Avg.Rate', fontsize=24, fontweight='bold')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.xlim(-6,6)
        plt.ylim(-6,6)
        plt.grid(True, linestyle='-')
        plt.savefig(logfile[i][:-4] + "_trajectory.png", format="png", bbox_inches="tight")
        # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_name",           default="cyberdog_drl", type=str)
    parser.add_argument("--file_path",            default="dayday-pc/ddpg_888_stage_4", type=str)
    args = parser.parse_args()
    print(args)
    main(args)
