import matplotlib.pyplot as plt
import numpy as np
import math
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

    plt.figure(figsize=(10,10))
    avg_episode_smooth = []
    avg_smooth = []
    base_path = os.getenv('DRLNAV_BASE_PATH') + f"/src/{args.agent_name}/model/{args.file_path}/"
    logfile = glob.glob(base_path + '/*_speed_*.txt')
    logfile.sort()
    # logfile = glob.glob(base_path + '/_test_*.txt')
    # if len(logfile) != 1:
    #     print(f"ERROR: found less or more than 1 logfile for: {base_path}")
    for i in range(len(logfile)):

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

        length = len(robot_pose_x_column)
        for j in range(1,length-1):
            smooth = (math.pow(robot_pose_x[j-1]+robot_pose_x[j+1]-2*robot_pose_x[j],2) + math.pow(robot_pose_y[j-1]+robot_pose_y[j+1]-2*robot_pose_y[j],2))
            avg_episode_smooth.append(smooth)
        avg_episode_smooth_value = np.mean(avg_episode_smooth)
        avg_smooth.append(avg_episode_smooth_value)
    avg_smooth_l1 = 10000*np.mean(avg_smooth[:40])
    avg_smooth_l2 = 10000*np.mean(avg_smooth[40:80])
    avg_smooth_l3 = 10000*np.mean(avg_smooth[80:])

    x=[1,2,3]  
    y=[avg_smooth_l1,avg_smooth_l2,avg_smooth_l3]  
    color=['blue','green','yellow','orchid','deepskyblue']
    x_label=['task_level_1','task_level_2','task_level_3']
    plt.xticks(x, x_label)
    plt.bar(x, y,color=color)
    plt.title("Smooth of trajectory")

    # plt.plot(xaxis, average_rewards, label="avg_rewards")
    # plt.plot(0.0, 0.0, "o", lw=3, c='black')
    # plt.plot(robot_pose_x, robot_pose_y, label="trajectory", lw=1, c='black')
    # plt.plot(goal_pose_x[0], goal_pose_y[0], "o",  label="goal", lw=3, c='red')

    # plt.xlabel('Episode', fontsize=24, fontweight='bold')
    # plt.ylabel('Avg.Rate', fontsize=24, fontweight='bold')
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.legend(fontsize=20)
    # plt.xlim(-6,6)
    plt.ylim(0,5)
    plt.grid(True, linestyle='--')
    plt.savefig(logfile[i][:-4] + "_trajectory_smooth.png", format="png", bbox_inches="tight")
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_name",           default="cyberdog_drl", type=str)
    parser.add_argument("--file_path",            default="dayday-pc/ddpg_888_stage_4", type=str)
    args = parser.parse_args()
    print(args)
    main(args)
