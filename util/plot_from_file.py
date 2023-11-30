from cProfile import label
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

    plt.figure(figsize=(16,10))
    j = 0
    base_path = os.getenv('DRLNAV_BASE_PATH') + f"/src/{args.agent_name}/model/{args.file_path}/"
    logfile = glob.glob(base_path + '/_train_l*.txt')
    if len(logfile) != 1:
        print(f"ERROR: found less or more than 1 logfile for: {base_path}")
    df = pd.read_csv(logfile[0])
    rewards_column = df[' reward']
    success_column = df[' success']
    steps_column   = df[' steps']
    rewards = rewards_column.tolist()
    success = success_column.tolist()
    steps = steps_column.tolist()
    average_rewards = []
    average_success = []
    average_timeout = []
    average_collision = []
    sum_rewards = 0
    sum_success = 0
    sum_timeout = 0
    sum_collision = 0
    episode_range = len(df.index)
    xaxis = numpy.array(range(0, episode_range - PLOT_INTERVAL, PLOT_INTERVAL))
    for i in range (episode_range):
        if i % PLOT_INTERVAL == 0 and i > 0:
            average_rewards.append(sum_rewards / PLOT_INTERVAL)
            sum_rewards = 0
            average_success.append(sum_success / PLOT_INTERVAL)
            sum_success = 0
            average_timeout.append(sum_timeout / PLOT_INTERVAL)
            sum_timeout = 0
            average_collision.append(sum_collision / PLOT_INTERVAL)
            sum_collision = 0
        sum_rewards += rewards[i]
        if success[i] == SUCCESS:
            sum_success += 1
        elif success[i] == TIMEOUT:
            sum_timeout += 1
        elif success[i] == COLLISION_OBSTACLE or success[i] == COLLISION_WALL:
            sum_collision += 1
    # plt.plot(xaxis, average_rewards, label="avg_rewards")
    plt.plot(xaxis, average_success, label="avg_success_rate", lw=3, c='orange')
    plt.plot(xaxis, average_timeout, label="avg_timeout_rate", lw=3, c='royalblue')
    plt.plot(xaxis, average_collision, label="avg_collision_rate", lw=3, c='tomato')

    plt.xlabel('Episode', fontsize=24, fontweight='bold')
    plt.ylabel('Avg.Rate', fontsize=24, fontweight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.ylim(0,1.05)
    plt.grid(True, linestyle='--')
    plt.savefig(base_path + "outcomes.png", format="png", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_name",           default="cyberdog_drl", type=str)
    parser.add_argument("--file_path",            default="dayday-pc/ddpg_44_stage_4", type=str)
    args = parser.parse_args()
    print(args)
    main(args)
