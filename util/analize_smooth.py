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

def data_plot(data, title, save_pth):

    plt.figure(figsize=(10,10))
    x=[1,2,3,4]  
    x_label=['task_level_1','task_level_2','task_level_3','total']


    plt.bar(x, data[0],  width=0.3, alpha=0.5, label='model_2700')
    plt.bar([i+0.3 for i in x], data[1],  width=0.3, alpha=0.5, label='model_3200')
    plt.bar([i+0.3*2 for i in x], data[2],  width=0.3, alpha=0.5, label='model_4000')
    plt.title(title, fontsize=20)
    plt.xticks([i+0.3 for i in x], x_label, size=15)
    plt.yticks(size=20)
    plt.legend(fontsize=20)
    # plt.xlim(-6,6)
    plt.ylim(0,max(max(data))*1.25)
    plt.grid(True, linestyle='--')
    plt.savefig(save_pth + f"/_{title}.png", format="png", bbox_inches="tight")


def main(args):

    plt.figure(figsize=(10,10))

    model_2700_path = os.getenv('DRLNAV_BASE_PATH') + f"/src/{args.agent_name}/model/{args.file_path_model_2700}/"
    model_3200_path = os.getenv('DRLNAV_BASE_PATH') + f"/src/{args.agent_name}/model/{args.file_path_model_3200}/"
    model_4000_path = os.getenv('DRLNAV_BASE_PATH') + f"/src/{args.agent_name}/model/{args.file_path_model_4000}/"
    logfile_2700 = glob.glob(model_2700_path + '/*_speed_*.txt')
    success_2700 = glob.glob(model_2700_path + '/*_total_log.txt')
    logfile_3200 = glob.glob(model_3200_path + '/*_speed_*.txt')
    success_3200 = glob.glob(model_3200_path + '/*_total_log.txt')
    logfile_4000 = glob.glob(model_4000_path + '/*_speed_*.txt')
    success_4000 = glob.glob(model_4000_path + '/*_total_log.txt')
    logfiles = [logfile_2700, logfile_3200, logfile_4000]
    success = [success_2700, success_3200, success_4000]

    smooth_list = []

    for i in range(len(logfiles)):
        avg_smooth = []
        logfiles[i].sort()
        for j in range(len(logfiles[i])):
            avg_episode_smooth = []
            df = pd.read_csv(logfiles[i][j])
            dfs = pd.read_csv(success[i][0])
            # print(logfiles[i][j])
            robot_pose_x_column = df[' robot_pose_x']
            robot_pose_y_column = df[' robot_pose_y']
            goal_pose_x_column = df[' goal_pose_x']
            goal_pose_y_column = df[' goal_pose_y']
            success_ = dfs[' outcome']
            # steps_column   = df['step']
            robot_pose_x = robot_pose_x_column.tolist()
            robot_pose_y = robot_pose_y_column.tolist()
            goal_pose_x = goal_pose_x_column.tolist()
            goal_pose_y = goal_pose_y_column.tolist()
            success_ = success_.tolist()
            # steps = steps_column.tolist()

            # only success
            if success_[j] == 1:
                length = len(robot_pose_x_column)
                for k in range(1,length-1):
                    # FF1
                    smooth = (math.pow(robot_pose_x[k-1]+robot_pose_x[k+1]-2*robot_pose_x[k],2) + math.pow(robot_pose_y[k-1]+robot_pose_y[k+1]-2*robot_pose_y[k],2))
                    # FF2
                    # numerator = (robot_pose_x[k]-robot_pose_x[k-1])*(robot_pose_x[k+1]-robot_pose_x[k]) + (robot_pose_y[k]-robot_pose_y[k-1])*(robot_pose_y[k+1]-robot_pose_y[k])
                    # denominator = (math.sqrt(math.pow(robot_pose_x[k]-robot_pose_x[k-1],2)+math.pow(robot_pose_y[k]-robot_pose_y[k-1],2))+math.sqrt(math.pow(robot_pose_x[k+1]-robot_pose_x[k],2)+math.pow(robot_pose_y[k+1]-robot_pose_y[k],2)))
                    # smooth = numerator/denominator if denominator!=0 else numerator

                    avg_episode_smooth.append(smooth)
                # avg_episode_smooth_value = np.mean(avg_episode_smooth)
                avg_episode_smooth_value = np.sum(avg_episode_smooth)
                avg_smooth.append(avg_episode_smooth_value)
            else:
                avg_smooth.append(0.)
        # only success
        t1 = avg_smooth[:40]
        t1_s = [x for x in t1 if x>0.]
        t2 = avg_smooth[40:80]
        t2_s = [x for x in t2 if x>0.]
        t3 = avg_smooth[80:]
        t3_s = [x for x in t3 if x>0.]
        tt = avg_smooth
        tt_s = [x for x in tt if x>0.]
        avg_smooth_l1 = 10000*np.mean(t1_s if len(t1_s)>0 else [0.05])
        avg_smooth_l2 = 10000*np.mean(t2_s if len(t2_s)>0 else [0.05])
        avg_smooth_l3 = 10000*np.mean(t3_s if len(t3_s)>0 else [0.05])
        avg_smooth_t  = 10000*np.mean(tt_s if len(tt_s)>0 else [0.05])

        # total
        # avg_smooth_l1 = 10000*np.mean(avg_smooth[:40])
        # avg_smooth_l2 = 10000*np.mean(avg_smooth[40:80])
        # avg_smooth_l3 = 10000*np.mean(avg_smooth[80:])
        # avg_smooth_t  = 10000*np.mean(avg_smooth)



        smooth_list.append([avg_smooth_l1,avg_smooth_l2,avg_smooth_l3,avg_smooth_t])

    save_path = os.getenv('DRLNAV_BASE_PATH') + f"/src/{args.agent_name}/model/dayday-pc/total_log"
    data_plot(smooth_list,"Smooth_of_trajectory_success",save_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_name",           default="cyberdog_drl", type=str)
    parser.add_argument("--file_path_model_2700",            default="dayday-pc/ddpg_model_2700", type=str)
    parser.add_argument("--file_path_model_3200",            default="dayday-pc/ddpg_model_3200", type=str)
    parser.add_argument("--file_path_model_4000",            default="dayday-pc/ddpg_model_4000", type=str)
    args = parser.parse_args()
    print(args)
    main(args)
