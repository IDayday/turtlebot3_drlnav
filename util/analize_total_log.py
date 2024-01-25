import matplotlib.pyplot as plt
import numpy as np
import math
import os
import glob
import sys
import pandas as pd
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


def statistics(data, condition):
    task_1 = np.array(data[:40])
    task_2 = np.array(data[40:80])
    task_3 = np.array(data[80:])
    task_t = np.array(data)

    con_1 = np.where(np.array(condition[:40])==1)
    con_2 = np.where(np.array(condition[40:80])==1)
    con_3 = np.where(np.array(condition[80:])==1)
    con_t = np.where(np.array(condition)==1)

    d_1 = np.mean(task_1[con_1] if task_1[con_1].size != 0 else [0])
    d_2 = np.mean(task_2[con_2] if task_2[con_2].size != 0 else [0])
    d_3 = np.mean(task_3[con_3] if task_3[con_3].size != 0 else [0])
    d_t = np.mean(task_t[con_t] if task_t[con_t].size != 0 else [0])



    return [d_1,d_2,d_3,d_t]

def data_plot(data, title, save_pth):
    t_1 = np.array(data)[:,0]
    t_2 = np.array(data)[:,1]
    t_3 = np.array(data)[:,2]
    t_t = np.array(data)[:,3]

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
    # plt.show()  

def main(args):

    avg_success_rate = []
    avg_reward_sum = []
    avg_episode_duration = []
    avg_distance = []
    base_path = os.getenv('DRLNAV_BASE_PATH') + f"/src/{args.agent_name}/model/{args.file_path}/"
    logfile = glob.glob(base_path + '/*_total_log.txt')
    logfile.sort()


    for i in range(len(logfile)):

        df = pd.read_csv(logfile[i])
        outcome = df[' outcome']
        count = df[' success/cw/co/timeout/tumble']
        reward_sum = df[' reward_sum']
        episode_duration = df[' episode_duration']
        distance = df[' distance']
        # steps_column   = df['step']
        outcome = outcome.tolist()
        count = count.tolist()
        reward_sum = reward_sum.tolist()
        episode_duration = episode_duration.tolist()
        distance = distance.tolist()
        # steps = steps_column.tolist()

        # calculate task level success rate
        s_1 = int(count[39].split('/')[0])
        s_2 = int(count[79].split('/')[0])
        s_3 = int(count[-1].split('/')[0])
        avg_success_rate.append([s_1/40, (s_2-s_1)/40, (s_3-s_2)/40, s_3/120])

        # calculate task level reward_sum
        # r_1 = np.mean(reward_sum[:40])
        # r_2 = np.mean(reward_sum[40:80])
        # r_3 = np.mean(reward_sum[80:])
        # r_t = np.mean(reward_sum)
        # avg_reward_sum.append([r_1, r_2, r_3, r_t])

        eval_reward_sum = statistics(reward_sum, outcome)
        eval_reward_sum[-1] = eval_reward_sum[0]*0.2 + eval_reward_sum[1]*0.3 + eval_reward_sum[2]*0.5
        eval_episode_duration = statistics(episode_duration, outcome)
        eval_distance = statistics(distance, outcome)
        avg_reward_sum.append(eval_reward_sum)
        avg_episode_duration.append(eval_episode_duration)
        avg_distance.append(eval_distance)


    data_plot(avg_success_rate,'avg_success',base_path)
    data_plot(avg_reward_sum,'avg_reward_sum',base_path)
    data_plot(avg_episode_duration,'avg_episode_duration',base_path)
    data_plot(avg_distance,'avg_distance',base_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_name",           default="cyberdog_drl", type=str)
    parser.add_argument("--file_path",            default="dayday-pc/total_log", type=str)
    args = parser.parse_args()
    print(args)
    main(args)
