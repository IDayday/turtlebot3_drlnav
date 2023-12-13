import copy
import os
import sys
import time
import numpy as np

from turtlebot3_drl.common.storagemanager import StorageManager
from turtlebot3_drl.common.replaybuffer import ReplayBuffer




load_session = "/home/dayday/project/turtlebot3_drlnav/src/cyberdog_drl/model/dayday-pc/ddpg_88_stage_4"
sm = StorageManager("ddpg", load_session, 400, "cuda", 4)
replay_buffer = ReplayBuffer(1024)
buffer_1 = sm.load_replay_buffer(1024, os.path.join(load_session, 'stage'+str(sm.stage)+'_latest_buffer.pkl'))
buffer_2 = sm.load_replay_buffer(1024, os.path.join(load_session, 'stage'+str(sm.stage)+'_latest_buffer.pkl'))
buffer_1.extend(buffer_2)
print("buffer",buffer_1)