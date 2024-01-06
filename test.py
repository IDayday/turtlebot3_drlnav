import random

# goals = [[1,3],[1,-3],[1,6],[1,-6],[4,4],[4,-4],[4,6],[4,-6],[5,0],[5,3],[5,-3],[7,0],[7,3],[7,-3],[-3,0],
#             [-3.5,3],[-3.5,-3],[-3.5,6],[-3.5,-6],[-6.5,0]]
# goal = random.choice(goals)
# goal_x, goal_y = goal[0], goal[1]
# print(goal_x, goal_y)


import torch
from collections import deque
import numpy as np
import itertools
# a = torch.randint(0,5,(10,1))
# b = torch.randint(0,5,(10,1))
# c = torch.cat([a,b],dim=-1)
# print(a)
# print(b)
# print(c)

# a = torch.randn(10,5)
# print(a)


buffer = deque(maxlen=100)
a = np.arange(0,10)
b = np.arange(11,21)
for i in range(10):
    buffer.append((a[i],b[i]))
print(buffer)
length = len(buffer)
c = list(itertools.islice(buffer, length-3, length))
d = random.sample(buffer,3)
d.extend(c)
x = np.float32([array[0] for array in d])
print(d)
print(x)