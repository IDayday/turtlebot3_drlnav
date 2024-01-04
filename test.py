import random

# goals = [[1,3],[1,-3],[1,6],[1,-6],[4,4],[4,-4],[4,6],[4,-6],[5,0],[5,3],[5,-3],[7,0],[7,3],[7,-3],[-3,0],
#             [-3.5,3],[-3.5,-3],[-3.5,6],[-3.5,-6],[-6.5,0]]
# goal = random.choice(goals)
# goal_x, goal_y = goal[0], goal[1]
# print(goal_x, goal_y)


import torch


# a = torch.randint(0,5,(10,1))
# b = torch.randint(0,5,(10,1))
# c = torch.cat([a,b],dim=-1)
# print(a)
# print(b)
# print(c)

a = torch.randn(10,5)
print(a)