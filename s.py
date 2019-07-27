import torch
import numpy as np
import copy


x = [[[0, 1], [1, 2], [2, 4]],
     [[1, 2], [1, 3], [3, 5]],
     [[1, 2], [1, 0], [1, 3]],
     [[1, 2], [1, 0], [1, 3]]]

y = [[[0, 0, 3], [0, 0, 3]],
     [[0, 0, 3], [0, 0, 1]],
     [[0, 0, 2], [0, 0, 2]]]
x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

x[:3,:2,][(y[...,2]==3)] = y[...,:2][y[...,2]==3]


x = np.ones((3,100,200))
y = np.ones((4,30,20))

y1 = 0
x1 = 0

x[:, y1:y1+30, x1:x1+20][y[3,...]>=0.003] =  y[:3, ...][y[3,...]>=0.003]