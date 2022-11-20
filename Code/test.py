import torch
'''
W_h = torch.randn(20,20,requires_grad=True)
W_h
W_x = torch.randn(20,10,requires_grad=True)
W_x
x = torch.randn(1,10)
x
prev_h = torch.randn(1,20)
prev_h
h2h = torch.mm(W_h,prev_h.t())
h2h

'''
import numpy as np
import pandas as pd

    
from torch.utils.data import Dataset

def compute_error_for_line_given_points(b,w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        totalError += (y - (w * x + b)) ** 2
        return totalError / float(len(points))
    
    

def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += -(2/N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2/N) * x * (y - ((w_current * x) + b_current))
        new_b = b_current - (learningRate * b_gradient)
        new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]


def gradient_descent_runner(points, starting_b, starting_m,learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return [b, m]


a = torch.randn(2,3)
a

a.type()
type(a)

isinstance(a,torch.FloatTensor)

data = a.cuda()

print(torch.__version__)
print(torch.cuda.is_available())
