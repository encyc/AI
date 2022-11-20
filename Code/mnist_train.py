# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 16:35:46 2022

@author: Administrator
"""
# In[]:
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision

from utils import plot_image, plot_curve, one_hot



# In[]:
# step1. load dataset

# 记录总批次数量
batch_size = 512

# 读取训练集
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data',train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms .Normalize(
                                        (0.1307,),(0.3081,))
                                    ])),
    batch_size=batch_size, shuffle=True)

# 读取测试集
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/',train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms .Normalize(
                                        (0.1307,),(0.3081,))
                                    ])),
    batch_size=batch_size, shuffle=True)

# 设置x, y
x, y = next(iter(train_loader))
print(x.shape, y.shape )

plot_image(x,y,'image sample')


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        # xw+b
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        # x: [b, 1, 28, 28]
        # h1 = relu(xw1+b1)
        x = F.relu(self.fc1(x))
        
        # h2 = relu(h1w2+b2)
        x = F.relu(self.fc2(x))
        
        # h3 = h2w3+b3
        #x = F.sigmoid(self.fc3(x))
        x = self.fc3(x)
        return x

net = Net()
# [w1, b1, w2, b2, w3, b3]
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

# 记录loss
train_loss = []
for epoch in range(10):
    
    for batch_idx, (x,y) in enumerate((train_loader)):
        
        # x: [b, 1, 28, 28], y: [512]
        # [b,784(feature)]
        x = x.view(x.size(0), 28*28)
        # => [b, 10]
        out = net(x)
        # [b,10]
        y_onehot = one_hot(y)
        # loss = mse(out, y_onehot)
        loss = F.mse_loss(out, y_onehot)
        
        # 清零梯度
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # w' = W - lr*grad 更新梯度
        optimizer.step()
        
        # 记录每次的loss
        train_loss.append(loss.item())
        
        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())
# we get optimal [w1, b1, w2, b2, w3, b3]

plot_curve(train_loss)


# 利用accuracy判断模型的效果

total_correct = 0
for x,y in test_loader:
    x = x.view(x.size(0), 28*28)
    out = net(x)
    # out: [b,10] => pred: [b]
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct
    
total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc:', acc)

# 直观查看是否预测正确

x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28*28))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')