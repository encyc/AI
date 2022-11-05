import torch

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

