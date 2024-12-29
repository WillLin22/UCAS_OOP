import torch.nn as nn
import torch
import time

class Net(nn.Module):
    def __init__(self, inputsize, outputsize):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(inputsize, 10)
        self.fc2 = nn.Linear(10, outputsize)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
net = Net(2, 1)
ten = torch.tensor([1.0, 2.0])
t = time.time()
out = net(ten)
t = time.time() - t
print(out)
print(t)