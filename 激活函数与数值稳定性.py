import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt

# sigmoid
x = torch.arange(-20,20,0.1,requires_grad=True)
y = torch.sigmoid(x)
def fax(x,y): 
    y.sum().backward()
    _ , axes = plt.subplots(1,1,figsize=(6,6))
    with torch.no_grad():
        axes.plot(x,y)
        axes.plot(x,x.grad)
    x.grad.zero_()
fax(x,y)
y=torch.relu(x)
fax(x,y)
y = torch.tanh(x)
fax(x,y)
plt.show()



