from util import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from util import default_train, default_initialize,\
        trainloader, testloader


net = Net()
default_initialize(net)
default_train(net, trainloader, None)



