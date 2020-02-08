from util import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import util


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

net = util.Net()
util.spq_initialize(net)
util.spq_train(net, trainloader, None)


net = util.Net()
util.default_initialize(net)
net = net.to(device)
util.default_train(net, trainloader, None)



