from util import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import util


verbose = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

net = util.Net()
util.spc_initialize(net)
prec_loss = util.spc_train(net, trainloader, None, verbose)

np.save('prec_loss.npy', prec_loss)

net = util.Net()
util.default_initialize(net)
net = net.to(device)
default_loss = util.default_train(net, trainloader, None, verbose)

np.save('default_loss.npy', default_loss)

# smooth loss with Gaussian kernel to reduce noise
kernel = np.exp(-0.5*np.linspace(-3,3,64)**2) 
kernel /= np.sum(kernel)
prec_graph = np.convolve(prec_loss, kernel, 'valid')
default_graph = np.convolve(default_loss, kernel, 'valid')
t = np.arange(len(prec_graph))
plt.plot(t, prec_graph, label='preconditioned')
plt.plot(t, default_graph, label='default')
plt.savefig('loss.png')
if verbose:
    plt.show()
else:
    plt.close()




