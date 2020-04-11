import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import argparse

import training


parser = argparse.ArgumentParser(description='Demo for preconditioning neural network trainig')
parser.add_argument('-e', '--epochs', type=int, nargs=1,
                           help='number of epochs to train', default=2)
parser.add_argument('-q', '--quiet', action='store_true',
            help='suppresses command line output')
parser.add_argument('-s', '--smoothing', type=int, nargs=1,
            help='6-sigma region for Gaussian kernel with which the loss is smoothed', default=100)

args = vars(parser.parse_args())
verbose = not args['quiet']
smoothing_len = args['smoothing']
epochs = args['epochs']



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

net = training.Net()
training.spc_initialize(net)
prec_loss = training.spc_train(net, epochs, training.trainloader, None, verbose)

np.save('prec_loss.npy', prec_loss)

net = training.Net()
training.default_initialize(net)
net = net.to(device)
default_loss = training.default_train(net, epochs, training.trainloader, None, verbose)

np.save('default_loss.npy', default_loss)

# smooth loss with Gaussian kernel to reduce noise
kernel = np.exp(-0.5*np.linspace(-3, 3, smoothing_len)**2) 
kernel /= np.sum(kernel)
prec_graph = np.convolve(prec_loss, kernel, 'valid')
default_graph = np.convolve(default_loss, kernel, 'valid')
epochs = np.linspace(0, len(default_loss)/12500, len(prec_graph))
plt.plot(epochs, prec_graph, label='preconditioned')
plt.plot(epochs, default_graph, label='default')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('loss.png', dpi=200)
if verbose:
    plt.show()
else:
    plt.close()




