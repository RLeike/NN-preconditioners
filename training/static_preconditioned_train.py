import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def infer_prior(shape, verbose=False, relu=None):
    """
    calculates initialization that causes all activations to be normal
    (almost like Xavier initialization, but considering biases)
    """
    if relu is None:
        relu = len(shape)==2
    if verbose:
        print(shape)
    if len(shape) > 1:
        if verbose:
            print("assuming all but first dimension is input")
        result = np.sqrt(1/np.prod(shape[1:])*.96) # letting 4% for bias
        if relu:
            result /= np.sqrt(.34) # account for relu
        return result
    elif len(shape) == 1:
        if verbose:
            print("assuming this is a bias")
        return .2 # causes a variance of .04

def infer_gradient_magnitude(shape, verbose=False, relu=None):
    """
    calculates which factor an independent normal gradient gets when
    being passed through a layer
    """
    if relu is None:
        relu = len(shape)==2
    std = infer_prior(shape)
    if verbose:
        print(shape)
    if len(shape) > 1:
        if verbose:
            print("assuming all but last dimension is input")
        size = np.prod(shape[0:-1]) # this is slightly wrong for padding=same
        result = np.sqrt(size)*std
        if relu:
            result *= np.sqrt(.34) # account for relu
        return result
    else:
        if verbose:
            print("assuming this is a bias")
        return 1. # biases don't modify the gradient


def spc_initialize(net):
    global prior
    prior = []
    for para in net.parameters():
        prior += [infer_prior(para.shape, verbose=False)]
        para.data = (torch.randn(*para.shape)*prior[-1]).data
        

def spc_train(net, epochs, trainloader, validationloader, verbose=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    prec = []
    for p in net.parameters():
        prec += [1/infer_gradient_magnitude(p.shape)]
    prec += [1.]
    for i in range(len(prec)-2,-1,-1):
        prec[i] *= prec[i+1]
    prec = prec[1:]
    if verbose:
        print("prior standard deviations:", prior)
        print("preconditioner weights:", prec)

    losses = []
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            batch_size = inputs.shape[0]
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            for j,p in enumerate(net.parameters()):
                p.grad.data.mul_(prec[j]*prior[j])
            optimizer.step()

            # print
            running_loss += loss.item()
            losses += [loss.item()]
            if i % 2000 == 1999:    # print every 2000 mini-batches
                if verbose:
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss/2000))
                running_loss = 0.0

    if verbose:
        print('Finished Training')
    return losses

