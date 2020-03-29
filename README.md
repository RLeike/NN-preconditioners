# NN-preconditioners

Exploring preconditioners from statistical considerations - a toy project

## Prerequisites

python3 3.8

pytorch

scipy

matplotlib

## Usage

This is a demonstration repository. Run the code with

`python3 main.py`.

Run the tests with

`pytest` (needs pytest).

## Background

It is known that network initialization plays a crucial role in how good a network can be trained, especially for deep neural networks.
Let us assume a normal distributed activation $a$ is the input to a $M \times N$ shaped fully connected layer with normal distributed weights of standard deviation $\sigma$.
The output of the layer will be normal distributed with standard deviation $\sqrt{M}\sigma$.
It was therefore recommended to initialize a network layer with $\sigma = \sqrt{M^{-1}}$ e.g. with Normal samples of standard deviation $\sqrt{M^{-1}}$.
This way standard normal distributed inputs to the layer lead to standard normal distributed outputs.
However the adjoint problem exists for the gradient; a normal distributed gradient that gets backpropagated will be modified by a factor $\sigma \sqrt{N}$.

For deep networks these factors accumulate multiplicatively and cause the exploding/vanishing gradient problem.
The goal of this toy project is to cancel this effect by introducing a diagonal preconditioner which is multiplied with the gradient.

## Implementation

To make the effect visible, one needs to consider a fairly deep network. 
Here the network of the pytorch demo was chosen.
There are two training/initialization routines.
The code in `util/default_train.py` mimics the procedure of the official pytorch demo, initializing with the pytorch default initializer and training with a momentum based approach.
The code in `util/static_preconditioned_train.py` initializes all nework layers such that the output is a priori standard normal distributed for normal distributed input.
It computes a diagonal preconditioner that causes the gradient to be standard normal distributed in all layers given a standard normal distributed gradient of the loss.

## Observations

While the preconditioned gradient leeds to faster training initially, the training then slows and it gets overtaken by the standard procedure at about epoch 2.
This is possibly due to covariate shifts, all considerations that were made were assuming normal distributed weights of a fixed variance, but as the network trains this variance changes.
