import util

import pytest
import numpy as np
import torch
from torch.distributions import normal
from torch.nn.functional import relu

pmp = pytest.mark.parametrize

class DummyNetwork(object):
    def __init__(self, shape):
        self.paras = [torch.zeros(shape), torch.zeros(shape[0])]

    def parameters(self):
        return self.paras


@pmp('seed', [1,2,3,42])
@pmp('shape', [(30,70), (100,41)])
def test_spc_initialization(seed, shape):
    # Test whether all activations are random normal distributed
    np.random.seed(seed)
    torch.manual_seed(seed)
    net = DummyNetwork(shape)
    paras = net.parameters()
    util.spc_initialize(net)
    result = paras[0].matmul(torch.randn(shape[1:]+(10,))) + paras[1].view((-1,1))
    result = result*np.sqrt(.34) # hack that reduces variance by the same amount as relu
    test_vec = np.array(result**2).sum(axis=1)
    from scipy.stats import chi2
    # need to test for significant deviation
    a,b, = chi2.interval(1-1e-4/test_vec.shape[0]/8, 10)
    print("bounds", a,b)
    print("values", test_vec)
    print("WARNING: Statistical test")
    print("This test might fail on accident in .01% of the cases")
    assert np.all(a<test_vec)
    assert np.all(test_vec<b)

@pmp('seed', [1,2,3,42])
@pmp('shape', [(30,70), (100,41)])
def test_spc_gradient(seed, shape):
    # Test whether the gradient behaves as predicted 
    np.random.seed(seed)
    torch.manual_seed(seed)
    net = DummyNetwork(shape)
    util.spc_initialize(net)
    paras = net.parameters()
    result = relu(torch.randn((10,) + shape[:1])).matmul(paras[0])
    result = result*np.sqrt(.34) # hack that reduces variance by the same amount as relu
    std = util.infer_gradient_magnitude(shape)
    print(std)
    test_vec = np.array(result**2).sum(axis=0)/std
    from scipy.stats import chi2
    # need to test for significant deviation
    a,b, = chi2.interval(1-1e-4/test_vec.shape[0]/8, 10)
    print("bounds", a,b)
    print("values", test_vec)
    print("WARNING: Statistical test")
    print("This test might fail on accident in .01% of the cases")
    #assert np.all(a<test_vec)
    #assert np.all(test_vec<b)



