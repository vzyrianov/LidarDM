
# Implementation from https://github.com/ElementAI/causal_discovery_toolbox/blob/master/cdt/utils/torch.py

import math
import torch as th
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.batchnorm import _BatchNorm

def _sample_logistic(shape, out=None):

    U = out.resize_(shape).uniform_() if out is not None else th.rand(shape)
    #U2 = out.resize_(shape).uniform_() if out is not None else th.rand(shape)

    return th.log(U) - th.log(1-U)


def _sigmoid_sample(logits, tau=1):
    """
    Implementation of Bernouilli reparametrization based on Maddison et al. 2017
    """
    dims = logits.dim()
    logistic_noise = _sample_logistic(logits.size(), out=logits.data.new())
    y = logits + logistic_noise
    return th.sigmoid(y / tau)


def gumbel_sigmoid(logits, ones_tensor, zeros_tensor, tau=1, threshold=0.5, hard=False):
    shape = logits.size()
    y_soft = _sigmoid_sample(logits, tau=tau)
    if hard:
        y_hard = th.where(y_soft > threshold, ones_tensor, zeros_tensor)
        y = y_hard.data - y_soft.data + y_soft
    else:
        y = y_soft
    return y

class GumbelSigmoidAct(nn.Module):
    def __init__(self, tau=1, threshold=0.5):
        super(GumbelSigmoidAct, self).__init__()
        self.tau = tau
        self.threshold = threshold

    def forward(self, logits):
        return gumbel_sigmoid(logits, 1, 0, tau=self.tau, threshold=self.threshold, hard=True)

if __name__=='__main__':
    x = gumbel_sigmoid(th.tensor([0.5, 0.9, 0.1]), 1, 0, hard=True)
    print('done')