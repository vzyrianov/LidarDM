import torch
import numpy as np

def sample_to_cuda(sample_cpu):
    sample = {}
    for k in sample_cpu:
        if(torch.is_tensor(sample_cpu[k])):
            sample[k] = sample_cpu[k][:].cuda()
        else:
            sample[k] = sample_cpu[k]

    return sample

def sample_to_torch(sample_cpu):
    sample = {}
    for k in sample_cpu:
        if(isinstance(sample_cpu[k], np.ndarray)):
            sample[k] = torch.from_numpy(sample_cpu[k][:]).unsqueeze(0)
        else:
            sample[k] = sample_cpu[k]

    return sample