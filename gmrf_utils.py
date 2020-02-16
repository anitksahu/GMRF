import torch
import torch.nn as nn
import torch.nn.functional as F

def gradient_hessian(J, params):
    d = torch.autograd.grad(J, params, create_graph=True)
    d2 = [torch.autograd.grad(f, params, retain_graph=(i < len(d)-1)) for i,f in enumerate(d)]
    return torch.tensor(d), torch.tensor(d2)