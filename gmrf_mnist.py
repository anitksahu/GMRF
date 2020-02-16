import torch
import torch.nn as nn
import torch.nn.functional as F
from gmrf_utils import gradient_hessian
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import models, transforms
from mnist_model import Net

def inv_cov_fft(alpha, beta, gamma, H, W):
    A = torch.zeros(H,W).type_as(alpha)
    A[0,0] = alpha
    A[0,1] = A[0,-1] = A[1,0] = A[-1,0] = beta
    A[-1,-1] = A[1,1] = A[-1,1] = A[1,-1] = gamma
    return torch.rfft(A,2,onesided=False)[:,:,0]

def log_det_fft(alpha, beta, gamma, H,W):
    return inv_cov_fft(alpha, beta, gamma, H, W).log().sum()

# multiplication by Lambda with optional power (includeing inverses)
def fft_Lambda_y(alpha, beta, gamma, H, W, Y, power=1):
    #print(Y.size())
    Y_fft = torch.rfft(Y[0,0], 2, onesided=False)
    A_fft = inv_cov_fft(alpha, beta, gamma, H, W)[:,:,None]
    return torch.irfft(A_fft**power * Y_fft, 2, onesided=False) 


def conv_obj(alpha, beta, gamma,Y, H, W):
    m,c,H,W = Y.shape
    Y_ = torch.cat([Y[:,:,-1:,:], Y, Y[:,:,:1,:]], 2)
    Y_ = torch.cat([Y_[:,:,:,-1:], Y_, Y_[:,:,:,:1]], 3)
    K = torch.zeros(3,3).type_as(alpha)
    K[1,1] = alpha
    K[0,1] = K[1,2] = K[1,0] = K[2,1] = beta
    K[0,0] = K[2,2] = K[2,0] = K[0,2] = gamma
    K = torch.unsqueeze(torch.unsqueeze(K,0),0)
    conv_result = F.conv2d(Y_.cpu(), K, stride=1, padding=0)
    return (Y.cpu() * conv_result.view_as(Y)/m).sum()

def f_objective(alpha, beta, gamma, Y, H, W):
    return  conv_obj(alpha, beta, gamma, Y, H, W) - log_det_fft(alpha, beta, gamma, H, W)

def newton(X, tol=1e-4, max_iter=50):
    m,c,H,W = X.shape
    alpha = torch.tensor(1.0, requires_grad=True)
    beta = torch.tensor(0., requires_grad=True)
    gamma = torch.tensor(0.1, requires_grad=True)
    it = 0
    g = torch.tensor(1.)
    while (g.norm().item() > 1e-4 and it < max_iter):
        f = f_objective(alpha, beta, gamma, X, H, W)
        g, H_ = gradient_hessian(f, [alpha, beta, gamma])
        dx = -H_.inverse() @ g
        
        t = 1.0
        while True:
            alpha0 = (alpha + t*dx[0]).detach()
            beta0 = (beta + t*dx[1]).detach()
            gamma0 = (gamma + t*dx[2]).detach()
            f0 = f_objective(alpha0, beta0, gamma0, X, H, W)
            if f0 < f+1000:
                break
            t = t*0.5
            
        alpha = (alpha + t*dx[0]).detach().requires_grad_()
        beta = (beta + t*dx[1]).detach().requires_grad_()
        gamma = (gamma + t*dx[2]).detach().requires_grad_()
        it += 1
    return alpha, beta, gamma

def inference_mnist():
    test_loader = DataLoader(MNIST(".", train=False, download=True, transform=transforms.ToTensor()), 
                                batch_size=100, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    model.load_state_dict(torch.load('lenet_mnist_model.pth'))
    epsilon = 0.1
    num = 100
    for X,y in test_loader:
        break
    X.detach_()
    G = torch.zeros_like(X)
    for j in range(num):
        pert = torch.randn(1, X.shape[1], X.shape[2], X.shape[3])
        dX = epsilon * pert
        f1 = F.nll_loss(model((X + dX).to(device)), y.to(device))
        f2 = F.nll_loss(model((X - dX).to(device)), y.to(device))
        G = G.to(device) + ((f1 - f2).view(-1,1,1,1)/(2*epsilon))*pert.to(device)
    G = G/num
    alpha, beta, gamma = newton(G)
    return alpha.item(), beta.item(), gamma.item()



