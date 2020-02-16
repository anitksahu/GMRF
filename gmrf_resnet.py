import torch
import torch.nn as nn
import torch.nn.functional as FF
from torchvision.datasets import ImageFolder
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from gmrf_utils import gradient_hessian

def inv_cov_fft_resnet(alpha, beta, gamma, kappa, nu, H, W):
    A = torch.zeros(3,H,W).type_as(alpha)
    A[0,0,0] = alpha
    A[0,0,1] = A[0,0,-1] = A[0,1,0] = A[0,-1,0] = beta
    A[0,1,1] = A[0,-1,1] = A[0,-1,-1] = A[0,1,-1] = kappa
    A[0,0,2] = A[0,0,-2] = A[0,-2,0] = A[0,2,0] = nu
    A[0,1,2] = A[0,-1,2] = A[0,2,1] = A[0,2,-1] = nu
    A[0,-1,-2] = A[0,1,-2] = A[0,-2,-1] = A[0,-2,1] = nu
    A[-1,0,0] = A[1,0,0] = gamma
    return torch.rfft(A,3,onesided=False)[:,:,:,0]

def log_det_fft(alpha, beta, gamma, kappa, nu, H,W):
    result = inv_cov_fft_resnet(alpha, beta, gamma, kappa, nu, H, W)
    return result.log().sum()

def fft_Lambda_y(alpha, beta, gamma, kappa, nu, H, W, Y, power=1):
    Y_fft = torch.rfft(Y[0], 3, onesided=False)
    A_fft = inv_cov_fft_resnet(alpha, beta, gamma, kappa, nu, H, W)[:,:,:,None]
    return torch.irfft(A_fft**power * Y_fft, 3, onesided=False) 

def conv_obj(alpha, beta, gamma, kappa, nu, Y, H, W):
    m,c,H,W = Y.shape
    Y_ = torch.cat([Y[:,-1:,:,:], Y, Y[:,:1,:,:]], 1)
    Y_ = torch.cat([Y_[:,:,-2:,:], Y_, Y_[:,:,:2,:]], 2)
    Y_ = torch.cat([Y_[:,:,:,-2:], Y_, Y_[:,:,:,:2]], 3)
    K = torch.zeros(3,5,5).type_as(alpha)
    K[1,2,2] = alpha
    K[1,1,2] = K[1,2,3] = K[1,2,1] = K[1,3,2] = beta
    K[1,1,1] = K[1,3,3] = K[1,3,1] = K[1,1,3] = kappa
    K[1,0,1] = K[1,0,2] = K[1,0,3] = K[1,1,-1] = nu
    K[1,2,-1] = K[1,3,-1] = K[1,-1,3] = K[1,-1,2] = nu
    K[1,-1,1] = K[1,3,0] = K[1,2,0] = K[1,1,0] = nu
    K[0,2,2] = K[-1,2,2] = gamma
    K = torch.unsqueeze(torch.unsqueeze(K,0),0)
    conv_result = FF.conv3d(Y_.unsqueeze_(1).cpu(), K, stride =1, padding=0)
    return (Y.cpu() * conv_result.view_as(Y)/m).sum()

def f_objective(alpha, beta, gamma, kappa, nu, Y, H, W):
    return  conv_obj(alpha, beta, gamma, kappa, nu, Y, H, W) - log_det_fft(alpha, beta, gamma, kappa, nu, H, W)

def normalized_eval(x):
    x_copy = F.normalize(x.view(3,224,224), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    return model_to_fool(x_copy.view(1,3,224,224))

def newton(X, tol=1e-4, max_iter=50):
    m,c,H,W = X.shape
    alpha = torch.tensor(10.0, requires_grad=True)
    beta = torch.tensor(-0.1, requires_grad=True)
    gamma = torch.tensor(-0.05, requires_grad=True)
    kappa = torch.tensor(0.01, requires_grad=True)
    nu = torch.tensor(0.001, requires_grad=True)
    it = 0
    g = torch.tensor(1.)
    while (g.norm().item() > tol and it < max_iter):
        f = f_objective(alpha, beta, gamma, kappa, nu, X, H, W)
        g, H_ = gradient_hessian(f, [alpha, beta, gamma, kappa,nu])
        print(f.item(), g.norm().item())
        dx = -H_.inverse() @ g
        t = 1.0
        while True:
            alpha0 = (alpha + t*dx[0]).detach()
            beta0 = (beta + t*dx[1]).detach()
            gamma0 = (gamma + t*dx[2]).detach()
            kappa0 = (kappa + t*dx[3]).detach()
            nu0 = (nu + t*dx[4]).detach()
            f0 = f_objective(alpha0, beta0, gamma0, kappa0, nu0, X, H, W)
            if f0 < f+1000:
                break
            t = t*0.5
            
        alpha = (alpha + t*dx[0]).detach().requires_grad_()
        beta = (beta + t*dx[1]).detach().requires_grad_()
        gamma = (gamma + t*dx[2]).detach().requires_grad_()
        kappa = (kappa + t*dx[3]).detach().requires_grad_()
        nu = (nu + t*dx[4]).detach().requires_grad_()
        it += 1
        #print(alpha.item(), beta.item(), gamma.item(),kappa.item(), nu.item())
    return alpha, beta, gamma, kappa, nu

def inference_resnet(IMAGENET_PATH):
    dataset = ImageFolder(IMAGENET_PATH,
                    transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor()
                    ]))
    dataset_loader = DataLoader(dataset, batch_size=5)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        model_type = models.resnet50
        model = model_type(pretrained=True).float().cuda()
        model = DataParallel(model)
        model.eval()
    epsilon = 1
    num = 5
    for X,y in dataset_loader:
        break
    X.detach_()
    G = torch.zeros_like(X)
    for i in range(num):
        pert = torch.randn(1, X.shape[1], X.shape[2], X.shape[3])
        dX = epsilon * pert
        G = G.to(device) + ((nn.CrossEntropyLoss(reduction="none")(model((X + dX).to(device)), y.to(device)) - nn.CrossEntropyLoss(reduction="none")(model((X - dX).to(device)), y.to(device))).view(-1,1,1,1)/(2*epsilon))*pert.to(device)
    G = G/num
    alpha, beta, gamma, kappa, nu = newton(G)
    return alpha.item(), beta.item(), gamma.item(), kappa.item(), nu.item()