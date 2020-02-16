import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn.modules import Upsample
from torchvision.datasets import MNIST
from gmrf_mnist import inference_mnist, inv_cov_fft
from mnist_model import Net
import argparse
import json
import pdb

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign().double()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad.to(device)
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def samples_gen(sig, delta, num_sample, Lam_fft_data):
    dX_fft_data = torch.rfft(2*samples[0:num_sample]*delta, 2, onesided=False).double()
    U_data = torch.irfft((1/Lam_fft_data[None,None,:,:,None]) * dX_fft_data, 2, onesided=False).double()
    Sig_data = (sig * torch.eye(num_sample).float() + (2*samples[0:num_sample]*delta).view(num_sample, -1).float() @ U_data.view(num_sample, -1).transpose(0,1).float()).inverse().float()
    return samples, U_data, Sig_data

def test(data, target, model, device, epsilon, delta, samples, U_data, Sig_data, num_sample, sig, Lam_fft_data):

    # Accuracy counter
    correct = 0
    data, target = data.double().to(device), target.to(device)
    G_diff = torch.zeros(num_sample).double()
    output = model(data).double()
    init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    loss_diff = 0
    count = 0
    size_data = 784
    dX_data = 2*samples[0:num_sample]*delta
    for sample in samples[0:num_sample]:
        noise = delta*sample
        noise = noise.double().to(device)
        output2 = model(data-noise.view_as(data))
        output1 = model(data+noise.view_as(data))
        loss_diff = (F.nll_loss(output1,target)-F.nll_loss(output2,target))
        G_diff.view(data.shape[0], -1)[:,count] = loss_diff
        count = count + 1
    z_data = (((2*samples[0:num_sample]*delta).view(num_sample,-1).transpose(0,1).float() @ G_diff[:,None].float()).view_as(data[0][None])/sig).float()
    z_fft_data = torch.rfft(z_data, 2, onesided=False)
    Lam_inv_z_data = torch.irfft((1/Lam_fft_data[None,None,:,:,None]).float() * z_fft_data.float(), 2, onesided=False).float()
    G_mu_data = Lam_inv_z_data - (U_data.view(num_sample,-1).float().transpose(0,1) @ (Sig_data.float() @ (U_data.view(num_sample,-1).float() @ z_data.view(-1).float()))).view_as(data[0][None]).float()
        # Call FGSM Attack
    perturbed_data = fgsm_attack(data, epsilon, G_mu_data)
        # Re-classify the perturbed image
    output = model(perturbed_data)
        # Check for success
    final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    if final_pred.item() == target.item():
        correct = 1
    return correct



def fft_basis():

    count = 0
    samples = torch.zeros(200, 1, 28, 28)
    B = torch.zeros(28,28,2)
    count = 0
    for k in range(10):
        i = k
        j = 0
        while i>=0:
            B[i,j,0] = 1.0
            samples[count] = torch.irfft(B,2,onesided=False)
            B[i,j,0] = 0
            count = count+1
            B[i,j,1] = 1.0
            samples[count] = torch.irfft(B,2,onesided=False)
            B[i,j,1] = 0
            count = count+1
            i=i-1
            j=j+1

    samples = samples*784
    return samples

def attack():
    sig = 1e-3
    delta = 0.15
    eps = args.eps
    num_samples = range(1,1+int(round(num_queries)))
    alpha, beta, gamma = inference_mnist()
    print("==============GMRF Inference Done===================")
    alpha = torch.tensor(alpha, requires_grad = True)
    beta = torch.tensor(beta, requires_grad = True)
    gamma = torch.tensor(gamma, requires_grad = True)
    Lam_fft_data = inv_cov_fft(alpha, beta, gamma, 28, 28)*1e-3
    with torch.no_grad():
        attacks = torch.zeros(10000)
        torch.cuda.empty_cache()
        with open("results_lenet_mnist.txt",'a') as f:
            f.write("Sweep: Lenet: Delta: {} sig: {}\n".format(delta,sig))
            f.close()
        for num_sample in num_samples:
            samples, U_data, Sig_data = samples_gen(sig, delta, num_sample, Lam_fft_data.double())
            torch.cuda.empty_cache()
            true_count = 0
            total_correct = 0
            for i,(images,targets) in enumerate(dataset):
                output = model(images.double().to(device))
                final_pred = output.max(1, keepdim=True)[1]
                if final_pred.item() != targets.item() or attacks[i]==1:
                    continue
                true_count = true_count+1
                if i%100 == 0 and i>0:
                    print(i, "iteration")
                    print(true_count, "Images attacked")
                    print(float(total_correct/true_count))
                    with open("results_lenet_mnist.txt",'a') as f:
                        f.write("Total Correct: {} \tTotal Images: {} \tTest Accuracy = {}\n".format(total_correct, true_count, float(total_correct/true_count)))
                        f.close()
                #if i >= test_len:
                    #break
                cor = test(images.cuda(), targets.cuda(), model, device, eps, delta, samples, U_data, Sig_data, num_sample, sig, Lam_fft_data)
                if cor is 0:
                    attacks[i] = 1
                total_correct += cor
                torch.cuda.empty_cache()
            print(total_correct)
            if true_count > 0:
                acc = float(total_correct/true_count)
                with open("results_lenet_mnist.txt",'a') as f:
                    #f.write("Sig: {}\n".format(sig))
                    f.write("Number Of Samples: {} \tEpsilon: {}\tDelta: {}\tTest Accuracy = {} / {} = {}\n".format(num_sample, eps, delta, total_correct, true_count, acc))
                    f.close()









if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', type=float, default=0.05)
    parser.add_argument('--iter', type=int, default=200)
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    model.load_state_dict(torch.load('lenet_mnist_model.pth'))
    model = model.double().to(device)
    num_queries = args.iter/2
    samples = fft_basis()
    dataset = DataLoader(MNIST(".", train=False, download=True, transform=transforms.ToTensor()), 
                                batch_size=1, shuffle=True)
    attack()