import torch
import torch.nn as nn
import random
import numpy as np
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn.modules import Upsample
from gmrf_resnet import inference_resnet, inv_cov_fft_resnet
from gmrf_vgg import inference_vgg, inv_cov_fft_vgg
from gmrf_inception import inference_inception, inv_cov_fft_inception
import argparse
import json
import pdb




#GLOBAL PARAMETERS
CLASSIFIERS = {
    "inception_v3": (models.inception_v3, 299),
    "resnet50": (models.resnet50, 224),
    "vgg16_bn": (models.vgg16_bn, 224),
}
IMAGENET_PATH = "/home/anit/val"

def fft_basis(dataset_size):

    count = 0
    samples = torch.zeros(2000, 3, dataset_size, dataset_size)
    B = torch.zeros(3,dataset_size,dataset_size,2)
    for k in range(43):
        i = k
        j = 0
        while i>=0:
            for s in range(3):
                B[s,i,j,0] = 1.0
                val = torch.irfft(B,3,onesided=False)
                if val.norm(p=2)!=0:
                    samples[count] = val
                    count = count+1
                B[s,i,j,0] = 0
                #to get the sine components uncomment the next part
                #B[s,i,j,1] = 1.0
                #val = torch.irfft(B,3,onesided=False)
                #if val.norm(p=2)!=0:
                    #samples[count] = val
                    #count = count+1
                #B[s,i,j,1] = 0
                i=i-1
                j=j+1

    samples = samples*dataset_size*dataset_size*3
    return samples


def test(image, target, model, device, epsilon, delta, samples, U_data, Sig_data, num_sample, sig, Lam_fft_data):

    # Accuracy counter
    correct = 0
    # Send the data and label to the device
    data, target = image.float().to(device), target.to(device)
    G_diff = torch.zeros(num_sample).float()
    # Forward pass the data through the model
    output = normalized_eval(data).float()
    init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    count = 0
    for sample in samples[0:num_sample]:
        noise = (delta*sample).float().to(device) 
        output2 = normalized_eval(data-noise.view_as(data))
        output1 = normalized_eval(data+noise.view_as(data))
        loss_diff = (nn.CrossEntropyLoss(reduction="none")(output1,target)-nn.CrossEntropyLoss(reduction="none")(output2,target))
        del noise
        torch.cuda.empty_cache()
        G_diff.view(data.shape[0], -1)[:,count] = loss_diff.float()
        count = count + 1
    z_data = ((2*samples[0:num_sample]*delta).view(num_sample,-1).transpose(0,1).float() @ G_diff[:,None]).float().view_as(data[0][None])/sig
    Lam_inv_z_data = torch.irfft((1/(Lam_fft_data[None,None,:,:,:,None]*sig)) * torch.rfft(z_data, 3, onesided=False), 3, onesided=False)
    G_mu_data = Lam_inv_z_data - (U_data.view(num_sample,-1).transpose(0,1).float() @ (Sig_data @ (U_data.view(num_sample,-1) @ z_data.view(-1)))).view_as(data[0][None])
    # Call FGSM Attack
    perturbed_data = fgsm_attack(data, epsilon, G_mu_data[0])
    del Lam_inv_z_data, z_data, G_diff
    torch.cuda.empty_cache()
    # Re-classify the perturbed image
    output = normalized_eval(perturbed_data)
    # Check for success
    final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    if final_pred.item() == target.item():
        correct = 1
    return correct

def samples_gen(sig, delta, num_sample, Lam_fft_data):
    dX_fft_data = torch.rfft(2*samples[0:num_sample]*delta, 3, onesided=False).float()
    torch.cuda.empty_cache()
    U_data = torch.irfft((1/(Lam_fft_data[None,None,:,:,:,None]*sig)) * dX_fft_data, 3, onesided=False).float()
    Sig_data = (sig * torch.eye(num_sample).float() + (2*samples[0:num_sample]*delta).view(num_sample, -1).float() @ U_data.view(num_sample, -1).transpose(0,1).float()).inverse().float()
    return samples, U_data, Sig_data

def normalized_eval(x):
    x_copy = F.normalize(x.view(3,dataset_size,dataset_size), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    return model(x_copy.view(1,3,dataset_size,dataset_size))

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign().to(device)
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad.float()
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def attack():
    eps = args.eps
    num_samples = range(1,1+int(round(num_queries)))
    attacks = torch.zeros(50000)
    if args.arch == 'resnet50':
        #In case when things go out of memory these are the values to use
        #alpha, beta, gamma = torch.tensor(2631.9353, requires_grad=True),torch.tensor(-263.3363, requires_grad=True),torch.tensor(-837.1694, requires_grad=True)
        #kappa = torch.tensor(6.7809, requires_grad=True)
        #nu = torch.tensor(28.0909, requires_grad=True)
        alpha, beta, gamma, kappa, nu = inference_resnet(IMAGENET_PATH)
        with open("results_{}.txt".format(args.arch),'a') as f:
            f.write("GMRF Parameters: alpha ={} beta = {} gamma ={} kappa ={} nu={}\n".format(alpha, beta, gamma, kappa, nu))
            f.close()
        print("==============GMRF Inference Done===================")
        alpha = torch.tensor(alpha, requires_grad = True)
        beta = torch.tensor(beta, requires_grad = True)
        gamma = torch.tensor(gamma, requires_grad = True)
        kappa = torch.tensor(kappa, requires_grad = True)
        nu = torch.tensor(nu, requires_grad = True)
        Lam_fft_data = inv_cov_fft_resnet(alpha, beta, gamma, kappa, nu, 224, 224)
        upsampler = Upsample(size=(299, 299))
        sig  = 0.5
        delta = 0.0375
    if args.arch == 'vgg16_bn':
        #In case when things go out of memory these are the values to use
        #alpha, beta, gamma = torch.tensor(633.4402, requires_grad=True),torch.tensor(-24.0530, requires_grad=True),torch.tensor(-232.0468, requires_grad=True)
        #kappa = torch.tensor(-2.0052, requires_grad=True)
        alpha, beta, gamma, kappa = inference_vgg(IMAGENET_PATH)
        with open("results_{}.txt".format(args.arch),'a') as f:
            f.write("GMRF Parameters: alpha ={} beta = {} gamma ={} kappa ={}\n".format(alpha, beta, gamma, kappa))
            f.close()
        print("==============GMRF Inference Done===================")
        alpha = torch.tensor(alpha, requires_grad = True)
        beta = torch.tensor(beta, requires_grad = True)
        gamma = torch.tensor(gamma, requires_grad = True)
        kappa = torch.tensor(kappa, requires_grad = True)
        Lam_fft_data = inv_cov_fft_vgg(alpha, beta, gamma, kappa, 224, 224)
        upsampler = Upsample(size=(299, 299))
        sig  = 1.0
        delta = 0.04
    if args.arch == 'inception_v3':
        #In case when things go out of memory these are the values to use
        #alpha = torch.tensor(10717.1230, requires_grad=True)
        #beta = torch.tensor(-3682.8098, requires_grad=True)
        #gamma = torch.tensor(-748.8978, requires_grad=True)
        #kappa = torch.tensor(684.9440, requires_grad=True)
        #nu = torch.tensor(430.169, requires_grad=True)
        #kappa1 = torch.tensor(-518.2526, requires_grad=True)
        alpha, beta, gamma, kappa, nu, kappa1 = inference_inception(IMAGENET_PATH)
        with open("results_{}.txt".format(args.arch),'a') as f:
            f.write("GMRF Parameters: alpha ={} beta = {} gamma ={} kappa ={} nu ={} kappa1 ={}\n".format(alpha, beta, gamma, kappa, nu, kappa1))
            f.close()
        print("==============GMRF Inference Done===================")
        alpha = torch.tensor(alpha, requires_grad = True)
        beta = torch.tensor(beta, requires_grad = True)
        gamma = torch.tensor(gamma, requires_grad = True)
        kappa = torch.tensor(kappa, requires_grad = True)
        nu = torch.tensor(nu, requires_grad = True)
        kappa1 = torch.tensor(kappa1, requires_grad = True)
        Lam_fft_data = inv_cov_fft_inception(alpha, beta, gamma, kappa, nu, kappa1, 299, 299)
        upsampler = Upsample(size=(299, 299))
        sig  = 0.1
        delta = 0.045
    with torch.no_grad():
        torch.cuda.empty_cache()
        with open("results_{}.txt".format(args.arch),'a') as f:
            f.write("FFT_Sweep_cos: {}: Delta: {} sig: {}\n".format(args.arch,delta,sig))
            f.close()
        for num_sample in num_samples:
            samples, U_data, Sig_data = samples_gen(sig, delta, num_sample, Lam_fft_data)
            #torch.cuda.empty_cache()
            true_count = 0
            count = 0
            total_correct = 0
            for i in set_attack:
                count = count+1
                images, targets = dataset[i]
                images = torch.from_numpy(np.array(images))
                targets = torch.tensor(targets)
                output = normalized_eval(images.cuda())
                final_pred = output.max(1, keepdim=True)[1]
                if final_pred.item() != targets.item() or attacks[i]==1:
                    continue
                true_count = true_count+1
                if count%100 == 0 and count>0:
                    print(count, "iteration")
                    print(true_count, "Images attacked")
                    print(float(total_correct/true_count))
                    with open("results_{}.txt".format(args.arch),'a') as f:
                        f.write("Total Correct: {} \tTotal Images: {} \tTest Accuracy = {}\n".format(total_correct, true_count, float(total_correct/true_count)))
                        f.close()
                cor = test(images.unsqueeze(0).cuda(), targets.unsqueeze(0).cuda(), model, device, eps, delta, samples, U_data, Sig_data, num_sample, sig, Lam_fft_data)
                if cor is 0:
                    attacks[i] = 1
                total_correct += cor
                torch.cuda.empty_cache()
            print(total_correct)
            acc = float(total_correct/true_count)
            with open("results_{}.txt".format(args.arch),'a') as f:
                f.write("Number Of Samples: {} \tEpsilon: {}\tDelta: {}\tTest Accuracy = {} / {} = {}\n".format(num_sample, eps, delta, total_correct, true_count, acc))
                f.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='vgg16_bn')
    parser.add_argument('--eps', type=float, default=0.05)
    parser.add_argument('--iter', type=int, default=1000)
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    num_queries = args.iter/2
    set_attack = random.sample(range(0,50000), 1000)

    with torch.no_grad():
        model_type = CLASSIFIERS[args.arch][0]
        model = model_type(pretrained=True).float().cuda()
        model = DataParallel(model)
        model.eval()
    dataset_size = CLASSIFIERS[args.arch][1]
    samples = fft_basis(dataset_size)
    if args.arch == 'inception_v3':
        dataset = ImageFolder(IMAGENET_PATH,
                transforms.Compose([
                    transforms.Resize(dataset_size),
                    transforms.CenterCrop(dataset_size),
                    transforms.ToTensor()
                ]))
    else:
        dataset = ImageFolder(IMAGENET_PATH,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(dataset_size),
                    transforms.ToTensor()
                ]))
    attack()




