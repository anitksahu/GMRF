# GMRF Covariance Modelling for Black Box Adversarial attacks

## MNIST

L\infty attacks: Arguments allowed epsilon (eps) and query budget (iter).

`python3 attack_mnist.py --eps 0.05 --iter 200`

## ImageNet

L\infty attacks: Arguments allowed epsilon (eps), query budget (iter) and architecture (vgg16_bn, resnet50, inception_v3)

`python3 attack_imagenet.py --eps 0.05 --iter 1000 --arch vgg16_bn`

