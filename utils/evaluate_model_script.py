import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from StochasticDepthResNets.src.ResidualBlocks import StochasticDepthResNet, ResNet110
from Train_Model import Train_Model
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import time


if __name__ == '__main__':
    # Data augmentation as outlined in paper
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_data = CIFAR10(root='/CIFAR', train=False,
                        download=False, transform=transform_test)

    model = ResNet110(pretrained=True).cuda()

    model_class = Train_Model(model, None, test_data, None)
    t0 = time.time()
    model_class.eval(train=False)
    t1 = time.time()
    print(f'Total inference time:{(t1-t0)/60.:2f} minutes.')
