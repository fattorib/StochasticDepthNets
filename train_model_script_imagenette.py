import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from StochasticDepthResNets.src.ResidualBlocks import ResNet110, ResNet50
from StochasticDepthResNets.utils.Train_Model import Train_Model
from torchvision.datasets import CIFAR10, ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import time


def main():
    cudnn.benchmark = True

    # Data augmentation as outlined in paper
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomCrop((128, 128), padding=4, fill=0,
                              padding_mode='constant'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, fillcolor=0, shear=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[-0.2278, -0.1259, -0.0213],
                             std=[1.0976, 1.0963, 1.1121])

    ])

    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[-0.2278, -0.1259, -0.0213],
                             std=[1.0976, 1.0963, 1.1121])
    ])

    train_data = ImageFolder(root='imagenette2-320/train',
                             transform=transform_train)

    test_data = ImageFolder(root='imagenette2-320/val',
                            transform=transform_test)

    # Getting a validation set
    train_data, val_data = torch.utils.data.random_split(train_data, [
        8500, 969])

    # Training ResNet50
    model = ResNet50(pretrained=False)

    meta_config = {'project name': "StochasticDepthResNets", 'batch size': 64,
                   'initial lr': 0.1, 'Optimizer': 'SGD',
                   'weight decay': 1e-4, 'lr annealing': True, 'accumulation': True,
                   'accumulation steps': 2, 'max epochs': 200}

    model_class = Train_Model(
        model, train_data, test_data, val_data, meta_config)
    t0 = time.time()
    model_class.train()
    t1 = time.time()
    print(f'Total training time:{(t1-t0)/60.:2f} minutes.')
    model_class.eval(train=False)
    model_class.eval(train=True)


if __name__ == '__main__':
    main()
