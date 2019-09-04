# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------

Copyright (C) 2019 Université catholique de Louvain (UCLouvain), Belgium.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

------------------------------------------------------------------------------

 "setup.py" - Setup configuration and dataset loading.
 
 Project: DRTP - Direct Random Target Projection

 Authors:  C. Frenkel and M. Lefebvre, Université catholique de Louvain (UCLouvain), 09/2019

 Cite/paper: C. Frenkel, M. Lefebvre and D. Bol, "Learning without feedback: Direct random target projection
    as a feedback-alignment algorithm with layerwise feedforward training," arXiv preprint arXiv:1909.01311, 2019.

------------------------------------------------------------------------------
"""


import torch
import torchvision
from torchvision import transforms,datasets
import numpy as np
import os
import sys
import subprocess


class SynthDataset(torch.utils.data.Dataset):

    def __init__(self, select, type):
        self.dataset, self.input_size, self.input_channels, self.label_features = torch.load( './DATASETS/'+select+'/'+type+'.pt')

    def __len__(self):
        return len(self.dataset[1])

    def __getitem__(self, index):
        return self.dataset[0][index], self.dataset[1][index]

def setup(args):
    args.cuda = not args.cpu and torch.cuda.is_available()
    if args.cuda:
        print("=== The available CUDA GPU will be used for computations.")
        memory_load = get_gpu_memory_usage()
        cuda_device = np.argmin(memory_load).item()
        torch.cuda.set_device(cuda_device)
        device = torch.cuda.current_device()
    else:
        device = torch.device('cpu')
    
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    if args.dataset == "regression_synth":
        print("=== Loading the synthetic regression dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_regression_synth(args, kwargs)
    elif args.dataset == "classification_synth":
        print("=== Loading the synthetic classification dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_classification_synth(args, kwargs)
    elif args.dataset == "MNIST":
        print("=== Loading the MNIST dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_mnist(args, kwargs)
    elif args.dataset == "CIFAR10":
        print("=== Loading the CIFAR-10 dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_cifar10(args, kwargs)
    elif args.dataset == "CIFAR10aug":
        print("=== Loading and augmenting the CIFAR-10 dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_cifar10_augmented(args, kwargs)
    else:
        print("=== ERROR - Unsupported dataset ===")
        sys.exit(1)
    args.regression = (args.dataset == "regression_synth")
    
    return (device, train_loader, traintest_loader, test_loader)

def get_gpu_memory_usage():
    if sys.platform == "win32":
        curr_dir = os.getcwd()
        nvsmi_dir = r"C:\Program Files\NVIDIA Corporation\NVSMI"
        os.chdir(nvsmi_dir)
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'])
        os.chdir(curr_dir)
    else:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'])
    gpu_memory = [int(x) for x in result.decode('utf-8').strip().split('\n')]
    return gpu_memory

def load_dataset_regression_synth(args, kwargs):

    trainset = SynthDataset("regression","train")
    testset  = SynthDataset("regression", "test")

    train_loader     = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,      shuffle=True , **kwargs)
    traintest_loader = torch.utils.data.DataLoader(trainset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader      = torch.utils.data.DataLoader(testset , batch_size=args.test_batch_size, shuffle=False, **kwargs)
    
    args.input_size     = trainset.input_size
    args.input_channels = trainset.input_channels
    args.label_features = trainset.label_features
    
    return (train_loader, traintest_loader, test_loader)

def load_dataset_classification_synth(args, kwargs):

    trainset = SynthDataset("classification","train")
    testset  = SynthDataset("classification", "test")

    train_loader     = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,      shuffle=True , **kwargs)
    traintest_loader = torch.utils.data.DataLoader(trainset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader      = torch.utils.data.DataLoader(testset , batch_size=args.test_batch_size, shuffle=False, **kwargs)

    args.input_size     = trainset.input_size
    args.input_channels = trainset.input_channels
    args.label_features = trainset.label_features
    
    return (train_loader, traintest_loader, test_loader)

def load_dataset_mnist(args, kwargs):
    train_loader     = torch.utils.data.DataLoader(datasets.MNIST('./DATASETS', train=True,  download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))])), batch_size=args.batch_size,      shuffle=True , **kwargs)
    traintest_loader = torch.utils.data.DataLoader(datasets.MNIST('./DATASETS', train=True,  download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))])), batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader      = torch.utils.data.DataLoader(datasets.MNIST('./DATASETS', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))])), batch_size=args.test_batch_size, shuffle=False, **kwargs)
    
    args.input_size     = 28
    args.input_channels = 1
    args.label_features = 10

    return (train_loader, traintest_loader, test_loader)

def load_dataset_cifar10(args, kwargs):
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]], std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_cifar10 = transforms.Compose([transforms.ToTensor(),normalize,])
    
    train_loader     = torch.utils.data.DataLoader(datasets.CIFAR10('./DATASETS', train=True,  download=True, transform=transform_cifar10), batch_size=args.batch_size,      shuffle=True , **kwargs)
    traintest_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./DATASETS', train=True,  download=True, transform=transform_cifar10), batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader      = torch.utils.data.DataLoader(datasets.CIFAR10('./DATASETS', train=False, download=True, transform=transform_cifar10), batch_size=args.test_batch_size, shuffle=False, **kwargs)
    
    args.input_size     = 32
    args.input_channels = 3
    args.label_features = 10

    return (train_loader, traintest_loader, test_loader)

def load_dataset_cifar10_augmented(args, kwargs):
    #Source: https://zhenye-na.github.io/2018/09/28/pytorch-cnn-cifar10.html
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                             std=[x/255.0 for x in [63.0, 62.1, 66.7]]),
    ])
    
    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]], std=[x/255.0 for x in [63.0, 62.1, 66.7]]),])
    
    trainset = torchvision.datasets.CIFAR10('./DATASETS', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    
    traintestset = torchvision.datasets.CIFAR10('./DATASETS', train=True, download=True, transform=transform_test)
    traintest_loader = torch.utils.data.DataLoader(traintestset, batch_size=args.test_batch_size, shuffle=False)
    
    testset = torchvision.datasets.CIFAR10('./DATASETS', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
    
    args.input_size     = 32
    args.input_channels = 3
    args.label_features = 10

    return (train_loader, traintest_loader, test_loader)