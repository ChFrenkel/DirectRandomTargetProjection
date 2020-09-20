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

 "train.py" - Initializing the network, optimizer and loss for training and testing.
 
 Project: DRTP - Direct Random Target Projection

 Authors:  C. Frenkel and M. Lefebvre, Université catholique de Louvain (UCLouvain), 09/2019

 Cite/paper: C. Frenkel, M. Lefebvre and D. Bol, "Learning without feedback: Direct random target projection
    as a feedback-alignment algorithm with layerwise feedforward training," arXiv preprint arXiv:1909.01311, 2019.

------------------------------------------------------------------------------
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import models
from tqdm import tqdm

def train(args, device, train_loader, traintest_loader, test_loader):
    torch.manual_seed(42)
    
    for trial in range(1,args.trials+1):
        # Network topology
        model = models.NetworkBuilder(args.topology, input_size=args.input_size, input_channels=args.input_channels, label_features=args.label_features, train_batch_size=args.batch_size, train_mode=args.train_mode, dropout=args.dropout, conv_act=args.conv_act, hidden_act=args.hidden_act, output_act=args.output_act, fc_zero_init=args.fc_zero_init, loss=args.loss, device=device)

        if args.cuda:
            model.cuda()
        
        if (args.trials > 1):
            print('\nIn trial {} of {}'.format(trial,args.trials))
        if (trial == 1):
            print("=== Model ===" )
            print(model)
        
        # Optimizer
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=False)
        elif args.optimizer == 'NAG':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        else:
            raise NameError("=== ERROR: optimizer " + str(args.optimizer) + " not supported")
        
        # Loss function
        if args.loss == 'MSE':
            loss = (F.mse_loss, (lambda l : l))
        elif args.loss == 'BCE':
            loss = (F.binary_cross_entropy, (lambda l : l))
        elif args.loss == 'CE':
            loss = (F.cross_entropy, (lambda l : torch.max(l, 1)[1]))
        else:
            raise NameError("=== ERROR: loss " + str(args.loss) + " not supported")
        
        print("\n\n=== Starting model training with %d epochs:\n" % (args.epochs,))        
        for epoch in range(1, args.epochs + 1):
            # Training
            train_epoch(args, model, device, train_loader, optimizer, loss)
            
            # Compute accuracy on training and testing set
            print("\nSummary of epoch %d:" % (epoch))
            test_epoch(args, model, device, traintest_loader, loss, 'Train')
            test_epoch(args, model, device, test_loader, loss, 'Test')


def train_epoch(args, model, device, train_loader, optimizer, loss):
    model.train()
    
    if args.freeze_conv_layers:
        for i in range(model.conv_to_fc):
            for param in model.layers[i].conv.parameters():
                param.requires_grad = False
    
    for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
        data, label = data.to(device), label.to(device)
        if args.regression:
            targets = label
        else:
            targets = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1, label.unsqueeze(1), 1.0)
        optimizer.zero_grad()
        output = model(data, targets)
        loss_val = loss[0](output, loss[1](targets))
        loss_val.backward()
        optimizer.step()


def test_epoch(args, model, device, test_loader, loss, phase):
    model.eval()

    test_loss, correct = 0, 0
    len_dataset = len(test_loader.dataset)
    
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            if args.regression:
                targets = label
            else:
                targets = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1, label.unsqueeze(1), 1.0)
            output = model(data, None)
            test_loss += loss[0](output, loss[1](targets), reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            if not args.regression:
                correct += pred.eq(label.view_as(pred)).sum().item()
    
    loss = test_loss / len_dataset
    if not args.regression:
        acc = 100. * correct / len_dataset
        print("\t[%5sing set] Loss: %6f, Accuracy: %6.2f%%" % (phase, loss, acc))
    else:
        print("\t[%5sing set] Loss: %6f" % (phase, loss))