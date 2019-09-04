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

 "main.py" - Main file for training fully-connected and convolutional networks using backpropagation (BP),
    feedback alignment (FA) [Lillicrap, Nat. Comms, 2016], direct feedback alignment (DFA) [Nokland, NIPS, 2016],
    and the proposed direct random target projection (DRTP).
    Example: use the following command to reach ~70% accuracy on the test set of CIFAR-10 using DRTP:
         python main.py --dataset CIFAR10aug --train-mode DRTP --epochs 200 --freeze-conv-layers
                        --dropout 0.05 --topology CONV_64_3_1_1_CONV_256_3_1_1_FC_2000_FC_10
                        --loss CE --output-act none --lr 5e-4
 
 Project: DRTP - Direct Random Target Projection

 Authors:  C. Frenkel and M. Lefebvre, Université catholique de Louvain (UCLouvain), 09/2019

 Cite/paper: C. Frenkel, M. Lefebvre and D. Bol, "Learning without feedback: Direct random target projection
    as a feedback-alignment algorithm with layerwise feedforward training," arXiv preprint arXiv:1909.01311, 2019.

------------------------------------------------------------------------------
"""


import argparse
import train
import setup

def main():
    parser = argparse.ArgumentParser(description='Training fully-connected and convolutional networks using backpropagation (BP), feedback alignment (FA), direct feedback alignment (DFA), and direct random target projection (DRTP)')
    # General
    parser.add_argument('--cpu', action='store_true', default=False, help='Disable CUDA and run on CPU.')
    # Dataset
    parser.add_argument('--dataset', type=str, choices = ['regression_synth', 'classification_synth', 'MNIST', 'CIFAR10', 'CIFAR10aug'], default='MNIST', help='Choice of the dataset: synthetic regression (regression_synth), synthetic classification (classification_synth), MNIST (MNIST), CIFAR-10 (CIFAR10), CIFAR-10 with data augmentation (CIFAR10aug). Synthetic datasets must have been generated previously with synth_dataset_gen.py. Default: MNIST.')
    # Training
    parser.add_argument('--train-mode', choices = ['BP','FA','DFA','DRTP','sDFA','shallow'], default='DRTP', help='Choice of the training algorithm - backpropagation (BP), feedback alignment (FA), direct feedback alignment (DFA), direct random target propagation (DRTP), error-sign-based DFA (sDFA), shallow learning with all layers freezed but the last one that is BP-trained (shallow). Default: DRTP.')
    parser.add_argument('--optimizer', choices = ['SGD', 'NAG', 'Adam', 'RMSprop'], default='NAG', help='Choice of the optimizer - stochastic gradient descent with 0.9 momentum (SGD), SGD with 0.9 momentum and Nesterov-accelerated gradients (NAG), Adam (Adam), and RMSprop (RMSprop). Default: NAG.')
    parser.add_argument('--loss', choices = ['MSE', 'BCE', 'CE'], default='BCE', help='Choice of loss function - mean squared error (MSE), binary cross entropy (BCE), cross entropy (CE, which already contains a logsoftmax activation function). Default: BCE.')
    parser.add_argument('--freeze-conv-layers', action='store_true', default=False, help='Disable training of convolutional layers and keeps the weights at their initialized values.')
    parser.add_argument('--fc-zero-init', action='store_true', default=False, help='Initializes fully-connected weights to zero instead of the default He uniform initialization.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout probability (applied only to fully-connected layers). Default: 0.')
    parser.add_argument('--trials', type=int, default=1, help='Number of training trials Default: 1.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs Default: 100.')
    parser.add_argument('--batch-size', type=int, default=100, help='Input batch size for training. Default: 100.')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='Input batch size for testing Default: 1000.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate. Default: 1e-4.')
    # Network
    parser.add_argument('--topology', type=str, default='CONV_32_5_1_2_FC_1000_FC_10', help='Choice of network topology. Format for convolutional layers: CONV_{output channels}_{kernel size}_{stride}_{padding}. Format for fully-connected layers: FC_{output units}.')
    parser.add_argument('--conv-act', type=str, choices = {'tanh', 'sigmoid', 'relu'}, default='tanh', help='Type of activation for the convolutional layers - Tanh (tanh), Sigmoid (sigmoid), ReLU (relu). Default: tanh.')
    parser.add_argument('--hidden-act', type=str, choices = {'tanh', 'sigmoid', 'relu'}, default='tanh', help='Type of activation for the fully-connected hidden layers - Tanh (tanh), Sigmoid (sigmoid), ReLU (relu). Default: tanh.')
    parser.add_argument('--output-act', type=str, choices = {'sigmoid', 'tanh', 'none'}, default='sigmoid', help='Type of activation for the network output layer - Sigmoid (sigmoid), Tanh (tanh), none (none). Default: sigmoid.')
    
    args = parser.parse_args()

    (device, train_loader, traintest_loader, test_loader) = setup.setup(args)
    train.train(args, device, train_loader, traintest_loader, test_loader)

if __name__ == '__main__':
    main()