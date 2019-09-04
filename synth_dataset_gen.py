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

 "synth_dataset_gen.py" - Generation of synthetic regression and classification datasets.
                          Launch with command 'python synth_dataset_gen.py'
 
 Project: DRTP - Direct Random Target Projection

 Authors:  C. Frenkel and M. Lefebvre, Université catholique de Louvain (UCLouvain), 09/2019

 Cite/paper: C. Frenkel, M. Lefebvre and D. Bol, "Learning without feedback: Direct random target projection
    as a feedback-alignment algorithm with layerwise feedforward training," arXiv preprint arXiv:1909.01311, 2019.

------------------------------------------------------------------------------
"""


import torch
import numpy as np
from sklearn.datasets import make_classification
import os
import math

def gen_regression(n_train, n_test, n_classes, input_size_sqrt):

    input_dim = pow(input_size_sqrt,2)
    for dataset in ["train","test"]:
        n = n_train if (dataset=="train") else n_test
        norms = math.pi*(torch.rand(n)*2-1)
        X = torch.normal(mean=norms.unsqueeze(1).repeat(1,input_dim),std=1)
        t = torch.zeros(n, n_classes)
        for i in range(n_classes):
            t[:,i] = torch.cos(torch.mean(X,dim=1)+math.pi*(i-4.5)/9)
        if dataset=="train":
            X_train, t_train = X, t
        else:
            X_test, t_test = X, t
            
    if not os.path.exists('./DATASETS/regression'):
        os.makedirs('./DATASETS/regression')
    torch.save(((X_train,t_train), input_size_sqrt, 1, n_classes), "./DATASETS/regression/train.pt")
    torch.save(((X_test ,t_test ), input_size_sqrt, 1, n_classes), "./DATASETS/regression/test.pt")


def gen_classification(n_train, n_test, n_classes, n_pix_sqrt, n_inf, n_clusters_per_class=5, class_sep=4.5, random_state=0):

    n_samples = n_train + n_test
    input_dim = pow(n_pix_sqrt,2)
    X, y = make_classification(n_samples=n_samples, n_features=input_dim, n_informative=n_inf, n_redundant=0, n_repeated=0, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, scale=np.ones(shape=(input_dim,)), class_sep=class_sep, shuffle=True, random_state=random_state)
    X = torch.Tensor(X)
    y = torch.Tensor(y).long()
    X_train, y_train = X[0:n_train, :], y[0:n_train]
    X_test, y_test = X[n_train:, :], y[n_train:]
    
    if not os.path.exists('./DATASETS/classification'):
        os.makedirs('./DATASETS/classification')
    torch.save(((X_train,y_train), n_pix_sqrt, 1, n_classes), "./DATASETS/classification/train.pt")
    torch.save(((X_test ,y_test ), n_pix_sqrt, 1, n_classes), "./DATASETS/classification/test.pt")


if __name__ == '__main__':
    gen_regression(n_train=5000, n_test=1000, n_classes=10, input_size_sqrt=16)
    gen_classification(n_train=25000, n_test=5000, n_classes=10, n_pix_sqrt=16, n_inf=128)