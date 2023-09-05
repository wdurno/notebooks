## packages 

import tempfile 
import pickle
from tqdm import tqdm 
import matplotlib.pyplot as plt
import os 
import shutil 
import zipfile
import math 
import numpy as np 

import torch
import torch.nn as nn 
from torchvision import datasets
from torch.nn.functional import one_hot 
from torchvision.transforms.functional import pil_to_tensor 
import random 

from az_blob_util import upload_to_blob_store, download_from_blob_store 

MNIST_DIM = 28 
LEARNING_RATE = 1e-3 
EMBEDDING_DIM = 20 ## this parameter must be tuned in experiments. it's very impactful 

## data utils 

def download_mnist_to_blob_storage(): 
    with tempfile.TemporaryDirectory() as tmpdir: 
        ## download and zip data 
        data_path = os.path.join(tmpdir, 'mnist_data') 
        print(f'downloading mnist to local path: {data_path}') 
        mnist_train = datasets.MNIST(data_path, train=True, download=True) 
        mnist_test = datasets.MNIST(data_path, train=False) 
        shutil.make_archive(data_path, 'zip', data_path) ## output path is data_path + '.zip'  
        zip_path = data_path + '.zip' 
        ## upload to blob storage 
        filename='mnist_data.zip'
        sas_key = os.environ['STORAGE_KEY']
        output_container_name = 'data'
        with open(zip_path, 'rb') as f: 
            upload_to_blob_store(f.read(), filename, sas_key, output_container_name) 
            pass 
        pass 
    pass 

def get_datasets(): 
    ## download from blob store 
    filename = 'mnist_data.zip' 
    sas_key = os.environ['STORAGE_KEY'] 
    container_name = 'data' 
    x = download_from_blob_store(filename, sas_key, container_name) 
    with tempfile.TemporaryDirectory() as tmpdir: 
        ## unpack 
        data_path = os.path.join(tmpdir, 'mnist_data') 
        zip_path = data_path + '.zip' 
        with open(zip_path, 'wb') as f: 
            f.write(x) 
            pass 
        mnist_dir = os.path.join(data_path, 'mnist') 
        with zipfile.ZipFile(zip_path, 'r') as zip_file: 
            zip_file.extractall(mnist_dir) 
        ## sample 
        data_pack = {} 
        mnist_train = datasets.MNIST(mnist_dir, train=True, download=False) 
        mnist_test = datasets.MNIST(mnist_dir, train=False, download=False) 
        pass 
    return mnist_train, mnist_test  

def sample(n, sub_sample=False, dataset=None): 
    dat = dataset 
    idx_list = random.choices(range(0, len(dat)), k=n) 
    if sub_sample: 
        ## subsamples the dataset, but does not convert to tensors 
        out = [] 
        for idx in idx_list: 
            image, label = dat[idx] 
            out.append((image, label)) 
            pass 
        return out  
    x_list = [] 
    y_list = [] 
    for idx in idx_list: 
        image, label = dat[idx] 
        image = pil_to_tensor(image) 
        image = image*2/255. - .5 
        image = image.reshape([1, -1]) 
        x_list.append(image) 
        y_list.append(label) ## type(label) == int 
        pass 
    x = torch.cat(x_list) 
    y = one_hot(torch.tensor(y_list), num_classes=10).type(torch.float32) 
    return x, y 

## define models 

class BaseLayer(nn.Module): 
    def __init__(self, 
            abstraction_dimension=20): 
        super().__init__() 
        self.abstraction_dimension = abstraction_dimension 
        self.fc1 = nn.Linear(MNIST_DIM*MNIST_DIM, self.abstraction_dimension) 
        self.relu1 = nn.LeakyReLU() 
        self.fc2 = nn.Linear(self.abstraction_dimension, EMBEDDING_DIM) 
        ## memory 
        self.hessian_approximation = None 
        self.hessian_center = None 
        pass 
    def forward(self, 
            x): 
        x = self.fc1(x) 
        x = self.relu1(x) 
        x = self.fc2(x) 
        return x 
    def memorize_grad(self, size=1.): 
        grad = [param.grad.detach().reshape([-1]) for param in self.parameters()] 
        grad = torch.cat(grad).reshape([-1, 1])  
        hessian_approx = grad * grad 
        if self.hessian_approximation is None: 
            self.hessian_approximation = hessian_approx * size 
        else: 
            self.hessian_approximation += hessian_approx * size  
            pass 
        self.hessian_center = nn.utils.parameters_to_vector(self.parameters()).detach() 
        pass 
    pass 

class Classifier(nn.Module): 
    def __init__(self, 
            abstraction_dimension=20, 
            base_layer_transfer=None, 
            n_labels=10, 
            infinite_lambda=False): 
        super().__init__() 
        self.abstraction_dimension = abstraction_dimension 
        self.n_labels=10 
        self.infinite_lambda = infinite_lambda 
        self.base_layer = BaseLayer(abstraction_dimension=self.abstraction_dimension) 
        self.relu1 = nn.LeakyReLU() 
        self.fc1 = nn.Linear(EMBEDDING_DIM, self.n_labels) 
        self.softmax = nn.Softmax(dim=1) 
        if base_layer_transfer is not None: 
            self.base_layer.load_state_dict(base_layer_transfer.state_dict()) 
            self.base_layer.hessian_approximation = base_layer_transfer.hessian_approximation.clone()  
            self.base_layer.hessian_center = base_layer_transfer.hessian_center.clone() 
            pass 
        optim_params = self.parameters() 
        if self.infinite_lambda: 
            optim_params = list(set(self.parameters()) - set(self.base_layer.parameters())) 
            pass 
        self.optimizer = torch.optim.Adam(optim_params, lr=LEARNING_RATE) 
        self.loss = nn.SmoothL1Loss() 
        pass 
    def forward(self, 
            x): 
        x = self.base_layer(x) 
        x = self.relu1(x) 
        x = self.fc1(x) 
        x = self.softmax(x) 
        return x 
    def fit(self, 
            iters=100, 
            batch_size=100, 
            eval_size=1000, 
            train_data=None, 
            memorize=False, 
            use_memory=False, 
            eval_dataset=None): 
        ## fit 
        self.train() 
        for _ in range(iters): 
            self.optimizer.zero_grad() 
            x, y = sample(n=batch_size, dataset=train_data) 
            y_hat = self(x) 
            loss = self.loss(y, y_hat) 
            if use_memory: 
                lmbda = 1. 
                if type(use_memory) == float: 
                    lmbda = use_memory 
                    pass 
                p = nn.utils.parameters_to_vector(self.base_layer.parameters()) 
                diff = (p - self.base_layer.hessian_center).reshape([-1, 1]) 
                ## equivalent to a quadratic form 
                reg = diff * self.base_layer.hessian_approximation.reshape([-1, 1]) 
                reg = reg.transpose(0, 1).matmul(diff).reshape([]) 
                loss += reg * lmbda ## add regularizer since we're minimizing a loss 
                pass 
            loss.backward() 
            if memorize: 
                self.base_layer.memorize_grad(size=x.shape[0]) 
                pass 
            self.optimizer.step() 
            pass 
        pass 
        ## eval 
        self.eval() 
        x, y = sample(n=eval_size, dataset=eval_dataset) 
        y_hat = self(x) 
        acc = (y.argmax(dim=1) == y_hat.argmax(dim=1)).type(torch.float32).mean() 
        return float(acc) 
    pass 

