## packages 

import tempfile 
import pickle
from tqdm import tqdm 
import matplotlib.pyplot as plt
import random
import os 
import shutil 
import zipfile
import math 
import numpy as np 

import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torchvision import datasets, transforms
from torch.utils.data import Dataset 
from torch import autograd 

from lanczos import l_lanczos, combine_krylov_spaces 
from az_blob_util import upload_to_blob_store, download_from_blob_store 

N_OUT = 10 
LEARNING_RATE = 0.001 
BATCH_SIZE = 100
TRAINING_ITERS = 1000 
MEMORIZATION_SIZE = 1000 
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

## data utils 

def download_mnist_to_blob_storage(): 
    with tempfile.TemporaryDirectory() as tmpdir: 
        ## download and zip data 
        data_path = os.path.join(tmpdir, 'mnist_data') 
        print(f'downloading mnist to local path: {data_path}') 
        mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=TRANSFORM) 
        mnist_test = datasets.MNIST(data_path, train=False, transform=TRANSFORM) 
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

def subset_dataset(dataset, n, drop_labels=[]):
    out = []
    for _ in range(n):
        idx = random.randint(0, len(dataset)-1)
        image, label = dataset[idx]
        while label in drop_labels:
            idx = random.randint(0, len(dataset)-1)
            image, label = dataset[idx]
            pass
        out.append((image, label))
    return out

def get_datasets(n=10000):  
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
        mnist_train = datasets.MNIST(mnist_dir, train=True, download=False, transform=TRANSFORM) 
        mnist_test = datasets.MNIST(mnist_dir, train=False, download=False, transform=TRANSFORM) 
        data_pack['01_train'] = subset_dataset(mnist_train, n=n, drop_labels=[2,3,4,5,6,7,8,9]) 
        data_pack['01_test']  = subset_dataset(mnist_test,  n=n, drop_labels=[2,3,4,5,6,7,8,9]) 
        data_pack['23_train'] = subset_dataset(mnist_train, n=n, drop_labels=[0,1,4,5,6,7,8,9]) 
        data_pack['23_test']  = subset_dataset(mnist_test,  n=n, drop_labels=[0,1,4,5,6,7,8,9]) 
        data_pack['45_train'] = subset_dataset(mnist_train, n=n, drop_labels=[0,1,2,3,6,7,8,9]) 
        data_pack['45_test']  = subset_dataset(mnist_test,  n=n, drop_labels=[0,1,2,3,6,7,8,9]) 
        data_pack['67_train'] = subset_dataset(mnist_train, n=n, drop_labels=[0,1,2,3,4,5,8,9]) 
        data_pack['67_test']  = subset_dataset(mnist_test,  n=n, drop_labels=[0,1,2,3,4,5,8,9]) 
        data_pack['89_train'] = subset_dataset(mnist_train, n=n, drop_labels=[0,1,2,3,4,5,6,7]) 
        data_pack['89_test']  = subset_dataset(mnist_test,  n=n, drop_labels=[0,1,2,3,4,5,6,7]) 
        pass 
    return data_pack 

def probability_merge_datasets(dataset1, dataset2, n, p): 
    'Randomly sample n obesrvations from dataset 1 & 2, sampling with probability p from dataset2.' 
    sample_n = np.random.binomial(n, p) 
    dataset1_subset = random.sample(dataset1, n - sample_n) 
    dataset2_subset = random.sample(dataset2, sample_n) 
    out = dataset1_subset + dataset2_subset 
    random.shuffle(out) ## shuffle in-place 
    return out 

## define model 

class Model(nn.Module): 
    def __init__(self, 
            losses=None, 
            accs=None,
            regs=None, 
            net_type='cnn', 
            batch_norm=True, 
            hessian_sum=None,
            hessian_sum_low_rank_half = None, 
            hessian_residual_variances = None, 
            hessian_denominator=None, 
            hessian_center=None, 
            log1p_reg=False): 
        super(Model, self).__init__() 
        ## init params 
        self.fl_params = None 
        self.named_fl_params = None 
        if net_type == 'dense': 
            self.fc1 = nn.Linear(28*28, 8) 
            if batch_norm:
                self.fc1_bn = nn.BatchNorm1d(8) 
            self.relu1 = nn.ReLU() 
            self.fc2 = nn.Linear(8, 16)  
            if batch_norm:
                self.fc2_bn = nn.BatchNorm1d(16) 
            self.relu2 = nn.ReLU() 
            self.fc3 = nn.Linear(16, N_OUT) 
            self.sigmoid = nn.Sigmoid() 
        elif net_type == 'dense_1': 
            self.fc1 = nn.Linear(28*28, 32) 
            if batch_norm:
                self.fc1_bn = nn.BatchNorm1d(32) 
            self.relu1 = nn.ReLU() 
            self.fc2 = nn.Linear(32, 64)  
            if batch_norm:
                self.fc2_bn = nn.BatchNorm1d(64) 
            self.relu2 = nn.ReLU() 
            self.fc3 = nn.Linear(64, N_OUT) 
            self.sigmoid = nn.Sigmoid() 
        elif net_type == 'cnn': 
            ## net structure 
            ## conv1 -> conv2 -> fc1 -> fc2 
            ## conv2 -> fl1 -> fl2 -> fc1 
            ## where fl1 & fl2 are "frontal lobe" fc blocks 
            ## 
            ## conv 
            self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=4) 
            if batch_norm: 
                self.conv1_bn = nn.BatchNorm2d(8) 
            self.relu1 = nn.ReLU() 
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2) 
            if batch_norm: 
                self.conv2_bn = nn.BatchNorm2d(32) 
            self.relu2 = nn.ReLU() 
            ## fl 
            self.fl1 = nn.Linear(16*2*2, 32) 
            self.fl1_relu = nn.ReLU() 
            if batch_norm: 
                self.fl1_bn = nn.BatchNorm1d(32) 
            self.fl2 = nn.Linear(32, 16) 
            self.fl2_relu = nn.ReLU() 
            if batch_norm: 
                self.fl2_bn = nn.BatchNorm1d(16) 
            ## store fl parameters, enabling clearing ## TODO get rid of this stuff, it doesn't work  
            self.fl_params = [] 
            self.fl_params.extend(list(self.fl1.parameters())) 
            self.fl_params.extend(list(self.fl2.parameters())) 
            self.named_fl_params = [] 
            self.named_fl_params.extend(list(self.fl1.named_parameters())) 
            self.named_fl_params.extend(list(self.fl2.named_parameters())) 
            ## fc 
            self.fc1 = nn.Linear(16*2*2 + 16, 16) 
            if batch_norm: 
                self.fc1_bn = nn.BatchNorm1d(16) 
            self.relu3 = nn.ReLU() 
            self.fc2 = nn.Linear(16, N_OUT) 
        elif net_type == 'nlp': 
            self.embedding = nn.Embedding(N_SHAKES_OUT, embedding_dim=32) 
            self.conv1 = nn.Conv1d(32, 64, kernel_size=5, stride=3) 
            if batch_norm: 
                self.conv1_bn = nn.BatchNorm1d(64) 
            self.relu1 = nn.ReLU() 
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=2) 
            if batch_norm: 
                self.conv2_bn = nn.BatchNorm1d(128) 
            self.relu2 = nn.ReLU() 
            self.fc1 = nn.Linear(128*2, 128) 
            if batch_norm: 
                self.fc1_bn = nn.BatchNorm1d(128) 
            self.relu3 = nn.ReLU() 
            self.fc2 = nn.Linear(128, N_SHAKES_OUT) 
            self.softmax = nn.Softmax(dim=1) 
            pass 
        ## data structures 
        self.hessian_sum = hessian_sum 
        self.hessian_sum_low_rank_half = hessian_sum_low_rank_half 
        self.hessian_residual_variances = hessian_residual_variances 
        self.hessian_denominator = hessian_denominator 
        self.hessian_center = hessian_center 
        if losses is None:
            self.losses = [] 
        else: 
            self.losses = losses 
            pass 
        if accs  is None:
            self.accs = [] 
        else: 
            self.accs = accs 
            pass 
        if regs  is None: 
            self.regs = [] 
        else: 
            self.regs = regs 
            pass 
        self.net_type = net_type
        self.batch_norm = batch_norm 
        self.log1p_reg = log1p_reg
        ## optimizer 
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE) 
        pass 
    def forward(self, x): 
        if self.net_type in ['dense', 'dense_1']: 
            x = self.fc1(x) 
            if self.batch_norm: 
                x = self.fc1_bn(x) 
            x = self.relu1(x) 
            x = self.fc2(x) 
            if self.batch_norm: 
                x = self.fc2_bn(x) 
            x = self.relu2(x) 
            x = self.fc3(x) 
            x = self.sigmoid(x) 
        elif self.net_type == 'cnn': 
            ## conv 
            x = self.conv1(x) 
            if self.batch_norm: 
                x = self.conv1_bn(x) 
            x = self.relu1(x) 
            x = self.conv2(x) 
            if self.batch_norm: 
                x = self.conv2_bn(x) 
            x = self.relu2(x) 
            x = x.flatten(start_dim=1) 
            ## fl 
            y = self.fl1(x) 
            if self.batch_norm: 
                y = self.fl1_bn(y) 
            y = self.fl1_relu(y) 
            y = self.fl2(y) 
            if self.batch_norm: 
                y = self.fl2_bn(y) 
            y = self.fl2_relu(y) 
            ## fc 
            x = torch.cat([x,y], dim=1) 
            x = self.fc1(x) 
            if self.batch_norm: 
                x = self.fc1_bn(x) 
            x = self.relu3(x) 
            x = self.fc2(x) 
        elif self.net_type == 'nlp': 
            x = self.embedding(x) 
            x = x.permute((0, 2, 1)) 
            x = self.conv1(x) 
            if self.batch_norm: 
                x = self.conv1_bn(x) 
            x = self.relu1(x) 
            x = self.conv2(x) 
            if self.batch_norm: 
                x = self.conv2_bn(x) 
            x = self.relu2(x) 
            x = x.flatten(start_dim=1) 
            x = self.fc1(x) 
            if self.batch_norm: 
                x = self.fc1_bn(x) 
            x = self.relu3(x) 
            x = self.fc2(x) 
            x = self.softmax(x) 
            pass 
        return x 
    def copy(self): 
        out = Model( 
            losses=self.losses.copy(), 
            accs=self.accs.copy(), 
            regs=self.regs.copy(), 
            net_type=self.net_type, 
            batch_norm=self.batch_norm, 
            hessian_sum=self.hessian_sum.detach().clone() if self.hessian_sum is not None else None, 
            hessian_sum_low_rank_half=self.hessian_sum_low_rank_half.detach().clone() if self.hessian_sum_low_rank_half is not None else None, 
            hessian_residual_variances=self.hessian_residual_variances.detach().clone() if self.hessian_residual_variances is not None else None, 
            hessian_denominator=self.hessian_denominator, 
            hessian_center=self.hessian_center.detach().clone() if self.hessian_center is not None else None, 
            log1p_reg=self.log1p_reg 
        ) 
        out.load_state_dict(self.state_dict()) 
        return out 
    def fit(self, training_dataset, testing_dataset, n_iters=TRAINING_ITERS, ams=False, drop_labels=[], 
            random_label_probability=0., silence_tqdm=False, acc_frequency=1, l2_reg=None, fl_reg=None, parameters=None, 
            information_minimum=None, high_info_prop=None): 
        ''' 
        fit the model 
        inputs: 
        - n_iters: how many optimizer iterations to run 
        - ams: use analytic memory system? if yes, set to lambda value 
        - drop_labels: omit these labels from training 
        - random_label_probability: probability of randomly selecting a label, instead of using the correct one 
        outputs:
        - idx_batch: returns a list of observation indices for optional, later memorization 
        side-effects: 
        - model parameter updates 
        ''' 
        if parameters is None: 
            parameters = list(self.parameters()) 
            pass 
        idx_batch = [] 
        pbar = tqdm(range(n_iters), disable=silence_tqdm) 
        for pbar_idx in pbar: 
            self.train() 
            self.zero_grad() 
            x, y, idx_list = self.__get_batch(training_dataset, drop_labels=drop_labels, random_label_probability=random_label_probability) 
            idx_batch.extend(idx_list) 
            loss_vec = self.__get_loss(x, y, reduction='none') 
            loss = loss_vec.mean() 
            reg = 0. 
            if ams and self.hessian_center is not None: 
                reg += ams * self.__get_regularizer(parameters=parameters) 
            if l2_reg is not None and self.hessian_center is not None: ## TODO not used, consider removing  
                p = self.get_parameter_vector(parameters=parameters) 
                p0 = self.hessian_center 
                l2_loss = (l2_reg*(p - p0).transpose(0,1).matmul(p - p0)).reshape([])  
                reg += l2_loss 
                pass 
            if fl_reg is not None: 
                reg += fl_reg * self.__fl_norm(loss_vec) 
                pass 
            loss += reg ## "+" because optimizer minimizes 
            self.regs.append(float(reg)) 
            loss_f = float(loss) 
            self.losses.append(loss_f) 
            loss.backward() 
            if high_info_prop is not None: 
                self.__zero_grads_above_minimum_information(self, high_info_proportion=high_info_prop) 
                pass 
            self.optimizer.step() 
            if pbar_idx % acc_frequency == 0: 
                if self.net_type in ['dense', 'dense_1', 'cnn']: 
                    self.accs.append(self.acc(testing_dataset)) 
                elif self.net_type in ['nlp']: 
                    raise Exception('Error: Feature not implemented!') 
                    pass  
                pass 
            pbar.set_description(f'loss: {loss_f}') 
            pass 
        return idx_batch  
    def memorize(self, dataset, drop_labels=[], memorization_size=MEMORIZATION_SIZE, 
            random_label_probability=0., silence_tqdm=False, krylov_rank=0, krylov_eps=0., idx_batch=None, parameters=None): 
        if parameters is None: 
            parameters = list(self.parameters()) 
            pass 
        self.eval() 
        if self.hessian_denominator is None: 
            self.hessian_denominator = 0. 
            pass 
        self.hessian_center = self.get_parameter_vector(parameters=parameters).detach().clone()  
        if idx_batch is not None: 
            if memorization_size < len(idx_batch): 
                ## compute times were annoyingly long, running a random sub-sample... 
                idx_batch = random.sample(idx_batch, memorization_size) 
                pass 
            pass 
        get_grad_generator = self.__get_get_grad_generator(dataset, n_grads=memorization_size, drop_labels=drop_labels, \
                random_label_probability=random_label_probability, idx_batch=idx_batch, parameters=parameters) 
        if krylov_rank < 1: 
            ## use full-rank Information Matrix estimate 
            if self.hessian_sum is None: 
                self.hessian_sum = 0. 
            grad_generator = get_grad_generator() 
            pbar = tqdm(grad_generator(), disable=silence_tqdm) 
            for grad in pbar: 
                self.hessian_sum += (grad.matmul(grad.transpose(1, 0))).detach() 
                self.hessian_denominator += 1 
                pass 
        elif self.hessian_sum_low_rank_half is None: 
            ## use Krylov method 
            p = int(self.get_parameter_vector(parameters=parameters).shape[0])  
            self.hessian_denominator += len(idx_batch)   
            self.hessian_sum_low_rank_half = l_lanczos(get_grad_generator, r=krylov_rank, p=p, eps=krylov_eps) 
            ## calculate hessian diagonal 
            hessian_sum_diagonal = self.__grad_sum(get_grad_generator()) 
            self.hessian_residual_variances = hessian_sum_diagonal - self.__outer_product_diagonal(self.hessian_sum_low_rank_half, self.hessian_sum_low_rank_half)
            self.hessian_residual_variances = self.hessian_residual_variances.maximum(torch.zeros(size=self.hessian_residual_variances.size())) ## clean-up numerical errors, keep it all >= e
        else: 
            ## update Krylov space 
            p = int(self.get_parameter_vector(parameters=parameters).shape[0]) 
            self.hessian_denominator += len(idx_batch)  
            new_krylov_space = l_lanczos(get_grad_generator, r=krylov_rank, p=p, eps=krylov_eps) 
            updated_krylov_space = combine_krylov_spaces(self.hessian_sum_low_rank_half, new_krylov_space, krylov_eps=krylov_eps) 
            self.hessian_sum_low_rank_half = updated_krylov_space 
            ## update diagonal 
            total_diagonal_variances = self.hessian_residual_variances + self.__outer_product_diagonal(self.hessian_sum_low_rank_half, self.hessian_sum_low_rank_half) 
            self.hessian_residual_variances = total_diagonal_variances - self.__outer_product_diagonal(self.hessian_sum_low_rank_half, self.hessian_sum_low_rank_half)
            self.hessian_residual_variances = self.hessian_residual_variances.maximum(torch.zeros(size=self.hessian_residual_variances.size()))  
            pass 
        pass 
    def __fl_norm(self, loss_vec): 
        'Frobenius norm of the information matrix, constrained to frontal lobe parameters' 
        ## combute jacobian [ d loglik(x_i) / d p_j ] 
        n = loss_vec.shape[0] 
        log_density = -loss_vec 
        J = [] 
        for i in range(n): 
            grad = autograd.grad(log_density[i], self.fl_params, create_graph=True, retain_graph=True, allow_unused=True) 
            grad = torch.cat([g.flatten() for g in grad]) 
            J.append(grad) 
            pass 
        J = torch.stack(J) 
        ## take norm and return 
        return J.mul(J).sum() / n 
    def __grad_sum(self, grad_generator):
        ''' 
        returns a sum of gradients. 
        divide by n to get a FIM diagonal estimate 
        '''
        out = 0.
        for g in grad_generator():
            out += g
            pass
        return out
    def __outer_product_diagonal(self, a, b):
        'numerically efficient caclulation of diag(a * b^T)'
        return (a*b).sum(dim=1).reshape([-1, 1]) 
    def acc(self, dataset, batch_size=1000, drop_labels=[]): 
        self.eval() 
        x, y, _ = self.__get_batch(dataset, batch_size=batch_size, drop_labels=drop_labels) 
        y_hat = self.forward(x) 
        acc = (y.argmax(dim=1) == y_hat.argmax(dim=1)).float().mean() 
        acc_f = float(acc) 
        return acc_f 
    def get_parameter_vector(self, parameters=None): 
        if parameters is None: 
            parameters = list(self.parameters()) 
            pass 
        return torch.cat([p.reshape([-1, 1]) for p in parameters]) 
    def save(self, path): 
        torch.save(self.state_dict(), path) 
        pass 
    def load(self, path): 
        self.load_state_dict(torch.load(path)) 
        pass 
    def __get_get_grad_generator(self, dataset, n_grads, parameters=None, **argv): 
        if parameters is None: 
            parameters = list(self.parameters()) 
            pass 
        def get_grad_generator(): 
            def grad_generator(): 
                for _ in range(n_grads): 
                    self.zero_grad() 
                    x, y, _ = self.__get_batch(dataset, batch_size=1, **argv) 
                    loss = self.__get_loss(x, y) 
                    loss.backward() 
                    grad = torch.cat([p.grad.reshape([-1, 1]) for p in parameters]) 
                    yield grad 
                pass 
            return grad_generator  
        return get_grad_generator 
    def __get_loss(self, x, y, reduction='mean'): 
        y_hat = self.forward(x) 
        loss = F.smooth_l1_loss(y_hat, y, reduction=reduction) 
        if reduction == 'none': 
            ## convert to vector 
            loss = loss.mean(dim=1)  
        return loss 
    def __get_regularizer(self, parameters=None): 
        if parameters is None: 
            parameters = list(self.parameters()) 
            pass 
        if self.hessian_center is None: 
            return 0. 
        p = self.get_parameter_vector(parameters=parameters) 
        p0 = self.hessian_center 
        if self.hessian_sum_low_rank_half is None: 
            hess = self.hessian_sum  
            reg = (p - p0).transpose(1, 0).matmul(hess).matmul(p - p0).reshape([]) 
        else: 
            hess_half = self.hessian_sum_low_rank_half.detach() 
            reg = (p - p0).transpose(1, 0).matmul(hess_half).matmul(hess_half.transpose(1, 0)).matmul(p - p0).reshape([]) 
            pass 
        if self.hessian_residual_variances is not None and reg is not None:
            reg += ((p - p0) * self.hessian_residual_variances * (p - p0)).sum() 
            pass
        reg /= self.hessian_denominator 
        if self.log1p_reg: 
            reg = torch.log1p(reg) 
        return reg 
    def __get_batch(self, dataset, batch_size=BATCH_SIZE, **argv): 
        if self.net_type in ['dense', 'dense_1', 'cnn']: 
            return self.__get_mnist_batch(dataset, batch_size=batch_size, **argv) 
        elif self.net_type in ['nlp']: 
            return self.__get_nlp_batch(dataset, batch_size=batch_size, **argv) 
        pass 
    def __get_nlp_batch(self, dataset, batch_size=BATCH_SIZE, random_label_probability=0., drop_labels=[], idx_batch=None): 
        x_list = [] 
        y_list = [] 
        idx_list = [] 
        if idx_batch is not None: 
            if batch_size < len(idx_batch): 
                idx_batch = random.sample(idx_batch, batch_size)  
        for i in range(batch_size): 
            if idx_batch is None: 
                idx = random.randint(0, len(dataset)-1) 
            else:
                idx = idx_batch[i] 
                pass 
            x, y = dataset[idx] 
            #y = torch.tensor([1. if int(y) == idx else 0. for idx in range(N_SHAKES_OUT)]) ## one-hot representation 
            if random_label_probability > 0.: 
                if bool(torch.rand([]) < random_label_probability): 
                    y = torch.tensor(random.randint(0, N_SHAKES_OUT-1)) 
                    pass 
                pass 
            y = nn.functional.one_hot(y, num_classes=N_SHAKES_OUT).float() 
            x = x.reshape([1, -1]) 
            y = y.reshape([1, -1]) 
            x_list.append(x) 
            y_list.append(y) 
            idx_list.append(idx) 
            pass 
        x = torch.cat(x_list, dim=0) 
        y = torch.cat(y_list, dim=0) 
        return x, y, idx_list  
    def __get_mnist_batch(self, dataset, batch_size=BATCH_SIZE, drop_labels=[], random_label_probability=0., idx_batch=None): 
        x_list = [] 
        y_list = [] 
        idx_list = [] 
        if idx_batch is not None:
            if batch_size < len(idx_batch):
                idx_batch = random.sample(idx_batch, batch_size) 
        for _ in range(batch_size): 
            idx = random.randint(0, len(dataset)-1) 
            while dataset[idx][1] in drop_labels: 
                idx = random.randint(0, len(dataset)-1) 
            image, label = dataset[idx] 
            if random_label_probability > 0.: 
                if bool(torch.rand([]) < random_label_probability): 
                    label = random.randint(0, N_OUT-1) 
                    pass 
                pass 
            x, y = self.__build_decimal_datum(image, label)  
            x_list.append(x) 
            y_list.append(y) 
            idx_list.append(idx) 
            pass 
        x = torch.cat(x_list, dim=0) 
        y = torch.cat(y_list, dim=0)  
        return x, y, idx_list 
    def __build_decimal_datum(self, image, label): 
        'one-hot encode an image-label pair' 
        y = torch.tensor([1. if label == idx else 0. for idx in range(N_OUT)]) ## one-hot representation 
        ## reshape to column vectors 
        x = None 
        if self.net_type in ['dense', 'dense_1']: 
            x = image.reshape([1, -1]) 
        elif self.net_type == 'cnn': 
            x = image.unsqueeze(dim=0) 
            pass 
        y = y.reshape([1, -1]) 
        return x, y 
    def get_information_diagonal(self): 
        info_vec = None 
        if self.hessian_sum is not None:
            info_vec = torch.diag(self.hessian_sum)
        elif self.hessian_sum_low_rank_half is not None:
            info_vec = self.__outer_product_diagonal(self.hessian_sum_low_rank_half, self.hessian_sum_low_rank_half)
            info_vec += self.hessian_residual_variances
            pass 
        info_vec = info_vec / self.hessian_denominator 
        return info_vec 
    def __zero_grads_above_minimum_information(self, minimum_information=None, high_info_proportion=None): 
        '''
        Apply before `optimizer.step()` to avoid adjusting high-information parameters. 
        if `minimum_information is not None`, then that cutoff is used. 
        if `high_info_proportion is not None`, then a `high_info_proportion` proportion of highest-info parameters will be frozen. 
            Expects a float in [0,1]. 
            Example: .2 => 20% of the highest-info parameters will be frozen 
        '''
        info_vec = self.get_information_diagonal() 
        if info_vec is None: 
            ## information not yet estimated 
            return None 
        if high_info_proportion is not None: 
            info_vec_n = info_vec.shape[0] 
            cut_off_idx = math.ceil(info_vec_n * high_info_proportion) 
            sorted_info_vec = torch.sort(info_vec, descending=True) 
            minimum_information = sorted_info_vec[cut_off_idx] 
            pass 
        ptr = 0 
        ## parameter-wise grad zero-ing 
        for param in self.parameters(): 
            n = param.grad.shape[0] ## TODO assuming grads are vecs here. Verify!  
            param.grad[info_vec[ptr:(ptr+n)] >= minimum_information] = 0. 
            ptr = n + 1 
            pass 
        pass 
    pass 


