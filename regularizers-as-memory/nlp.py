## build MNIST data 

from tqdm import tqdm 
#from torchvision import datasets, transforms 
from torch.utils.data import Dataset 

import matplotlib.pyplot as plt 
import random 

#transform=transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.1307,), (0.3081,)) 
#    ]) 
#
#dataset1 = datasets.MNIST('../../data', train=True, download=True, transform=transform) 
#dataset2 = datasets.MNIST('../../data', train=False, transform=transform) 
#
#def subset_dataset(dataset, n, drop_labels=[]): 
#    out = [] 
#    for _ in range(n): 
#        idx = random.randint(0, len(dataset)-1) 
#        image, label = dataset[idx] 
#        while label in drop_labels: 
#            idx = random.randint(0, len(dataset)-1)  
#            image, label = dataset[idx] 
#            pass 
#        out.append((image, label)) 
#    return out 
#
#dataset1_0_to_4_n1000 = subset_dataset(dataset1, n=1000, drop_labels=[5,6,7,8,9]) 
#dataset1_5_to_9_n1000 = subset_dataset(dataset1, n=1000, drop_labels=[0,1,2,3,4]) 
#dataset1_5_to_9_n30 = subset_dataset(dataset1, n=60, drop_labels=[0,1,2,3,4]) 
#dataset1_5_to_9_n30_1 = subset_dataset(dataset1, n=60, drop_labels=[0,1,2,3,4]) 
#
#dataset2_0_to_4_n1000 = subset_dataset(dataset2, n=1000, drop_labels=[5,6,7,8,9]) 
#dataset2_5_to_9_n1000 = subset_dataset(dataset2, n=1000, drop_labels=[0,1,2,3,4]) 
#
#class BiasedDataset(Dataset):
#    'sample from 5 to 9 with probability `p`'
#    def __init__(self,
#            p=.5
#            ):
#        self.p = p 
#        pass  
#    def __len__(self):
#        return 1000 
#    def __getitem__(self, idx): 
#        d = dataset1_0_to_4_n1000
#        if bool(torch.rand([]) < self.p):
#            d = dataset1_5_to_9_n1000
#            pass 
#        image, label = d[idx] 
#        return image, label 
#    pass 

## build NLP data 

import torch 
import pickle 
from collections import Counter 

## You'll need to uncomment this before running `build_nlp_data` 
#!pip install nltk

def build_nlp_data():
    import nltk
    nltk.download('punkt')
    import requests 
    from nltk.tokenize import word_tokenize
    shakespeare = requests.get('https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt').text 
    tokens = word_tokenize(shakespeare) 
    unique_tokens = list(set(tokens)) 
    map = dict(zip(unique_tokens, list(range(len(unique_tokens))))) 
    tokens_as_ints = [] 
    for token in tokens:
        tokens_as_ints.append(map[token]) 
        pass 
    with open('shakespeare_tokens.pkl', 'wb') as f: 
        pickle.dump(tokens_as_ints, f)
        pass 
    pass 

RARE_TOKEN_CUTOFF = 10 
MAX_TOKENS = 100 

def load_shakes_tokens(): 
    with open('shakespeare_tokens.pkl', 'rb') as f: 
        shakes_tokens = pickle.load(f) 
        pass 
    ## map rare values to zero, then re-index tokens 
    shakes_tokens = [t+1 for t in shakes_tokens] ## reserve 0 for rare tokens 
    token_to_count_map = dict(Counter(shakes_tokens)) 
    idx = 1
    for token, count in token_to_count_map.items():
        if count < RARE_TOKEN_CUTOFF: 
            token_to_count_map[token] = 0 ## repurpose `count` to `new_index`
        else: 
            token_to_count_map[token] = idx 
            idx += 1 
            pass 
        pass 
    shakes_tokens = [token_to_count_map[token] for token in shakes_tokens] 
    ## reduce number of tokens 
    shakes_tokens = [ token % MAX_TOKENS for token in shakes_tokens] 
    return shakes_tokens 

#build_nlp_data() 
shakes_tokens = load_shakes_tokens() 
N_SHAKES_OUT = len(set(shakes_tokens)) 
print(f'Number of unique tokens: {N_SHAKES_OUT}') 
SHAKES_SERIES_LEN = 20 
N_TRANSFER_LEARNING_SIZE = 10 

class NLPDataset(Dataset):
    'converts a sequence of token indices into a dataset'
    def __init__(self, 
            token_list, 
            sample_length=10, 
            ): 
        ## store content 
        self.token_list = torch.tensor(token_list) 
        self.sample_length = sample_length 
        self.n = len(token_list) - sample_length - 1 ## -1 for final token to be used as a label 
        pass  
    def __len__(self): 
        return self.n 
    def __getitem__(self, idx): 
        x = self.token_list[idx:(idx+self.sample_length)] 
        y = self.token_list[idx + self.sample_length] 
        return x, y 
    pass 

class EvenOrOddNLPDataset(Dataset):
    'subsamples an NLPDataset to either even or odd values'
    def __init__(self, nlp_dataset, even=True, random_subset_size=None): 
        self.nlp_dataset = nlp_dataset 
        self.even = even 
        even_odd_filter = None 
        def even_odd_filter(value):
            if even:
                if value % 2 == 0: 
                    return True 
            else: ## odd 
                if value % 2 == 1:
                    return True 
                pass 
            return False 
        if random_subset_size is None: 
            self.indices = [idx for idx in range(len(nlp_dataset)) if even_odd_filter(int(nlp_dataset[idx][1]))] 
        else:
            ## subset the data randomly 
            self.indices = [] 
            for _ in range(random_subset_size): 
                continue_loop = True 
                while continue_loop: 
                    idx = random.randint(0, len(nlp_dataset)-1) 
                    if even_odd_filter(int(nlp_dataset[idx][1])): 
                        ## found! add to indices 
                        self.indices.append(idx) 
                        ## stop looking 
                        continue_loop = False 
                        pass 
                    pass 
                pass 
            pass
        self.n = len(self.indices) 
        pass 
    def __len__(self): 
        return self.n 
    def __getitem__(self, idx): 
        idx = self.indices[idx]  
        return self.nlp_dataset[idx] 
    pass 

shakes_tokens_train = shakes_tokens[:700000] 
shakes_tokens_test  = shakes_tokens[700000:] 
nlp_train = NLPDataset(shakes_tokens_train, sample_length=SHAKES_SERIES_LEN) 
nlp_test = NLPDataset(shakes_tokens_test, sample_length=SHAKES_SERIES_LEN) 
#nlp_even_train = EvenOrOddNLPDataset(nlp_train, even=True, random_subset_size=N_TRANSFER_LEARNING_SIZE) ## subset demonstrates transfer learning 
#nlp_odd_train = EvenOrOddNLPDataset(nlp_train, even=False) 
#nlp_even_test = EvenOrOddNLPDataset(nlp_test, even=True) 
#nlp_odd_test = EvenOrOddNLPDataset(nlp_test, even=False) 

#print(f'len(nlp_even_train): {len(nlp_even_train)}, counts: {dict(Counter([int(nlp_even_train[idx][1]) for idx in range(len(nlp_even_train))]))}') 
#print(f'len(nlp_odd_train): {len(nlp_odd_train)}, counts: {dict(Counter([int(nlp_odd_train[idx][1]) for idx in range(len(nlp_odd_train))]))}')
#print(f'len(nlp_even_test): {len(nlp_even_test)}, counts: {dict(Counter([int(nlp_even_test[idx][1]) for idx in range(len(nlp_even_test))]))}')
#print(f'len(nlp_odd_test): {len(nlp_odd_test)}, counts: {dict(Counter([int(nlp_odd_test[idx][1]) for idx in range(len(nlp_odd_test))]))}') 

## define model 

import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 

from lanczos import l_lanczos, combine_krylov_spaces  

N_OUT = 10 
LEARNING_RATE = 0.001 
BATCH_SIZE = 100
TRAINING_ITERS = 1000 
MEMORIZATION_SIZE = 1000 

class Model(nn.Module): 
    def __init__(self, 
            losses=None, 
            accs_low=None,
            accs_high=None, 
            regs=None, 
            net_type='dense', 
            batch_norm=True, 
            hessian_sum=None,
            hessian_sum_low_rank_half = None, 
            hessian_denominator=None, 
            hessian_center=None, 
            log1p_reg=False): 
        super(Model, self).__init__() 
        ## init params 
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
            self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=4) 
            if batch_norm: 
                self.conv1_bn = nn.BatchNorm2d(8) 
            self.relu1 = nn.ReLU() 
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2) 
            if batch_norm: 
                self.conv2_bn = nn.BatchNorm2d(32) 
            self.relu2 = nn.ReLU() 
            self.fc1 = nn.Linear(16*2*2, 16) 
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
        self.hessian_denominator = hessian_denominator 
        self.hessian_center = hessian_center 
        if losses is None:
            self.losses = [] 
        else: 
            self.losses = losses 
            pass 
        if accs_low  is None:
            self.accs_low = [] 
        else: 
            self.accs_low = accs_low 
            pass 
        if accs_high  is None: 
            self.accs_high = [] 
        else: 
            self.accs_high = accs_high 
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
            accs_low=self.accs_low.copy(), 
            accs_high=self.accs_high.copy(), 
            regs=self.regs.copy(), 
            net_type=self.net_type, 
            batch_norm=self.batch_norm, 
            hessian_sum=self.hessian_sum.detach().clone() if self.hessian_sum is not None else None, 
            hessian_sum_low_rank_half=self.hessian_sum_low_rank_half.detach().clone() if self.hessian_sum_low_rank_half is not None else None, 
            hessian_denominator=self.hessian_denominator, 
            hessian_center=self.hessian_center.detach().clone() if self.hessian_center is not None else None, 
            log1p_reg=self.log1p_reg 
        ) 
        out.load_state_dict(self.state_dict()) 
        return out 
    def fit(self, training_dataset, n_iters=TRAINING_ITERS, ams=False, drop_labels=[], 
            random_label_probability=0., silence_tqdm=False, acc_frequency=1, halt_acc=None, nlp_even_test=None, nlp_odd_test=None): 
        ''' 
        fit the model 
        inputs: 
        - n_iters: how many optimizer iterations to run 
        - ams: use analytic memory system? if yes, set to lambda value 
        - drop_labels: omit these labels from training 
        - random_label_probability: probability of randomly selecting a label, instead of using the correct one 
        side-effects: 
        - model parameter updates 
        ''' 
        pbar = tqdm(range(n_iters), disable=silence_tqdm) 
        for pbar_idx in pbar: 
            self.train() 
            self.zero_grad() 
            x, y = self.__get_batch(training_dataset, drop_labels=drop_labels, random_label_probability=random_label_probability) 
            loss = self.__get_loss(x, y) 
            if ams: 
                reg = self.__get_regularizer() 
                loss += ams * reg ## "+" because optimizer minimizes 
                self.regs.append(float(ams * reg)) 
            else: 
                self.regs.append(0.) 
                pass 
            loss_f = float(loss) 
            self.losses.append(loss_f) 
            loss.backward() 
            self.optimizer.step() 
            if pbar_idx % acc_frequency == 0: 
                if self.net_type in ['dense', 'dense_1', 'cnn']: 
                    self.accs_low.append(self.acc(dataset2_0_to_4_n1000)) 
                    self.accs_high.append(self.acc(dataset2_5_to_9_n1000)) 
                elif self.net_type in ['nlp'] and nlp_even_test is not None and nlp_odd_test is not None: 
                    self.accs_low.append(self.acc(nlp_odd_test, batch_size=100)) ## TODO small batch size for testing purposes 
                    high_acc = self.acc(nlp_even_test, batch_size=100) 
                    self.accs_high.append(high_acc) 
                    if halt_acc is not None: 
                        ## TODO WARNING: DATA LEAKAGE: fitting informed by test dataset.   
                        ## THIS FEATURE DOES NOT YET PRODUCE LEGITIMATE RESULTS 
                        if high_acc > halt_acc: 
                            self.hessian_center = self.get_parameter_vector().detach() 
                            halt_acc = high_acc
                        pass 
                    pass  
                pass 
            pbar.set_description(f'loss: {loss_f}') 
            pass 
        pass 
    def memorize(self, dataset, drop_labels=[], memorization_size=MEMORIZATION_SIZE, 
            random_label_probability=0., silence_tqdm=False, krylov_rank=0, krylov_eps=0.): 
        self.eval() 
        if self.hessian_denominator is None: 
            self.hessian_denominator = 0. 
            pass 
        self.hessian_center = self.get_parameter_vector().detach() 
        get_grad_generator = self.__get_get_grad_generator(dataset, n_grads=memorization_size, drop_labels=drop_labels, random_label_probability=random_label_probability) 
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
            p = int(self.get_parameter_vector().shape[0])  
            self.hessian_denominator += memorization_size 
            self.hessian_sum_low_rank_half = l_lanczos(get_grad_generator, r=krylov_rank, p=p, eps=krylov_eps) 
        else: 
            ## update Krylov space 
            p = int(self.get_parameter_vector().shape[0]) 
            self.hessian_denominator += memorization_size 
            new_krylov_space = l_lanczos(get_grad_generator, r=krylov_rank, p=p, eps=krylov_eps) 
            updated_krylov_space = combine_krylov_spaces(self.hessian_sum_low_rank_half, new_krylov_space, krylov_eps=krylov_eps) 
            self.hessian_sum_low_rank_half = updated_krylov_space 
            pass 
        pass 
    def acc(self, dataset, batch_size=1000, drop_labels=[]): 
        self.eval() 
        x, y = self.__get_batch(dataset, batch_size=batch_size, drop_labels=drop_labels) 
        y_hat = self.forward(x) 
        acc = (y.argmax(dim=1) == y_hat.argmax(dim=1)).float().mean() 
        acc_f = float(acc) 
        return acc_f 
    def get_parameter_vector(self): 
        return torch.cat([p.reshape([-1, 1]) for p in self.parameters()]) 
    def save(self, path): 
        torch.save(self.state_dict(), path) 
        pass 
    def load(self, path): 
        self.load_state_dict(torch.load(path)) 
        pass 
    def __get_get_grad_generator(self, dataset, n_grads, **argv):
        def get_grad_generator(): 
            def grad_generator(): 
                for _ in range(n_grads): 
                    self.zero_grad() 
                    x, y = self.__get_batch(dataset, batch_size=1, **argv) 
                    loss = self.__get_loss(x, y) 
                    loss.backward() 
                    grad = torch.cat([p.grad.reshape([-1, 1]) for p in self.parameters()]) 
                    yield grad 
                pass 
            return grad_generator  
        return get_grad_generator 
    def __get_loss(self, x, y): 
        y_hat = self.forward(x) 
        loss = F.smooth_l1_loss(y_hat, y) 
        return loss 
    def __get_regularizer(self): 
        if self.hessian_sum is None and self.hessian_sum_low_rank_half is None: 
            return 0. 
        p = self.get_parameter_vector() 
        p0 = self.hessian_center 
        if self.hessian_sum_low_rank_half is None: 
            hess = self.hessian_sum / self.hessian_denominator 
            reg = (p - p0).transpose(1, 0).matmul(hess).matmul(p - p0).reshape([]) 
        else: 
            hess_half = (self.hessian_sum_low_rank_half / self.hessian_denominator).detach()  
            reg = (p - p0).transpose(1, 0).matmul(hess_half).matmul(hess_half.transpose(1, 0)).matmul(p - p0).reshape([])
            pass 
        if self.log1p_reg: 
            reg = torch.log1p(reg) 
        return reg 
    def __get_batch(self, dataset, batch_size=BATCH_SIZE, **argv): 
        if self.net_type in ['dense', 'dense_1', 'cnn']: 
            return self.__get_mnist_batch(dataset, batch_size=batch_size, **argv) 
        elif self.net_type in ['nlp']: 
            return self.__get_nlp_batch(dataset, batch_size=batch_size, **argv) 
        pass 
    def __get_nlp_batch(self, dataset, batch_size=BATCH_SIZE, random_label_probability=0., drop_labels=[]): 
        x_list = [] 
        y_list = [] 
        for _ in range(batch_size): 
            idx = random.randint(0, len(dataset)-1) 
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
            pass 
        x = torch.cat(x_list, dim=0) 
        y = torch.cat(y_list, dim=0) 
        return x, y  
    def __get_mnist_batch(self, dataset, batch_size=BATCH_SIZE, drop_labels=[], random_label_probability=0.): 
        x_list = [] 
        y_list = [] 
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
            pass 
        x = torch.cat(x_list, dim=0) 
        y = torch.cat(y_list, dim=0)  
        return x, y 
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
    pass 


