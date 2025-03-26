## SSR Agents 
## Approximate sufficient statistics for deep nets 
## Optimally leverage old data as a regression target moves 

import random 
import torch 
import torch.nn as nn 
from lanczos import l_lanczos, combine_krylov_spaces 

GPU = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
CPU = torch.device('cpu') 

# Define the actor and critic networks 
class SSRAgent(nn.Module): 
    'Abstract SSRAgent class. Define `loss` in concrete subclass.' 
    def __init__(self, replay_buffer, ssr_rank=2, gpu_saver=True, dt_mean_N=10): 
        '''Initialize core, abstract SSRAgent. 
        args:
         - replay_buffer: instance of the `replay_buffer` class, holds reinforcement learning transitions 
         - ssr_rank: increases Hessian approximation accuracy, but needs [ssr_rank]*[model dim] RAM. Keep it low 
         - gpu_saver: save GPU RAM by moving non-core processing to CPU 
         - dt_mean_N: Statistical manifold traversal is assumed to follow a trended Brownian motion, with stats estimated up to `dt_mean_N` samples. 
        '''
        super(SSRAgent, self).__init__() 
        self.device = GPU 
        self.gpu_saver = self.device 
        if self.gpu_saver: 
            self.gpu_saver = CPU 
            pass 
        self.ssr_rank = ssr_rank 
        self.ssr_low_rank_matrix = None ## =: A 
        self.ssr_residual_diagonal = None ## =: resid 
        ## N * Fisher Information \approx AA^T + resid 
        self.ssr_center = None 
        self.ssr_prev_center = None 
        self.ssr_n = None 
        self.ssr_cov_trace = None 
        self.ssr_cov_n = None 
        self.ssr_model_dimension = None 
        self.dt_mean = None 
        self.dt_mean_N = dt_mean_N 
        self.dt_mean_trend = torch.tensor(0.).to(self.device) ## init to 0 heuristically since traversal is continuous from a very stable point 
        self.dt_mean_norm_trend = 0. 
        self.dt_mean_trace_cov = 0.  
        self.dt_prev_pi = .5 
        self.replay_buffer = replay_buffer 
        pass 
    def ssr_dict(self): 
        d = {'device': self.device,
                'ssr_rank': self.ssr_rank, 
                'ssr_low_rank_matrix': self.ssr_low_rank_matrix, 
                'ssr_residual_diagonal': self.ssr_residual_diagonal, 
                'ssr_center': self.ssr_center, 
                'ssr_prev_center': self.ssr_prev_center, 
                'ssr_n': self.ssr_n, 
                'ssr_cov_trace': self.ssr_cov_trace, 
                'ssr_cov_n': self.ssr_cov_n, 
                'ssr_model_dimension': self.ssr_model_dimension,  
                'dt_mean_N': self.dt_mean_N, 
                'dt_mean_trend': self.dt_mean_trend, 
                'dt_mean_norm_trend': self.dt_mean_norm_trend, 
                'dt_mean_trace_cov': self.dt_mean_trace_cov, 
                'dt_prev_pi': self.dt_prev_pi  
                } 
        return d 
    def load_ssr_dict(self, d): 
        self.device = d['device'] 
        self.ssr_rank = d['ssr_rank'] 
        self.ssr_low_rank_matrix = d['ssr_low_rank_matrix'].to(self.device) 
        self.ssr_residual_diagonal = d['ssr_residual_diagonal'].to(self.device) 
        self.ssr_center = d['ssr_center'].to(self.device) 
        self.ssr_prev_center = d['ssr_prev_center'].to(self.gpu_saver) 
        self.ssr_n = d['ssr_n'] 
        self.ssr_cov_trace = d['ssr_cov_trace'] 
        self.ssr_cov_n = d['ssr_cov_n'] 
        self.ssr_model_dimension = d['ssr_model_dimension'] 
        self.dt_mean_N = d['dt_mean_N'] 
        self.dt_mean_trend = d['dt_mean_trend'] 
        self.dt_mean_norm_trend = d['dt_mean_norm_trend'] 
        self.dt_mean_trace_cov = d['dt_mean_trace_cov'] 
        self.dt_prev_pi = d['dt_prev_pi'] 
        pass 
    def save(self, path):
        torch.save(self.state_dict(), path + '.state.pt')
        torch.save(self.ssr_dict(), path + '.ssr.pt')
        pass 
    def load(self, path): 
        self.load_state_dict(torch.load(path + '.state.pt')) 
        self.load_ssr_dict(torch.load(path + '.ssr.pt')) 
        pass 
    def loss(self, transitions): 
        raise NotImplementedError('ERROR: loss not implemented!') 
    def memorize(self, n=None, random_idx=False, disable_tqdm=False): 
        'memorize oldest `n` transitions, or all if `n is None`' 
        if n is None: 
            n = len(self.replay_buffer) 
            pass 
        ## track current and prev estimates 
        self.ssr_prev_center = self.ssr_center.to(self.gpu_saver) if self.ssr_center is not None else None  
        self.ssr_center = self.get_param().clone().detach() ## elliptical centroid 
        ## get model dim if we don't already have it 
        if self.ssr_model_dimension is None: 
            self.ssr_model_dimension = self.ssr_center.shape[0] 
            pass 
        ## updates dt stats 
        if self.ssr_prev_center is not None: 
            ## use low-mem approximate moving averages 
            rescale = ( self.dt_mean_N - 1 ) / self.dt_mean_N 
            if self.dt_mean_trend.shape == torch.Size([]): 
                ## prepare for a broadcast operation 
                self.dt_mean_trend = float(self.dt_mean_trend)
                pass 
            self.dt_mean_trend *= rescale 
            self.dt_mean_trend += (self.ssr_center - self.ssr_prev_center)/(self.dt_mean_N) ## not spending memory to store many dts 
            self.dt_mean_norm_trend *= rescale 
            self.dt_mean_norm_trend += (self.ssr_center - self.ssr_prev_center).pow(2).sum()/(self.dt_mean_N)
            self.dt_mean_trace_cov *= rescale 
            self.dt_mean_trace_cov += (self.ssr_center - self.ssr_prev_center - self.dt_mean_trend).pow(2).sum() / (self.dt_mean_N) 
            pass
        ## limited memory Lanczos algo calculates Krylov space for new data's information matrix 
        ssr_low_rank_matrix, ssr_residual_diagonal = l_lanczos(self.__get_get_grad_generator(n, random_idx=random_idx), self.ssr_rank, self.ssr_model_dimension, calc_diag=True, device=self.device, disable_tqdm=disable_tqdm) 
        ## handle l-Lanczos outputs 
        if self.ssr_low_rank_matrix is None: 
            ## first memorization 
            self.ssr_low_rank_matrix = ssr_low_rank_matrix 
            self.ssr_residual_diagonal = ssr_residual_diagonal 
            self.ssr_n = n 
        else: 
            ## combine with previous memories 
            self.ssr_low_rank_matrix = combine_krylov_spaces(self.ssr_low_rank_matrix, ssr_low_rank_matrix, device=self.device) 
            self.ssr_residual_diagonal += ssr_residual_diagonal 
            self.ssr_n += n 
            pass 
        if self.ssr_prev_center is not None:
            dt = self.ssr_center.to(self.gpu_saver) - self.ssr_prev_center 
            if self.ssr_cov_trace is None:
                self.ssr_cov_trace = (dt * dt).sum()
                self.ssr_cov_n = 1
            else:
                self.ssr_cov_trace += (dt * dt).sum()
                self.ssr_cov_n += 1
                pass
            pass 
        pass 
    def ssr(self, lmbda=None): 
        '''Get the ssr regularizer. If `lmbda is None`, `lmbda` will be set to 1 when `self.ssr_prev_center is None`, 
        otherwise `lmbda` will be the approximately optimal `n_A` value.'''
        if self.ssr_low_rank_matrix is None: 
            return 0. 
        p = self.get_param() 
        p0 = self.ssr_center 
        d = p - p0 
        A = self.ssr_low_rank_matrix 
        res = self.ssr_residual_diagonal 
        dTA = d.transpose(0,1).matmul(A) 
        ATd = dTA.transpose(0,1) 
        dTresd = (d * res).transpose(0,1).matmul(d) 
        ssr_sum = dTA.matmul(ATd) + dTresd 
        ssr_mean = ssr_sum / self.ssr_n 
        return .5 * ssr_mean  
    def optimal_lambda(self, pi_min=0., pi_max=1., return_pi=True): 
        "a linear approximation of pi or lambda's optimal value" 
        if self.dt_mean_norm_trend == 0.: 
            pi = torch.tensor(.5).to(self.device) 
        else: 
            pi = 1. - .5 * self.dt_mean_trace_cov / self.dt_mean_norm_trend  
            #pi = 1. - .5 * self.dt_prev_pi * self.dt_mean_trace_cov / self.dt_mean_norm_trend ## DEBUGGING 
            pi = pi.to(self.device).clone().detach()  
            pass 
        if float(pi) < pi_min: 
            pi = torch.tensor(pi_min).to(self.device) 
        if float(pi) > pi_max: 
            pi = torch.tensor(pi_max).to(self.device) 
            pass 
        if not return_pi: 
            lmbda = self.ssr_n * (1. - pi) ## lambda = n_A 
            return lmbda 
        ## returning pi, probability of sampling with theta_B 
        return pi 
    def get_param(self): 
        'only for SSR calculations' 
        return torch.cat([p.reshape([-1, 1]) for p in self.parameters()], dim=0)     
    def fit(self, batch_size, iters=1, pi_min=.1, pi_max=.9): 
        self.train() 
        self.dt_prev_pi = pi = self.optimal_lambda(pi_min=pi_min, pi_max=pi_max) 
        for _ in range(iters): 
            self.optimizer.zero_grad() 
            data = self.replay_buffer.sample(batch_size=batch_size) 
            loss = self.loss(data) 
            loss = pi * loss + (1 - pi) * self.ssr() 
            loss.backward() 
            self.optimizer.step() 
            pass 
        return float(pi), float(loss) 
    def __get_get_grad_generator(self, n=None, random_idx=False): 
        ## The double get hides `self` in a function context,  
        ## packaging `get_grad_generator` for calling without 
        ## the SSRAgent instance.  
        if n is None: 
            n = self.replay_buffer.n 
            pass 
        if n > self.replay_buffer.n: 
            n = self.replay_buffer.n 
            pass 
        def get_grad_generator(): 
            'l-Lanczos alg uses grad at least `ssr_rank` times' 
            def grad_generator(): 
                self.eval() 
                for idx in range(n): 
                    if random_idx: 
                        idx = random.randint(0, len(self.replay_buffer)-1)
                        pass 
                    self.zero_grad() 
                    transition = self.replay_buffer.sample(idx_list=[idx]) 
                    loss = self.loss(transition) 
                    loss.backward() 
                    for p in self.parameters(): 
                        if p.grad is None: 
                            p.grad = torch.zeros(size=p.shape) 
                        pass 
                    grad_vec = torch.cat([p.grad.reshape([-1, 1]) for p in self.parameters()], dim=0).clone().detach()  
                    yield grad_vec 
                    pass
                pass 
            return grad_generator 
        return get_grad_generator 
    pass 

