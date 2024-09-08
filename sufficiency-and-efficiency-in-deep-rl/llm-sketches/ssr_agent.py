## SSR Agents 
## Approximate sufficient statistics for deep nets 
## Optimally leverage old data as a regression target moves 

import random 
import torch 
import torch.nn as nn 
from lanczos import l_lanczos, combine_krylov_spaces 

# Define the actor and critic networks 
class SSRAgent(nn.Module): 
    def __init__(self, replay_buffer, ssr_rank=2): 
        super(SSRAgent, self).__init__() 
        self.device = None 
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
                'ssr_model_dimension': self.ssr_model_dimension  
                } 
        return d 
    def load_ssr_dict(self, d): 
        self.device = d['device'] 
        self.ssr_rank = d['ssr_rank'] 
        self.ssr_low_rank_matrix = d['ssr_low_rank_matrix'] 
        self.ssr_residual_matrix = d['ssr_residual_matrix'] 
        self.ssr_center = d['ssr_center'] 
        self.ssr_prev_center = d['ssr_prev_center'] 
        self.ssr_n = d['ssr_n'] 
        self.ssr_cov_trace = d['ssr_cov_trace'] 
        self.ssr_cov_n = d['ssr_cov_n'] 
        self.ssr_model_dimension = d['ssr_model_dimension'] 
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
    def memorize(self, n=None): 
        'memorize oldest `n` transitions, or all if `n is None`' 
        if n is None: 
            n = len(self.replay_buffer) 
            pass 
        self.ssr_prev_center = self.ssr_center 
        self.ssr_center = self.get_param().clone().detach() ## elliptical centroid 
        if self.ssr_model_dimension is None: 
            self.ssr_model_dimension = self.ssr_center.shape[0] 
            pass 
        ssr_low_rank_matrix, ssr_residual_diagonal = l_lanczos(self.__get_get_grad_generator(n), self.ssr_rank, self.ssr_model_dimension, calc_diag=True, device=self.device) 
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
            dt = self.ssr_center - self.ssr_prev_center 
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
        if lmbda is None: 
            lmbda = self.optimal_lambda() 
        return lmbda * .5 * ssr_mean ## TODO move lmbda out  
        pass 
    def optimal_lambda(self, pi_min=0., pi_max=1.): 
        "a rough approximation of lambda's optimal value" 
        if self.ssr_cov_trace is None: 
            return 0. 
        p0 = self.ssr_center 
        dt = p0 - self.ssr_prev_center 
        dt2_sum = (dt * dt).sum() ## TODO bad estimator, consider rayleigh quotient iteration  
        fi_inv_trace = self.ssr_cov_trace / self.ssr_cov_n 
        pi = 1. - .5 * fi_inv_trace / dt2_sum / self.ssr_n 
        if pi < pi_min: 
            pi = pi_min 
        if pi > pi_max: 
            pi = pi_max 
            pass 
        ## lmbda = self.ssr_n * (1. - pi) ## lambda = n_A 
        lmbda = 1. - pi ## dropping ssr_n as constant under optimization  
        return lmbda  
    def get_param(self):
        'only for SSR calculations'
        return torch.cat([p.reshape([-1, 1]) for p in self.parameters()], dim=0)
    def __get_get_grad_generator(self, n=None): 
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
                    self.zero_grad() 
                    transition = self.replay_buffer.sample(idx_list=[idx]) 
                    loss = self.loss(transition) 
                    loss.backward() 
                    grad_vec = torch.cat([p.grad.reshape([-1, 1]) for p in self.parameters()], dim=0).clone().detach()  
                    yield grad_vec 
                    pass
                pass 
            return grad_generator 
        return get_grad_generator 
    pass 

