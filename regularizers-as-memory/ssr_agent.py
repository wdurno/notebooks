## SSR Agents 
## Approximate sufficient statistics for deep nets 
## Optimally leverage old data as a regression target moves 

import random 
import torch 
import torch.nn as nn 
from lanczos import l_lanczos, combine_krylov_spaces 

class Object(object):
    pass

class ReplayBuffer(): 
    def __init__(self, capacity=10000): 
        self.n = 0 
        self.capacity = capacity 
        self.state_list = [] 
        self.action_list = [] 
        self.reward_list = [] 
        self.next_state_list = [] 
        self.done_list = [] 
        pass 
    def __len__(self):
         return self.n 
    def add(self, state, action, reward, next_state, done): 
        if self.n >= self.capacity: 
            ## discard earliest observation 
            self.state_list = self.state_list[1:] 
            self.action_list = self.action_list[1:] 
            self.reward_list = self.reward_list[1:] 
            self.next_state_list = self.next_state_list[1:] 
            self.done_list = self.done_list[1:] 
            self.n -= 1 
        pass 
        ## cast to torch  
        state = torch.tensor(state) 
        action = torch.tensor(action) 
        reward = torch.tensor(reward) 
        next_state = torch.tensor(next_state) 
        done = torch.tensor(done) 
        ## append to buffer 
        self.state_list.append(state) 
        self.action_list.append(action) 
        self.reward_list.append(reward) 
        self.next_state_list.append(next_state) 
        self.done_list.append(done) 
        self.n += 1 
        pass 
    def sample(self, batch_size=32, idx_list=None): 
        ## sample lists 
        out = Object() ## transitions 
        out.state = [] 
        out.action = [] 
        out.reward = [] 
        out.next_state = [] 
        out.done = [] 
        if idx_list is not None: 
            batch_size = len(idx_list) 
        for i in range(batch_size): 
            if idx_list is None: 
                idx = random.randint(0, self.n-1) 
            else: 
                idx = idx_list[i]
                pass 
            out.state.append(self.state_list[idx]) 
            out.action.append(self.action_list[idx]) 
            out.reward.append(self.reward_list[idx]) 
            out.next_state.append(self.next_state_list[idx]) 
            out.done.append(self.done_list[idx]) 
            pass 
        ## stack  
        out.state = torch.stack(out.state) 
        out.action = torch.stack(out.action) 
        out.reward = torch.stack(out.reward).reshape([-1,1]) 
        out.next_state = torch.stack(out.next_state) 
        out.done = torch.stack(out.done).reshape([-1,1]) 
        return out 
    def clear(self, n=None): 
        'clears first `n` transitions, or all if `n is None`'
        if n is None: 
            n = self.n 
            pass 
        self.state_list = self.state_list[n:] 
        self.action_list = self.action_list[n:] 
        self.reward_list = self.reward_list[n:] 
        self.next_state_list = self.next_state_list[n:] 
        self.done_list = self.done_list[n:] 
        self.n = len(self.state_list) 
        pass 
    pass 

# Define the actor and critic networks 
class SSRAgent(nn.Module): 
    def __init__(self, replay_buffer, ssr_rank=2): 
        super(SSRAgent, self).__init__() 
        self.ssr_rank = ssr_rank 
        self.ssr_low_rank_matrix = None ## =: A 
        self.ssr_residual_diagonal = None ## =: resid 
        self.ssr_center = None 
        self.ssr_prev_center = None 
        self.ssr_n = None 
        ## N * Fisher Information \approx AA^T + resid 
        self.ssr_model_dimension = None 
        self.replay_buffer = replay_buffer 
        pass 
    def loss(self, transitions): 
        raise NotImplementedError('ERROR: loss not implemented!') 
    def memorize(self, n=None): 
        'memorize oldest `n` transitions, or all if `n is None`' 
        if n is None: 
            n = len(self.replay_buffer) 
            pass 
        self.ssr_prev_center = self.ssr_center 
        self.ssr_center = self.__get_param() ## elliptical centroid 
        if self.ssr_model_dimension is None: 
            self.ssr_model_dimension = self.ssr_center.shape[0] 
            pass 
        ssr_low_rank_matrix, ssr_residual_diagonal = l_lanczos(self.__get_get_grad_generator(n), self.ssr_rank, self.ssr_model_dimension, calc_diag=True) 
        if self.ssr_low_rank_matrix is None: 
            ## first memorization 
            self.ssr_low_rank_matrix = ssr_low_rank_matrix 
            self.ssr_residual_diagonal = ssr_residual_diagonal 
            self.ssr_n = n 
        else: 
            ## combine with previous memories 
            self.ssr_low_rank_matrix = combine_krylov_spaces(self.ssr_low_rank_matrix, ssr_low_rank_matrix) 
            self.ssr_residual_diagonal += ssr_residual_diagonal 
            self.ssr_n += n 
            pass 
        pass 
    def ssr(self, lmbda=None): 
        '''Get the ssr regularizer. If `lmbda is None`, `lmbda` will be set to 1 when `self.ssr_prev_center is None`, 
        otherwise `lmbda` will be the approximately optimal `n_A` value.'''
        if self.ssr_low_rank_matrix is None: 
            return 0. 
        if lmbda is None: 
            lmbda = 1. 
            pass 
        p = self.__get_param() 
        p0 = self.ssr_center 
        d = p - p0 
        A = self.ssr_low_rank_matrix 
        res = self.ssr_residual_diagonal 
        dTA = d.transpose(0,1).matmul(A) 
        ATd = dTA.transpose(0,1) 
        dTresd = (d * res).transpose(0,1).matmul(d) 
        ssr_sum = dTA.matmul(ATd) + dTresd 
        if lmbda is None: 
            if self.ssr_prev_center is None: 
                lmbda = 1. 
            else: 
                ## approximately optimal lambda 
                dt = p0 - self.ssr_prev_center 
                dtTA = dt.transpose(0,1).matmul(A) 
                ATdt = dtTA.transpose(0,1) 
                dtTresdt = (dt * res).transpose(0,1).matmul(dt) 
                lmbda = dtTA.matmul(ATdt) + dtTresdt 
                pass 
            pass 
        ## no average, because we're using the log lik  
        ## also, `ssr_n` becomes vague as we iteratively apply optimal lambda 
        return lmbda * .5 * ssr_sum / self.ssr_n  
    def __get_param(self): 
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
                    transition = self.replay_buffer.sample(idx_list=[idx]) 
                    loss = self.loss(transition) 
                    loss.backward() 
                    grad_vec = torch.cat([p.grad.reshape([-1, 1]) for p in self.parameters()], dim=0) 
                    yield grad_vec 
                    pass
                pass 
            return grad_generator 
        return get_grad_generator 
    pass 

