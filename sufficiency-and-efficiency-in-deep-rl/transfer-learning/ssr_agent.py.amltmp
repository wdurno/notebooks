## SSR Agents 
## Approximate sufficient statistics for deep nets 
## Optimally leverage old data as a regression target moves 

import random 
import torch 
import torch.nn as nn 

from lanczos import l_lanczos, combine_krylov_spaces 
from replay_buffer import Object 
from grad_replay_buffer import GradReplayBuffer 

GPU = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
CPU = torch.device('cpu') 

# Define the actor and critic networks 
class SSRAgent(nn.Module): 
    'Abstract SSRAgent class. Define `loss` in concrete subclass.' 
    def __init__(self, replay_buffer, ssr_rank=2, gpu_saver=True, dt_mean_N=10, ssr_info_model=False): 
        '''Initialize core, abstract SSRAgent. 
        args:
         - replay_buffer: instance of the `replay_buffer` class, holds reinforcement learning transitions 
         - ssr_rank: increases Hessian approximation accuracy, but needs [ssr_rank]*[model dim] RAM. Keep it low 
         - gpu_saver: save GPU RAM by moving non-core processing to CPU 
         - dt_mean_N: Statistical manifold traversal is assumed to follow a trended Brownian motion, with stats estimated up to `dt_mean_N` samples. 
         - ssr_info_model: Boolean, True if you want an SSR Agent modelling the information matrix 
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
        self.ssr_info_model = None 
        if ssr_info_model: 
            ## initialize upon obtaining data 
            ## Use buffer to avoid `.parameters()` picking-up sub-parameters 
            ## These are approximately statistically independent models 
            self.ssr_info_model = Object() ## TODO invent serialization scheme 
            self.ssr_info_model.model = None 
            pass 
        self.dt_mean = None 
        self.dt_mean_N = dt_mean_N 
        self.dt_mean_trend = torch.tensor(0.).to(self.device) ## init to 0 heuristically since traversal is continuous from a very stable point 
        self.dt_mean_norm_trend = 0. 
        self.dt_mean_trace_cov = 0.  
        self.dt_prev_pi = .5 
        self.replay_buffer = replay_buffer 
        ## self.optimizer = ... ? ## TODO at least document that this is needed in the subclass 
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
        '''Must be a mean-scaled loss. 
        '''
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
        ## stored in sum-scale; no division by ssr_n 
        ssr_low_rank_matrix, ssr_residual_diagonal = l_lanczos(self.__get_get_grad_generator(n, random_idx=random_idx), self.ssr_rank, self.ssr_model_dimension, calc_diag=True, device=self.device, disable_tqdm=disable_tqdm) 
        ## handle l-Lanczos outputs 
        if self.ssr_low_rank_matrix is None: 
            ## first memorization 
            self.ssr_low_rank_matrix = ssr_low_rank_matrix 
            self.ssr_residual_diagonal = ssr_residual_diagonal 
            self.ssr_n = n 
        elif self.ssr_prev_center is not None and self.ssr_info_model is None: 
            ## combine with previous memories via Krylov subspace 
            self.ssr_low_rank_matrix = combine_krylov_spaces(self.ssr_low_rank_matrix, ssr_low_rank_matrix, device=self.device) 
            self.ssr_residual_diagonal += ssr_residual_diagonal 
            self.ssr_n += n 
        elif self.ssr_prev_center is not None and self.ssr_info_model is None: 
            ## combine with previous memories via SSR info model 
            ## TODO finish 
            if self.ssr_info_model.model is None: 
                ## initialize SSR info model 
                self.ssr_info_model.model = DefaultSSRInfoModel(ssr_rank=self.ssr_rank, \
                    grad_replay_buffer = GradReplayBuffer(self), \
                    initial_L = self.ssr_low_rank_matrix, \
                    initial_D_vec = self.ssr_residual_diagonal, \
                    intial_n = self.ssr_n \
                    ) 
                pass 
            ## update SSR info 
            self.ssr_info_model.model.fit(batch_size=1, iters=len(self.replay_buffer)) 
            self.ssr_info_model.model.memorize() 
            self.ssr_low_rank_matrix = self.ssr_info_model.model.L.detach() 
            self.ssr_residual_diagonal = torch.exp(self.ssr_info_model.model.log_D.detach()) 
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
        self.optimizer.zero_grad() 
        ssr = self.ssr() 
        for _ in range(iters): 
            data = self.replay_buffer.sample(batch_size=batch_size) ## TODO provide whole-sample-in-batches option 
            loss = self.loss(data) / iters 
            loss = pi * loss + (1 - pi) * self.ssr() 
            loss.backward() 
            pass 
        self.optimizer.step() 
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
                    grad_vec = self.get_grad_vec(idx) 
                    yield grad_vec 
                    pass
                pass 
            return grad_generator 
        return get_grad_generator 
    
    def get_grad_vec(self, observation_index): 
        self.zero_grad() 
        transition = self.replay_buffer.sample(idx_list=[observation_index]) 
        loss = self.loss(transition) 
        loss.backward() 
        for p in self.parameters(): 
            if p.grad is None: 
                p.grad = torch.zeros(size=p.shape) 
            pass 
        grad_vec = torch.cat([p.grad.reshape([-1, 1]) for p in self.parameters()], dim=0).clone().detach() 
        return grad_vec 
    pass 

class DefaultSSRInfoModel(SSRAgent): 
    '''A sufficient statistic regularized model for estimating information matrices (equivalently, covariances matrices) without regressors
    '''
    def __init__(self, ssr_rank, grad_replay_buffer, initial_L, initial_D_vec, initial_n): 
        '''Constructs a deep learning model estimating information (covariance) matrices. 
        Utilizes low rank approximation LL^T + D, where L is p X r-shaped and D is diagonal, positive. 
        args: 
        - ssr_rank: positive integer, sets r. Warning: applying this model to an O(p*r)-sized SSR Agent model will increase space requirements to O(p*r^2)! 
        - grad_replay_buffer: Must be of type GradReplayBuffer initialized with the parent SSR Agent's replay buffer. 
        - initial_L: p X r-shaped tensor providing an initial estimate for L. 
        - initial_D: p X 1-shaped tensor providing an initial estimate for the diagonal of D. 
        - initial_n: integer providing initial sample size used to estimate initial parameters. 
        '''
        super(DefaultSSRInfoModel, self).__init__(ssr_rank=ssr_rank, replay_buffer=grad_replay_buffer) 
        self.L = torch.nn.parameter(initial_L) 
        initial_D_vec[initial_D_vec < 1e-5] = 1e-5 ## don't want to store any negative inifinities 
        self.log_D = torch.nn.parameter(torch.log(initial_D_vec)) ## store on log scale so the optimizer can utilize all Euclidian space 
        self.ssr_n = initial_n 
        self.optimizer = optim.Adam(self.parameters(), lr=0.001) ## TODO do not hard code - this whole init scheme needs a refactor 
        pass 
    def forward(self, transitions): 
        '''Calculates \| LL^T + D - YY^T \|_F^2 with a few \|A + B \|_F^2 = \|A \|_F^2 + \|B \|_F^2 - 2 tr[ AB ] expansions. 
        If this was actually used for forecasting, the forecast would just be LL^T + D, but that's just an auto-OOM in deep learning. 
        So, instead, I'm calculating a sum-scale loss, ready for batching. 
        '''
        y = transitions.y ## p X n tensor of gradients ## TODO refactor so rows are n 
        # loss = ((self.L.T @ self.L) * (self.L.T @ self.L)).sum() \
        #     + (torch.exp(self.log_D) * torch.exp(self.log_D)).sum() \
        #     - 2. * torch.trace(self.L.T @ torch.exp(self.log_D) @ self.L) \
        #     + ((y.T @ T) * (y.T @ T)).sum() \
        #     + 2. * ((y.T @ self.L) * (y.T @ self.L)).sum() \
        #     + 2. * torch.trace(y.T @ torch.exp(self.log_D) @ y)  
        d = self.log_D.exp().reshape([-1])  # (p,)

        G = self.L.T @ self.L                 # (r, r)
        term_LL = (G * G).sum()               # ||LL^T||_F^2 = ||L^T L||_F^2

        term_D = (d * d).sum()                # ||D||_F^2 (D diagonal)

        rowL2 = (self.L * self.L).sum(dim=1)  # (p,) rowwise ||L_i||^2
        term_cross_LD = 2.0 * (d * rowL2).sum()

        YY = y.T @ y                          # (m, m)
        term_YY = (YY * YY).sum()             # ||YY^T||_F^2 = ||Y^T Y||_F^2

        YL = y.T @ self.L                     # (m, r)
        term_cross_LY = -2.0 * (YL * YL).sum()  # -2 ||Y^T L||_F^2

        rowY2 = (y * y).sum(dim=1)            # (p,) rowwise ||Y_i||^2
        term_cross_DY = -2.0 * (d * rowY2).sum()

        loss = term_LL + term_D + term_cross_LD + term_YY + term_cross_LY + term_cross_DY
        return loss 
    def loss(self, transitions): 
        # n = len(self.replay_buffer) 
        # loss = 0. 
        # ## loop over batches, calling `.backward` per iteration to free execution graphs from memory 
        # for idx_batch in (list(range(n))[i:i+batch_size] for i in range(0, n, batch_size)): 
        #     batch_loss = self.forward(self.replay_buffer.sample(idx_list=idx_batch)) * (len(idx_batch) / n) 
        #     batch_loss.backward() ## avoid those OOMs! 
        #     loss += batch_loss 
        #     pass 
        loss = self.forward(transitions) 
        return loss  
    pass 








