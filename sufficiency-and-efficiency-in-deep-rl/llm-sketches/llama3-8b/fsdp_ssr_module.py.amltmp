## FSDP SSR Module  
## Approximate sufficient statistics for deep nets 
## Optimally leverage old data as a regression target moves 
## FSDP adjustments distribute the model over several GPUs 
## However, the SSR is caclculated in CPU RAM 
## So, this isn't very scalable but decent for demonstrations 

import random 
import os 
import torch 
import torch.nn as nn 
import torch.distributed as dist 
from torch.utils.data import DataLoader 
from torch.utils.data.distributed import DistributedSampler 
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP 
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy 
from lanczos import l_lanczos, combine_krylov_spaces 

class AbstractFsdpSsrModule(nn.Module): 
    'Abstract FSDP SSR Module class. Define `loss` and `optimizer` in concrete subclass.' 
    def __init__(self, module, replay_buffer, ssr_rank=2, dt_mean_N=10): 
        '''Initialize core, abstract FSDP SSR Module. 
        args: 
        - module: the module to be FSDP-wrapped storing all differentiable parameters. 
        - replay_buffer: instance of the `replay_buffer` class, holds reinforcement learning transitions. Data must be loaded to all ranks! 
        - ssr_rank: increases Hessian approximation accuracy, but needs [ssr_rank]*[model dim] RAM. Keep it low. 
        - dt_mean_N: Statistical manifold traversal is assumed to follow a trended Brownian motion, with stats estimated up to `dt_mean_N` samples. 
        '''
        super(AbstractFsdpSsrModule, self).__init__() 
        device_id = torch.device(f"cuda:{dist.get_rank()}") 
        module = module.to(device_id) 
        self.module = FSDP(module, auto_wrap_policy=size_based_auto_wrap_policy) 
        ## store params 
        self.ssr_rank = ssr_rank 
        self.ssr_low_rank_matrix = None ## =: A 
        self.ssr_residual_diagonal = None ## =: resid 
        ## N * Fisher Information \approx AA^T + resid 
        self.ssr_center = None ## yes, I have several copies - results before optimization 
        self.ssr_prev_center = None 
        self.ssr_n = None 
        self.ssr_cov_trace = None 
        self.ssr_cov_n = None 
        self.ssr_model_dimension = None 
        self.dt_mean = None 
        self.dt_mean_N = dt_mean_N 
        self.dt_mean_trend = torch.tensor(0.) ## init to 0 heuristically since traversal is continuous from a very stable point 
        self.dt_mean_norm_trend = 0. 
        self.dt_mean_trace_cov = 0. 
        self.dt_prev_pi = .5 
        self.replay_buffer = replay_buffer 
        self.optimizer = None ## abstract attribute; must optimize over `self.module` parameters 
        pass 
    def ssr_dict(self): 
        d = {'ssr_rank': self.ssr_rank, 
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
        self.ssr_rank = d['ssr_rank'] 
        self.ssr_low_rank_matrix = d['ssr_low_rank_matrix'] 
        self.ssr_residual_diagonal = d['ssr_residual_diagonal'] 
        self.ssr_center = d['ssr_center'] 
        self.ssr_prev_center = d['ssr_prev_center']  
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
        'pulls parameters from ranks to CPU RAM and writes to disk' 
        if dist.get_rank() == 0: 
            print(f'Saving model at {path}...') 
            torch.save(self.ssr_dict(), path + 'full-model.ssr.pt')  
            pass 
        def _save_state(): 
            torch.save(self.module.state_dict(), path + 'full-model.state.pt') 
            pass 
        with FSDP.summon_full_params(self.module, offload_to_cpu=True, rank0_only=True, writeback=False): 
            AbstractFsdpSsrModule.__rank_0_run(_save_state) 
            pass 
        pass 
    def load(self, path): 
        'Loads from disk to CPU RAM, then distributes parameters over the cluster' 
        state_path_ssr = os.path.join(path, 'full-model.ssr.pt') 
        state_path_model = os.path.join(path, 'full-model.state.pt') 
        if dist.get_rank() == 0: 
            print(f'Loading model from {path}...') 
            self.load_ssr_dict(torch.load(state_path_ssr, map_location="cpu")) 
            pass 
        with FSDP.summon_full_params(self.module, offload_to_cpu=True): 
            state_dict = torch.load(state_path_model, map_location="cpu") 
            self.module.load_state_dict(state_dict) 
            pass 
        ## Does not resdistribute parameters! FSDP only sends shards owned by current rank 
        #def _load_state():  
        #    self.load_state_dict(torch.load(os.path.join(path, 'full-model.state.pt'), map_location="cpu")) 
        #    pass 
        #with FSDP.summon_full_params(self.module, rank0_only=True, offload_to_cpu=True): ## redistributes on context close 
        #    AbstractFsdpSsrModule.__rank_0_run(_load_state) 
        #    pass 
        pass 
    def loss(self, transitions): 
        '''Abstract function which you must implement. 
        FSDP constraints: 
        - Sum losses; do not average. FSDP all-reduces gradients with a sum operation - so, avoid summing averages. 
        - Be sure to cover empty `len(transitions) == 0` cases, perhaps by returning `tensor(0.)`. 
        '''
        raise NotImplementedError('ERROR: loss not implemented!') 
    def memorize(self, n=None, random_idx=False, disable_tqdm=False): 
        'memorize oldest `n` transitions, or all if `n is None`' 
        def _memorize(): 
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
        with FSDP.summon_full_params(model, rank0_only=True, offload_to_cpu=True):
            AbstractFsdpSsrModule.__rank_0_run(_memorize) 
            pass 
        pass 
    def ssr(self, lmbda=None): 
        '''Get the ssr regularizer and its gradient vector. 
        If `lmbda is None`, `lmbda` will be set to 1 when `self.ssr_prev_center is None`, 
        otherwise `lmbda` will be the approximately optimal `n_A` value.
        
        WARNING: Only run inside a `with FSDP.summon_full_params(model, offload_to_cpu=True, rank0_only=True):` block!
        The block is not enforced here to enable gradient handling.'''
        if self.ssr_low_rank_matrix is None: 
            return 0. 
        ## p = self.get_param() 
        p = self.get_param().clone().detach() ## FSDP modification: break the graph and calculate gradients manually 
        p0 = self.ssr_center 
        d = p - p0 
        A = self.ssr_low_rank_matrix 
        res = self.ssr_residual_diagonal 
        dTA = d.transpose(0,1).matmul(A) 
        ATd = dTA.transpose(0,1) 
        dTresd = (d * res).transpose(0,1).matmul(d) 
        ssr_sum = dTA.matmul(ATd) + dTresd 
        ssr_mean = ssr_sum / self.ssr_n 
        ssr_grad_sum = A.matmul(ATd) + (d * res).reshape([-1, 1]) 
        ssr_grad_mean = ssr_grad_sum / self.ssr_n 
        ## I've opted against extending autograd.Function because FSDP expects the whole graph to be on the GPUs. 
        ## My CPU-bound code is proven and I want results now. 
        return .5 * ssr_mean , ssr_grad_mean 
    def optimal_pi(self, pi_min=0., pi_max=1., return_lambda=False): 
        "Returns a linear approximation of pi's optimal value. This is a CPU-bound calculation." 
        if self.dt_mean_norm_trend == 0.: 
            pi = torch.tensor(.5) 
        else: 
            pi = 1. - .5 * self.dt_mean_trace_cov / self.dt_mean_norm_trend 
            pi = pi.clone().detach().reshape([]) 
            pass 
        if float(pi) < pi_min: 
            pi = torch.tensor(pi_min) 
        if float(pi) > pi_max: 
            pi = torch.tensor(pi_max) 
            pass 
        if return_lambda: 
            lmbda = self.ssr_n * (1. - pi) ## lambda = n_A 
            return lmbda 
        ## returning pi, probability of sampling with theta_B 
        return pi 
    def get_param(self): 
        '''Only for SSR calculations. 
        
        WARNING: Only run inside a `with FSDP.summon_full_params(model, offload_to_cpu=True, rank0_only=True):` block! 
        The block is not enforced here to enable gradient handling.''' 
        ## For an FSDP module, all self.module parameters will be `FlatParameter`s 
        return torch.cat([p.reshape([-1, 1]) for p in self.module.parameters() if p.requires_grad], dim=0) 
    def fit(self, batch_size, iters=1, pi_min=.1, pi_max=.9): 
        '''Runs numerical fitting iterations over the `replay_buffer` dataset. 
        inputs: 
        - `batch_size`: the per-rank batch size. 
        - `iters`: the number of iterations, each updating the parameter and pulling from new random data batch. 
        outputs: 
        - `pi` (float): the `pi` estimate used for this round of fits. 
        - `loss` (float): the final observed `loss` after all fitting iterations. 
        side-effects: 
        - `self.model.parameter` is updated where `requires_grad=True`.'''
        ## check for sufficient concreteness 
        if self.optimizer is None: 
            raise NotImplementedError('ERROR: optimizer not implemented!') 
        self.train() 
        ## distribute pi from rank 0 CPU, since that's where SSR stats are stored 
        pi = torch.tensor(0.) 
        if dist.get_rank() == 0: 
            ## store `dt_prev_pi` for reporting purposes 
            self.dt_prev_pi = pi = self.optimal_pi(pi_min=pi_min, pi_max=pi_max) 
            pass 
        ## broadcast pi via GPU since NCCL requires it 
        pi = pi.cuda() 
        dist.broadcast(pi, src=0) 
        pi = pi.cpu() 
        ## New loaders & samplers are needed because overall dataset size frequently changes in RL 
        loader, sampler = self.__get_distributed_loader_and_sampler(batch_size) 
        ## start fit iterations  
        for epoch_idx in range(iters): 
            n = 0 
            sampler.set_epoch(epoch_idx) 
            self.optimizer.zero_grad() 
            for data in loader: 
                n += data[0].shape[0] ## loader stacks samples tuples of tensors into a tuple of stacked tensors 
                loss = self.loss(data) 
                loss.backward() ## FSDP aggregates grads over ranks with an allreduce OP SUM 
                pass 
            ## get actual sample size so we can adjust ssr_grad scale to fit summed gradients 
            ## I can't just use dataset size because distributed loaders are capable of small degrees of double sampling 
            n = torch.tensor(n) 
            dist.all_reduce(n, op=dist.ReduceOp.SUM) 
            n = n.item() 
            ## calculate the average SSR and its gradient on rank 0's CPU 
            ## I'll adjust upward by `n` because `loss` isn't averaged 
            with FSDP.summon_full_params(model, offload_to_cpu=True, rank0_only=True): 
                if dist.get_rank() == 0: 
                    ssr, ssr_grad = self.ssr() 
                    ssr *= n 
                    ssr_grad *= n 
                    pass 
                dist.barrier() 
                pass 
            self.__adjust_grads(ssr_grad, pi) ## grads applied here 
            self.optimizer.step() ## apply gradients 
            ## return average loss for reporting purposes because it'll be comparable over different samples sizes 
            loss = (pi * loss + (1 - pi) * ssr)/n 
            pass 
        return float(pi), float(loss) 
    def __get_get_grad_generator(self, n=None, random_idx=False): 
        ## The double get hides `self` in a function context, 
        ## packaging `get_grad_generator` for calling without 
        ## the AbstractFsdpSsrModule instance. 
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
                    grad_vec = torch.cat([p.grad.reshape([-1, 1]) for p in self.module.parameters() if p.requires_grad], dim=0).clone().detach()  
                    yield grad_vec 
                    pass 
                pass 
            return grad_generator 
        return get_grad_generator 
    def __adjust_grads(self, ssr_grad, pi): 
        'applies the ssr gradient over all FSDP GPU gradients' 
        cursor = 0 
        for p in [p for p in self.module.parameters() if p.requires_grad]: 
            n = p.numel() 
            communication_tensor = torch.zeros([n]) 
            if dist.get_rank() == 0: 
                communication_tensor.copy_(ssr_grad[cursor : cursor + n, 1]) 
                pass 
            ## TODO Replace this terribly inefficient code, ideally with FSDP-native gradient calculation. 
            ## This should be paired with refactoring my `l_lanczos` eigenvector algorithm to FSDP as well. 
            ## Distributed numerical engineering takes time, expertise, and care, hence the inefficient-but-effective alternative here. 
            dist.broadcast(communication_tensor, src=0) 
            if p._fsdp_shard_metadata is not None: 
                ## current rank owns this parameter 
                p.grad = (pi * p.grad) + ((1 - pi) * communication_tensor.to(p.device))  
                pass 
            cursor += n 
            pass 
        pass 
    def __get_distributed_loader_and_sampler(self, batch_size):
        sampler = DistributedSampler(
            self.replay_buffer,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True, 
            drop_last=False,
        )
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True
        )
        return dataloader, sampler
    @staticmethod 
    def __rank_0_run(f): 
        'run f on rank 0 while blocking the rest of the cluster'
        if dist.get_rank() == 0: 
            f() 
            pass 
        dist.barrier() 
        pass 
    pass 

