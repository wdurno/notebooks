## FSDP SSR Module  
## Approximate sufficient statistics for deep nets 
## Optimally leverage old data as a regression target moves 
## FSDP adjustments distribute the model over several GPUs 
## However, the SSR is caclculated in CPU RAM 
## So, this isn't very scalable but decent for demonstrations 

import random 
import os 
import math 
import torch 
import torch.nn as nn 
import torch.distributed as dist 
from torch.utils.data import Subset 
from torch.utils.data import DataLoader 
from torch.utils.data.distributed import DistributedSampler 
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP 
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy 
from lanczos import l_lanczos, combine_krylov_spaces, distributed_multiply_fisher 

class AbstractFsdpSsrModule(nn.Module): 
    'Abstract FSDP SSR Module class. Define `loss` and `optimizer` in concrete subclass.' 
    def __init__(self, module, replay_buffer, ssr_rank=2, dt_mean_N=10, use_fsdp=True): 
        '''Initialize core, abstract FSDP SSR Module. 
        args: 
        - module: the module to be FSDP-wrapped storing all differentiable parameters. 
        - replay_buffer: instance of the `replay_buffer` class, holds reinforcement learning transitions. Data must be loaded to all ranks! 
        - ssr_rank: increases Hessian approximation accuracy, but needs [ssr_rank]*[model dim] RAM. Keep it low. 
        - dt_mean_N: Statistical manifold traversal is assumed to follow a trended Brownian motion, with stats estimated up to `dt_mean_N` samples. 
        - use_fsdp: if true, the model is FSDP-distributed. Otherwise, a full copy is loaded per GPU. 
        '''
        super(AbstractFsdpSsrModule, self).__init__() 
        device_id = torch.device(f"cuda:{dist.get_rank()}") 
        self.module = module.to(device_id) 
        if use_fsdp: 
            self.module = FSDP(self.module, auto_wrap_policy=size_based_auto_wrap_policy) 
            n_fsdp = len([m for m in self.module.modules() if isinstance(m, FSDP)]) 
            print(f'DEBUG: num FSDP modules detected: {n_fsdp}') 
            pass 
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
            torch.save(self.ssr_dict(), os.path.join(path, 'full-model.ssr.pt'))  
            pass 
        # def _save_state(): 
        #     torch.save(self.module.state_dict(), path + 'full-model.state.pt') 
        #     pass 
        if dist.get_rank() == 0:
            with FSDP.summon_full_params(self.module, offload_to_cpu=True, rank0_only=True, writeback=False): 
                torch.save(self.module.state_dict(), os.path.join(path, 'full-model.state.pt')) 
                pass 
            pass 
        pass 
    def load(self, path, module=None): 
        'Loads from disk to CPU RAM, then distributes parameters over the cluster' 
        state_path_ssr = os.path.join(path, 'full-model.ssr.pt') 
        state_path_model = os.path.join(path, 'full-model.state.pt') 
        if dist.get_rank() == 0: 
            print(f'Loading model from {path}...') 
            ## TODO single node load times for the SSR are rediculous. 
            ## This is a strong motivator for fully distributed algorithms. 
            self.load_ssr_dict(torch.load(state_path_ssr, map_location="cpu")) 
            pass 
        ## load entire module, then apply FSDP to discard unneeded parameters 
        if module is None: 
            module = self.module 
            pass 
        state_dict = torch.load(state_path_model) 
        module.load_state_dict(state_dict) 
        ## TODO remove following old code 
        ### TODO only use an FSDP block if `use_fsdp=True` 
        ### TODO ideally, loading occurs before FSDP is called on a module thereby saving this wasted communication 
        ### I cannot load shards because I cannot save shards because I must load without FSDP to update the SSR 
        #with FSDP.summon_full_params(self.module, offload_to_cpu=True, writeback=True): 
        #    state_dict = torch.load(state_path_model, map_location="cpu") 
        #    self.module.load_state_dict(state_dict) 
        #    pass 
        ### Does not resdistribute parameters! FSDP only sends shards owned by current rank 
        ##def _load_state(): 
        ##    self.load_state_dict(torch.load(os.path.join(path, 'full-model.state.pt'), map_location="cpu")) 
        ##    pass 
        ##with FSDP.summon_full_params(self.module, rank0_only=True, offload_to_cpu=True): ## redistributes on context close 
        ##    AbstractFsdpSsrModule.__rank_0_run(_load_state) 
        ##    pass 
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
        print(f'DEBUG 16: memorizing...')
        if dist.get_rank() == 0: 
            if n is None: 
                n = len(self.replay_buffer) 
                pass 
            ## track current and prev estimates 
            self.ssr_prev_center = self.ssr_center.to(self.gpu_saver) if self.ssr_center is not None else None  
            self.ssr_center = self.get_param().clone().detach() ## elliptical centroid 
            pass 
        ## get model dim if we don't already have it 
        if self.ssr_model_dimension is None: 
            if dist.get_rank() == 0: 
                ## get actual value 
                self.ssr_model_dimension = torch.tensor(self.ssr_center.shape[0], dtype=torch.long)
            else: 
                ## prepare space to recieve 
                self.ssr_model_dimension = torch.tensor(0, dtype=torch.long) 
                pass 
            ## distribute 
            dist.broadcast(self.ssr_model_dimension, src=0) 
            ## recast to basic type 
            self.ssr_model_dimension = self.ssr_model_dimension.item() 
            pass 
        if dist.get_rank() == 0: 
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
            pass 
        dist.barrier() 
        print(f'DEBUG 17: distributed l_lanczos...') 
        ssr_low_rank_matrix, ssr_residual_diagonal = l_lanczos(self.__get_get_grad_generator(n, random_idx=random_idx), self.ssr_rank, self.ssr_model_dimension, calc_diag=True, device=torch.device('cpu'), disable_tqdm=disable_tqdm, mfi_alternate=distributed_multiply_fisher) 
        print(f'DEBUG 18: integrating information...')
        if dist.get_rank() == 0:
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
            print(f'DEBUG 19: memorization complete...') 
        dist.barrier() 
        pass 
    def ssr(self, lmbda=None): 
        '''Get the ssr regularizer and its gradient vector. 
        If `lmbda is None`, `lmbda` will be set to 1 when `self.ssr_prev_center is None`, 
        otherwise `lmbda` will be the approximately optimal `n_A` value.
        
        WARNING: Only run inside a `with FSDP.summon_full_params(model, offload_to_cpu=True, rank0_only=True):` block!
        The block is not enforced here to enable gradient handling.'''
        print(f'DEBUG SSR 1: type(self.ssr_low_rank_matrix) {type(self.ssr_low_rank_matrix)}')
        if self.ssr_low_rank_matrix is None: 
            return 0., 0.
        ## p = self.get_param() 
        p = self.get_param().detach() ## FSDP modification: break the graph and calculate gradients manually 
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
    def fit(self, batch_size, iters=1, pi_min=.1, pi_max=.9, subset_size=-1, pg_gloo=None): 
        '''Runs numerical fitting iterations over the `replay_buffer` dataset. 
        inputs: 
        - `batch_size`: the per-rank batch size. 
        - `iters`: the number of iterations, each updating the parameter and pulling from new random data batch. 
        - `subset_size`: if > 0, subset the replay_buffer randomly. Default -1. 
        - `pg_gloo`: process group running GLOO protocol 
        outputs: 
        - `pi` (float): the `pi` estimate used for this round of fits. 
        - `loss` (float): the final observed `loss` after all fitting iterations. 
        side-effects: 
        - `self.module.parameter` is updated where `requires_grad=True`.'''
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
        dist.barrier() 
        ## broadcast pi via GPU since NCCL requires it 
        pi = pi.cuda() 
        dist.broadcast(pi, src=0) 
        pi = pi.cpu() 
        ## New loaders & samplers are needed because overall dataset size frequently changes in RL 
        loader, sampler = self.__get_distributed_loader_and_sampler(batch_size, subset_size=subset_size) 
        ## start fit iterations 
        for epoch_idx in range(iters): 
            n = 0 
            sampler.set_epoch(epoch_idx) 
            ## zeroing grads with Nones because 
            ## 1. zeros and Nones are applied to _all_ parameters, and 
            ## 2. FSDP only needs zeros applied to some parameters per GPU. 
            self.optimizer.zero_grad(set_to_none=True) 
            for data_idx, data in enumerate(loader): ## TODO FSDP is not designed to accommodate this kind of loop, so has biased MLEs 
                print(f'DEBUG 11: type(data): {type(data)}, type(data[0]): {type(data[0])}') 
                data = AbstractFsdpSsrModule.__load_stacker(data) 
                n += data[0].shape[0] ## loader stacks samples tuples of tensors into a tuple of stacked tensors 
                print(f'DEBUG 2: calculating loss...') 
                loss = self.loss(data) 
                print(torch.cuda.memory_summary()) ## MORE DEBUGGING 
                print(f'DEBUG 6: calculating backward... data_idx: {data_idx}, epoch_idx: {epoch_idx}') 
                loss.backward() ## FSDP aggregates grads over ranks with an allreduce OP SUM 
                #self.optimizer.zero_grad(set_to_none=True) ## TODO REMOVE !!! THIS IS AN OOM TEST 
                ##torch.cuda.empty_cache() ## THIS DIDN'T WORK 
                pass 
            ## get actual sample size so we can adjust ssr_grad scale to fit summed gradients 
            ## I can't just use dataset size because distributed loaders are capable of small degrees of double sampling 
            n = torch.tensor(n, dtype=torch.long, device=torch.cuda.current_device()) 
            dist.all_reduce(n, op=dist.ReduceOp.SUM) 
            n = n.item() 
            ## calculate the average SSR and its gradient on rank 0's CPU 
            ## I'll adjust upward by `n` because `loss` isn't averaged 
            print(f'DEBUG 6.5 starting pre-ssr parameter transfer...') 
            ssr_grad = None 
            with FSDP.summon_full_params(self.module, offload_to_cpu=True, writeback=False, rank0_only=True): 
                if dist.get_rank() == 0: 
                    print('DEBUG 7: calculating ssr...') 
                    ssr, ssr_grad = self.ssr() 
                    ssr *= n 
                    ssr_grad *= n 
                    pass 
                dist.barrier() 
                pass 
            ## communicate ssr 
            if dist.get_rank() == 0: 
                if type(ssr) == float: 
                    ssr = torch.tensor(ssr).reshape([1,1]) 
            else: 
                ssr = torch.tensor(0.).reshape([1,1]) 
            dist.broadcast(ssr, src=0, group=pg_gloo) 
            dist.barrier(group=pg_gloo) 
            print('DEBUG 8: adjusting grads...') 
            self.__adjust_grads(ssr_grad, pi, pg_gloo) ## grads applied here 
            print('DEBUG 9: applying grads...') 
            self.optimizer.step() ## apply gradients 
            self.optimizer.zero_grad(set_to_none=True) 
            ## return average loss for reporting purposes because it'll be comparable over different samples sizes 
            loss = (pi * loss + (1 - pi) * ssr)/n 
            pass 
        return float(pi), float(loss) 
    def __get_get_grad_generator(self, n=None, random_idx=False): 
        ## The double get hides `self` in a function context, 
        ## packaging `get_grad_generator` for calling without 
        ## the AbstractFsdpSsrModule instance. 
        if n is None: 
            n = len(self.replay_buffer) 
            pass 
        if n > len(self.replay_buffer): 
            n = len(self.replay_buffer) 
            pass 
        def get_grad_generator(distributed=False): 
            'l-Lanczos alg uses grad at least `ssr_rank` times' 
            def grad_generator(distributed=distributed): 
                if not distributed: 
                    iterator = range(n)
                else: 
                    def padded_rank_iterator(n: int):
                        'always returns ceil(n/world) elements, padded with Nones if necessary'
                        rank = dist.get_rank()
                        world = dist.get_world_size()
                        # Generate the rank-specific indices
                        indices = list(range(rank, n, world))
                        max_len = math.ceil(n / world)  # longest slice length
                        # Pad with None if this rank's slice is shorter
                        indices += [None] * (max_len - len(indices))
                        return iter(indices)
                    iterator = padded_rank_iterator(n)
                    pass 
                for idx in iterator: 
                    if random_idx: 
                        idx = random.randint(0, len(self.replay_buffer)-1) 
                        pass 
                    self.zero_grad() 
                    #transition = self.replay_buffer.sample(idx_list=[idx]) ## TODO remove / fix: sample no-longer defined 
                    transition = [[t] for t in self.replay_buffer[idx]]
                    transition = AbstractFsdpSsrModule.__load_stacker(transition) 
                    transition = [t.cuda() for t in transition] 
                    loss = self.loss(transition) 
                    loss.backward() 
                    for p in self.module.parameters(): 
                        if p.requires_grad:
                            if p.grad is None: 
                                p.grad = torch.zeros_like(p) 
                                pass
                            pass
                        pass 
                    ## early .cpu() call keeps copy off-of GPU 
                    grad_vec = torch.cat([p.grad.cpu().reshape([-1, 1]) for p in self.module.parameters() if p.requires_grad], dim=0).detach()  
                    yield grad_vec 
                    pass 
                pass 
            return grad_generator 
        return get_grad_generator 
    # def __adjust_grads(self, ssr_grad, pi): 
    #     'applies the ssr gradient over all FSDP GPU gradients' 
    #     cursor = 0 
    #     for p in [p for p in self.module.parameters() if p.requires_grad]: 
    #         n = p.numel() 
    #         device = torch.device("cuda", torch.cuda.current_device()) 
    #         communication_tensor = torch.zeros([n], device=device, dtype=torch.float32) 
    #         if dist.get_rank() == 0: 
    #             communication_tensor.copy_(ssr_grad[cursor : cursor + n, 0]) 
    #             pass 
    #         ## TODO Replace this terribly inefficient code, ideally with FSDP-native gradient calculation. 
    #         ## This should be paired with refactoring my `l_lanczos` eigenvector algorithm to FSDP as well. 
    #         ## Distributed numerical engineering takes time, expertise, and care, hence the inefficient-but-effective alternative here. 
    #         dist.broadcast(communication_tensor, src=0) 
    #         if p._fsdp_shard_metadata is not None: ## TODO pick-up here, _fsdp_shard_metadata now always defined 
    #             ## current rank owns this parameter 
    #             p.grad = (pi * p.grad) + ((1 - pi) * communication_tensor)  
    #             pass 
    #         cursor += n 
    #         ## free GPU memory 
    #         del communication_tensor 
    #         pass 
    #     pass 
    def __adjust_grads(self, ssr_grad: torch.Tensor, pi: float, pg_gloo):
        """
        Apply global SSR gradient to parameter grads.
        `ssr_grad` is the full concatenated vector on rank 0; dtype float32.
        """
        ## distribute ssr_grad over CPUs 
        p = torch.tensor(1) 
        if type(ssr_grad) == float: 
            ssr_grad = torch.tensor(ssr_grad).reshape([1,1]) ## degenerate case, communicate break 
        if dist.get_rank() == 0: 
            p = torch.tensor(ssr_grad.shape[0]) 
        dist.broadcast(p, src=0, group=pg_gloo) ## send grad size 
        if dist.get_rank() != 0: 
            ssr_grad = torch.zeros([p,1], dtype=torch.float32) 
        dist.broadcast(ssr_grad, src=0, group=pg_gloo) ## enables reads per rank 
        dist.barrier(group=pg_gloo) 
        if p == 1:
            ## distributed break 
            return 
        ## update grads 
        ## This requires loading a whole model into the GPU with grads -- it likely won't fit and will force me to reprogram for full distribution 
        cursor = 0
        with FSDP.summon_full_params(
            self.module, 
            rank0_only=False,            # supported
            with_grads=True,             # gather grads
            offload_to_cpu=False         # gather to GPU
            ):
            for p in (p for p in self.module.parameters() if p.requires_grad):
                n = p.numel()
                # slice & move to param device/dtype
                sl = ssr_grad[cursor:cursor+n,0].to(p.device, p.dtype, non_blocking=True).view_as(p)
                cursor += n
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                with torch.no_grad():
                    p.grad.mul_(pi).add_(sl, alpha=1.0 - pi)
                del sl 
    def __get_distributed_loader_and_sampler(self, batch_size, subset_size=-1):
        dataset = self.replay_buffer 
        if subset_size > 0: 
            dataset = self.random_dataset_subset(subset_size)
            pass 
        sampler = DistributedSampler(
            dataset,
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
    def random_dataset_subset(self, size):
        #indices = random.sample(range(len(dataset)), size) 
        indices = self.replay_buffer.biased_subsample(size)  
        return Subset(self.replay_buffer, indices)
    @staticmethod 
    def __rank_0_run(f): 
        'run f on rank 0 while blocking the rest of the cluster'
        if dist.get_rank() == 0: 
            f() 
            pass 
        dist.barrier() 
        pass 
    @staticmethod
    def __load_stacker(tensor_tuple_list): 
        '''`DataLoader` batches arrive in a list of results and inconsistent tensor ranks. 
        This function corrects that there's only a single RL tuple of tensors, 
        and each tensor has a sample index.
        inputs:
        - tensor_tuple_list: a list of tuples of tensors 
        outputs:
        - tensor_tuple: a tuple of tensors 
        ''' 
        out = [] 
        for tensor_tuple in tensor_tuple_list: 
            out.append(torch.stack([t for t in tensor_tuple])) 
            pass 
        return out 
    @staticmethod
    def __add_sample_idx_if_missing(tensor_tuple): 
        'helper to __load_stacker'
        ## check first tensor for adequate rank 
        ## need shape with len 2 or more, like [n, 2048] 
        out = [] 
        for tensor in tensor_tuple: 
            rank = len(tensor.shape) 
            while rank < 2: 
                ## inadequate rank indicates row vector or scalar 
                tensor = tensor.unsqueeze(0) 
                rank = len(tensor.shape) 
                pass 
            out.append(tensor) ## adds first index so [...] becomes [1, ...] 
            pass 
        return out 
    pass 

