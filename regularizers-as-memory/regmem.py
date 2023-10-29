## Define model 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random 
import gym 
from tqdm import tqdm 
import numpy as np 
import matplotlib.pyplot as plt 
import copy 
from lanczos import lanczos 

INPUT_DIM = 4
N_ACTIONS = 2
MAX_SAMPLE = 100000
DISCOUNT = .95 
EPS = 1e-5
EXPLORE_PROBABILITY_FUNC = lambda idx: 0.99**idx 
BATCH_SIZE = 30 
LEARNING_RATE = 0.001 
GRAD_CLIP = 10.0 
SHORT_TERM_MEMORY_LENGTH = 3 
LBFGS = False 
ENV_NAME = 'CartPole-v1' 

class Model(nn.Module): 
    def __init__(self, 
            input_dim=INPUT_DIM, 
            n_actions=N_ACTIONS, 
            max_sample=MAX_SAMPLE, 
            discount = DISCOUNT, 
            eps=EPS, 
            explore_probability_func=EXPLORE_PROBABILITY_FUNC, 
            batch_size=BATCH_SIZE, 
            learning_rate=LEARNING_RATE, 
            grad_clip=GRAD_CLIP, 
            short_term_memory_length=SHORT_TERM_MEMORY_LENGTH, 
            lbfgs=LBFGS, 
            env_name=ENV_NAME,
            hessian_sum=None, 
            hessian_sum_low_rank_half=None, 
            hessian_rank=None, 
            hessian_denominator=None, 
            hessian_center=None, 
            observations=None,
            total_iters=0,
            info_prop_regularizer=None, 
            regularizing_lambda_function=None, 
            eta_space=None): 
        super(Model, self).__init__() 
        ## store config 
        self.input_dim = input_dim 
        self.n_actions = n_actions 
        self.max_sample = max_sample 
        self.discount = discount 
        self.eps = eps 
        self.explore_probability_func = explore_probability_func 
        self.batch_size = batch_size 
        self.learning_rate = learning_rate 
        self.grad_clip = grad_clip 
        self.short_term_memory_length = short_term_memory_length 
        self.lbfgs = lbfgs 
        self.env_name = env_name 
        self.hessian_sum = hessian_sum 
        self.hessian_denominator = hessian_denominator
        self.hessian_center = hessian_center 
        self.hessian_sum_low_rank_half = hessian_sum_low_rank_half
        self.hessian_rank = hessian_rank 
        self.total_iters = total_iters
        self.info_prop_regularizer = info_prop_regularizer
        self.regularizing_lambda_function = regularizing_lambda_function 
        self.eta_space = eta_space 
        ## init feed forward net 
        self.fc1 = nn.Linear(input_dim * short_term_memory_length, 32) 
        self.pfc1 = PluggableLinear(self.fc1) 
        self.fc1_bn = nn.BatchNorm1d(32) 
        self.pfc1_bn = PluggableBatchNorm1d(self.fc1_bn) 
        self.fc2 = nn.Linear(32, 32) 
        self.pfc2 = PluggableLinear(self.fc2) 
        self.fc2_bn = nn.BatchNorm1d(32) 
        self.pfc2_bn = PluggableBatchNorm1d(self.fc2_bn) 
        self.fc3 = nn.Linear(32, n_actions) 
        self.pfc3 = PluggableLinear(self.fc3) 
        ## init data structures 
        if observations is None: 
            self.observations = [] 
        else: 
            self.observations = observations 
            pass 
        self.env = None 
        if self.lbfgs: 
            ## Misbehavior observed with large `history_size`, ie. >20 
            ## RAM req = O(history_size * model_dim) 
            self.optimizer = optim.LBFGS(self.parameters(), history_size=5) 
        else: 
            ## LBFGS was giving nan parameters 
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate) 
            pass
        ## count parameters 
        self.parameter_dim = self.get_parameter_vector().shape[0] 
        ## store this for future overwrites 
        self.parameter_names = [] 
        for n, _ in self.named_parameters(): 
            self.parameter_names.append(n) 
            pass 
        pass 
    
    def copy(self): 
        out = Model(input_dim=self.input_dim, 
                n_actions=self.n_actions, 
                max_sample=self.max_sample, 
                discount=self.discount, 
                eps=self.eps, 
                explore_probability_func=self.explore_probability_func, 
                batch_size=self.batch_size, 
                learning_rate=self.learning_rate, 
                grad_clip=self.grad_clip, 
                short_term_memory_length=self.short_term_memory_length, 
                lbfgs=self.lbfgs, 
                env_name=self.env_name,
                hessian_sum=self.hessian_sum.detach().clone() if self.hessian_sum is not None else None, 
                hessian_sum_low_rank_half=self.hessian_sum_low_rank_half, 
                hessian_rank=self.hessian_rank, 
                hessian_denominator=self.hessian_denominator, ## this is an int or None, so it is copied via '=' 
                hessian_center=self.hessian_center.detach().clone() if self.hessian_center is not None else None, 
                observations=self.observations.copy(), 
                total_iters=self.total_iters, 
                info_prop_regularizer=self.info_prop_regularizer, 
                regularizing_lambda_function=self.regularizing_lambda_function, 
                eta_space=self.eta_space.copy() if self.eta_space is not None else None)  
        out.load_state_dict(self.state_dict()) 
        return out 

    def forward(self, x): 
        if self.eta_space is not None: 
            ## TODO this gets called a lot. Is there a better place to put it? 
            theta_vec = self.eta_space.derive_theta() 
            self.assign_vec_to_params_as_tensors(theta_vec) 
            pass 
        x = x.reshape([-1, self.input_dim * self.short_term_memory_length]) 
        x = self.pfc1(x)
        x = self.fc1_bn(x) 
        x = torch.relu(x) 
        x = self.pfc2(x) 
        x = self.fc2_bn(x)  
        x = torch.relu(x) 
        x = self.pfc3(x) 
        x = x*x 
        return x 
    
    def get_action(self, env_state): 
        env_state = torch.tensor(env_state).float() 
        env_state = env_state.reshape([1, -1]) 
        predicted_reward_per_action_idx = self.forward(env_state) 
        return int(predicted_reward_per_action_idx.argmax()) 
    
    def store_observation(self, observation): 
        if len(self.observations) > self.max_sample: 
            self.observations = self.observations[1:] 
        self.observations.append(observation) 
        pass 
    
    def clear_observations(self): 
        self.observations = [] 
        pass 

    def convert_observations_to_memory(self, n_eigenvectors=None, lanczos_rank=None, n_reparameterized_dims=None): 
        if self.eta_space is not None: 
            ## optimize eta via theta, not theta directly 
            for p in self.parameters(): 
                p.requires_grad = False 
            pass 
        ## convert current observations to a Hessian matrix 
        target_model = self.copy() 
        for obs in self.observations: 
            ## update hessian  
            predicted, target, regularizer = self.__memory_replay(target_model=target_model, batch_size=None, fit=False, batch=[obs]) 
            loss = F.smooth_l1_loss(predicted, target) 
            ## mis-behaved  
            #if regularizer is not None: ## TODO function is duplicate -> package in a single function 
            #    if self.regularizing_lambda_function is not None:
            #        regularizer *= self.regularizing_lambda_function(self)
            #        pass
            #    loss += regularizer 
            loss.backward() 
            grad_vec = torch.cat([p.grad.reshape([-1]) for p in self.parameters()]) 
            outter_product = grad_vec.reshape([-1,1]).matmul(grad_vec.reshape([1,-1])) 
            if self.hessian_sum is None or self.hessian_denominator is None: 
                self.hessian_sum = outter_product 
                self.hessian_denominator = 1 
            else: 
                self.hessian_sum += outter_product 
                self.hessian_denominator += 1 
                pass
            pass 
        ## center quadratic form on current estimate 
        self.hessian_center = target_model.get_parameter_vector().detach() 
        ## wipe observations, and use memory going forward instead 
        self.clear_observations() 
        ## for simulation purposes only - no computational benefit is derived from using these features 
        if lanczos_rank is not None: 
            self.hessian_sum = torch.tensor(lanczos(self.hessian_sum.numpy(), lanczos_rank)).float()  
        if n_eigenvectors is not None: 
            ## extract low-rank approximation via eigenvector derivation from full-rank matrix 
            eigs = torch.linalg.eig(self.hessian_sum) 
            ## extract and truncate 
            vecs = eigs.eigenvectors.real
            vals = eigs.eigenvalues.real
            if n_reparameterized_dims is not None: 
                eta_basis = torch.tensor(vecs[:, -n_reparameterized_dims:]) 
                if self.eta_space is None: 
                    self.eta_space = EtaSpace(basis=eta_basis, origin=self.hessian_center) 
                else: 
                    self.eta_space.update_coordinates(basis=eta_basis, origin=self.hessian_center) 
                    pass 
                pass 
            vecs[:,n_eigenvectors:] = 0 
            vals[n_eigenvectors:] = 0  
            self.hessian_sum = vecs.matmul(torch.diag(vals)).matmul(vecs.transpose(0,1)) 
        pass 

    def assign_vec_to_params_as_tensors(self, vec): 
        ptr = 0 
        for pn in self.parameter_names: 
            ## find the parameter (or tensor) 
            op_str, param_str = pn.split('.') ### example: 'fc1.weights' 
            op = getattr(self, op_str) 
            param = getattr(op, param_str) 
            param_shape = param.shape 
            param_params = torch.prod(torch.tensor(param_shape)) 
            t = vec[ptr : ptr + param_params] ## tensor 
            t = t.reshape(param_shape) 
            ptr += param_params 
            target_op = getattr(self, 'p'+op_str) ## 'p'+ modifies PluggableLinear, because nn.Linear rejects Tensors  
            setattr(target_op, param_str, t) 
            pass 
        pass 

    def zero_grads_on_high_info_dims(self, proportion_zeroed): 
        ## ONLY WORKS FOR FULL HESSIANS 
        if self.hessian_sum is None: 
            return None 
        ## find info cut-off point 
        info_vec = torch.diag(self.hessian_sum) 
        info_vec_sorted = torch.sort(info_vec, descending=True).values 
        p = info_vec_sorted.shape[0] 
        info_max = info_vec_sorted[int(p * proportion_zeroed)] ## zero grads when info above this value 
        ## apply cut-off to grads 
        ptr = 0 
        for p in self.parameters(): 
            ## zero when above cut-off 
            grad_shape = p.grad.shape 
            grad_total_dims = torch.prod(torch.tensor(grad_shape)) 
            info_tensor = info_vec[ptr : ptr + grad_total_dims].reshape(grad_shape) 
            p.grad[info_tensor > info_max] = 0. 
            ## increment pointer 
            ptr += grad_total_dims 
            pass 
        pass 

    def __memory_replay(self, target_model, batch_size=None, fit=True, batch=None, optim_memorize=False): 
        ## random sample 
        obs = self.observations 
        if batch_size is not None: 
            if batch_size < len(self.observations): 
                obs = random.sample(self.observations, batch_size) 
        if batch is not None: 
            obs = batch 
        ## unpack samples 
        samples = [(torch.tensor(env_state).float(), \
                torch.tensor(reward).float(), \
                torch.tensor(done).int(), \
                torch.tensor(prev_env_state).float(), \
                torch.tensor(action).int()) for \
                (env_state, reward, done, info, prev_env_state, action) in obs] 
        ## build matrices 
        env_state = torch.stack([obs[0] for obs in samples], dim=0) ## inserts dim 0 
        observed_rewards = torch.stack([obs[1] for obs in samples], dim=0) 
        done = torch.stack([obs[2] for obs in samples], dim=0) 
        prev_env_state = torch.stack([obs[3] for obs in samples], dim=0) 
        action = torch.stack([obs[4] for obs in samples], dim=0).reshape([-1, 1]).type(torch.int64) 
        ## calculate target 
        with torch.no_grad(): 
            target_model.eval() 
            predicted_rewards = target_model.forward(env_state) 
            predicted_rewards = torch.max(predicted_rewards, dim=1, keepdim=True).values.reshape([-1]) 
            target = observed_rewards + (1 - done) * self.discount * predicted_rewards 
            target = target.reshape([-1, 1]).detach() 
            pass 
        ## calculate prediction 
        self.zero_grad() 
        if fit: 
            self.train() 
        else:
            self.eval() 
        predicted_rewards = self.forward(prev_env_state) 
        prediction = predicted_rewards.gather(1, action) 
        ## calculate memory regularizer 
        regularizer = None 
        if (self.hessian_sum is not None or self.hessian_sum_low_rank_half is not None) and \
                self.hessian_denominator is not None and \
                self.hessian_center is not None: 
            t = self.get_parameter_vector() 
            #t0 = target_model.get_parameter_vector().detach() 
            t0 = self.hessian_center 
            ## quadratic form around matrix estimating fisher information 
            if self.hessian_sum is not None:
                ## p X p hessian 
                regularizer = (t - t0).reshape([1, -1]).matmul(self.hessian_sum) 
            elif self.hessian_sum_low_rank_half is not None: 
                ## low-rank hessian 
                regularizer = (t - t0).reshape([1, -1]).matmul(self.hessian_sum_low_rank_half).matmul(self.hessian_sum_low_rank_half.transpose(0,1)) 
                pass
            regularizer = regularizer.matmul((t - t0).reshape([-1, 1]))
            regularizer *= .5 / self.hessian_denominator 
            regularizer = regularizer.reshape([]) 
        return prediction, target, regularizer 
    
    def get_parameter_vector(self): 
        return nn.utils.parameters_to_vector(self.parameters()) 
    
    def optimize(self, max_iter=None, batch_size=None, l2_regularizer=None, high_info_proportion=None): 
        if len(self.observations) < batch_size: 
            ## do not optimize without sufficient sample size 
            return None, None, None 
        iter_n = 0 
        n_dels = 30 
        dels = [None]*n_dels 
        continue_iterating = True 
        halt_method = None 
        loss_f = None 
        mean_reward = None 
        target_model = self.copy() 
        target_model.eval() 
        while continue_iterating: 
            prev_theta = self.get_parameter_vector() 
            predicted, target, regularizer = self.__memory_replay(target_model=target_model, batch_size=batch_size) 
            mean_reward = predicted.mean() 
            #loss = F.mse_loss(predicted, target) 
            loss = F.smooth_l1_loss(predicted, target) ## avg loss  
            if regularizer is not None and high_info_proportion is None: # and self.eta_space is None: 
                ## consider skipping if `eta_space is None`, because eta_space is orthogonal to memorized dimensions 
                ## i'm keeping it here, because i sometimes use different regularizers 
                if self.regularizing_lambda_function is not None:
                    regularizer *= self.regularizing_lambda_function(self) 
                    pass 
                loss += regularizer 
            if l2_regularizer is not None: 
                ## for experimental purposes only 
                t0 = self.hessian_center = target_model.get_parameter_vector().detach() 
                t = self.get_parameter_vector() 
                dt = (t - t0).reshape([-1, 1])
                loss += l2_regularizer * dt.transpose(0,1).matmul(dt).reshape([]) 
            loss_f = float(loss) 
            if self.eta_space is not None: 
                for p in self.parameters(): 
                    p.requires_grad = False 
                pass 
            loss.backward() 
            if not self.lbfgs: 
                ## lbfgs really doesn't like this 
                nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip) 
            ## lbfgs must re-evaluate target, hence lambda 
            if self.lbfgs: 
                self.optimizer.step(lambda: float(F.smooth_l1_loss(predicted, target))) 
            elif self.eta_space is not None:
                self.eta_space.optim_step()  
            elif high_info_proportion is not None: 
                self.zero_grads_on_high_info_dims(high_info_proportion) 
                self.optimizer.step() 
            else: 
                self.optimizer.step() 
                pass 
            updated_theta = self.get_parameter_vector() 
            ## decide to continue iterating or not 
            if max_iter is not None: 
                if iter_n > max_iter: 
                    ## halt: iters have hit limit 
                    continue_iterating = False 
                    halt_method = 'max-iter' 
            if iter_n >= n_dels: 
                ## test convergence with chebyshev ineq 
                dels = dels[1:] + [(updated_theta - prev_theta).abs().sum()] 
                sigma = torch.tensor(dels).square().mean().sqrt() 
                if (sigma/self.eps)**2 < .95: 
                    ## halt: convergance 
                    continue_iterating = False 
                    halt_method = 'cauchy-convergence' 
            else: 
                ## collect data for variance estimate 
                dels[iter_n] = (updated_theta - prev_theta).abs().sum() 
                pass 
            iter_n += 1 
            pass 
        return loss_f, halt_method, mean_reward  
    
    def simulate(self, fit=True, total_iters=10000, plot_rewards=False, plot_prob_func=False, tqdm_seconds=10, l2_regularizer=None, \
            game_modifier=0, silence_tqdm=True, high_info_proportion=None, memorization_frequency=None): 
        if plot_prob_func: 
            plt.plot([self.explore_probability_func(idx) for idx in range(total_iters)]) 
            plt.show() 
            pass 
        env = gym.make(self.env_name) 
        env_state, _ = env.reset() 
        env_state_list = [torch.tensor(env_state) for _ in range(self.short_term_memory_length)] 
        env_state = torch.cat(env_state_list) 
        last_start = 0 
        last_total_reward = 0 
        n_restarts = 0 
        self.total_rewards = [] 
        simulation_results = [] 
        ## run experiment 
        iters = tqdm(range(total_iters), disable=silence_tqdm, mininterval=tqdm_seconds, maxinterval=tqdm_seconds) 
        for iter_idx in iters: 
            prev_env_state = env_state 
            if self.explore_probability_func(iter_idx) > np.random.uniform(): ## TODO move to get_action 
                ## explore 
                action = env.action_space.sample()
            else: 
                ## exploit 
                self.eval() 
                action = self.get_action(env_state) 
                pass 
            env_state, reward, terminated, truncated, info = env.step(action) 
            done = truncated or terminated  
            if game_modifier > 0: 
                for _ in range(game_modifier): 
                    if not done: 
                        ## if not done, apply `action` iteratively 
                        env_state, rwd, terminated, truncated, info = env.step(action) 
                        done = truncated or terminated 
                        pass 
                    if not done: 
                        reward += rwd ## mechanic allows zero reward at done 
                    pass 
            elif done: 
                reward = 0
                pass 
            env_state_list = env_state_list[1:] + [torch.tensor(env_state)] 
            env_state = torch.cat(env_state_list) 
            last_total_reward += reward 
            observation = env_state, reward, done, info, prev_env_state, action 
            ## store for model fitting 
            self.store_observation(observation) 
            ## store for output 
            self.total_iters += 1 
            simulation_results.append((reward, done, self.total_iters)) 

            if len(self.observations) > self.batch_size and iter_idx % 1 == 0: ## online learning may not optimize as frequently! 
                _ = self.optimize(max_iter=1, batch_size=self.batch_size, l2_regularizer=l2_regularizer, high_info_proportion=high_info_proportion) 
                pass 

            if memorization_frequency is not None: 
                if memorization_frequency > self.batch_size:
                    if memorization_frequency < len(self.observations): 
                        self.convert_observations_to_memory() 
                        pass 
                    pass 
                else: 
                    raise Exception('memorization_frequency must be greater than batch_size!') 
                    pass 
                pass 

            if done: 
                env_state, _ = env.reset() 
                env_state_list = [torch.tensor(env_state) for _ in range(self.short_term_memory_length)] 
                env_state = torch.cat(env_state_list) 
                self.total_rewards.append(last_total_reward) 
                last_total_reward = 0 
                n_restarts += 1 
                pass 
            pass 
        env.close() 

        if plot_rewards: 
            plt.plot(self.total_rewards) 
            plt.show()
        return simulation_results 
    pass 

class EtaSpace(): 
    'a subspace paramerization in theta space' 
    def __init__(self, basis, origin, learning_rate=LEARNING_RATE): 
        '''
        - basis: (tensor in R^2) n_rows = dim(theta), n_cols = dim(eta)
        - origin: (tensor in theta space) Hessian center vector  
        ''' 
        self.basis = basis 
        self.theta_dim = basis.shape[0] 
        self.eta_dim = basis.shape[1] 
        self.eta = nn.Parameter(torch.zeros([self.eta_dim, 1])) 
        self.origin = origin 
        if origin.shape[0] != self.theta_dim: 
            raise Exception('ERROR: origin dimension and basis rows must equate!') 
            pass 
        self.learning_rate = learning_rate 
        self.optimizer = optim.Adam([self.eta], lr=learning_rate) 
        pass 
    def update_coordinates(self, basis, origin): 
        if basis.shape[0] != self.theta_dim or \
                basis.shape[1] != self.eta_dim or \
                origin.shape[0] != self.theta_dim: 
            raise Exception('ERROR: eta-space update must have matching dimensions!') 
            pass 
        self.basis = basis 
        self.origin = origin 
        pass 
    def derive_theta(self): 
        theta_vec = self.basis.matmul(self.eta) + self.origin.reshape([-1,1]) 
        return theta_vec 
    def optim_step(self): 
        self.optimizer.step() 
        pass 
    def copy(self): 
        return EtaSpace( 
                basis=self.basis.detach().clone(), 
                origin=self.origin.detach().clone(), 
                learning_rate=self.learning_rate 
                ) 
    pass 

class PluggableLinear(): 
    'nn.Linear, but parameters can be swapped-out with Tensors' 
    def __init__(self, nnLinear): 
        '''
        First, initialize nn.Linear. 
        Then, initialize this. 
        ''' 
        self.weight = nnLinear.weight 
        self.bias = nnLinear.bias 
        pass 
    def __call__(self, x): 
        return nn.functional.linear(x, self.weight, self.bias) 
    pass 

class PluggableBatchNorm1d(): 
    'nn.BatchNorm1d, but parameters can be swapped-out with Tensors' 
    def __init__(self, nnBatchNorm1d): 
        self.running_mean = nnBatchNorm1d.running_mean 
        self.running_var = nnBatchNorm1d.running_var 
        self.weight = nnBatchNorm1d.weight 
        self.bias = nnBatchNorm1d.bias 
        self.nnBatchNorm1d = nnBatchNorm1d ## used to ref bools and floats, since they can only pass by value 
        pass 
    def __call__(self, x): 
        training = self.nnBatchNorm1d.training 
        momentum = self.nnBatchNorm1d.momentum 
        eps = self.nnBatchNorm1d.eps 
        return nn.functional.batch_norm(
                x, 
                running_mean=self.running_mean, 
                running_var=self.running_var, 
                weight=self.weight, 
                bias=self.bias,
                training=training, 
                momentum=momentum, 
                eps=eps
                ) 
    pass 

