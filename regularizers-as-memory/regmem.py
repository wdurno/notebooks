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
            hessian_denominator=None, 
            hessian_center=None, 
            observations=[], 
            total_iters=0): 
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
        self.total_iters = total_iters 
        ## init feed forward net 
        self.fc1 = nn.Linear(input_dim * short_term_memory_length, 32) 
        self.fc1_bn = nn.BatchNorm1d(32) 
        self.fc2 = nn.Linear(32, 32) 
        self.fc2_bn = nn.BatchNorm1d(32) 
        self.fc3 = nn.Linear(32, n_actions) 
        ## init data structures 
        self.observations = observations 
        self.env = None 
        if self.lbfgs: 
            ## Misbehavior observed with large `history_size`, ie. >20 
            ## RAM req = O(history_size * model_dim) 
            self.optimizer = optim.LBFGS(self.parameters(), history_size=5) 
        else: 
            ## LBFGS was giving nan parameters 
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate) 
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
                hessian_denominator=self.hessian_denominator, ## this is an int or None, so it is copied via '=' 
                hessian_center=self.hessian_center.detach().clone() if self.hessian_center is not None else None, 
                observations=self.observations.copy(), 
                total_iters=self.total_iters) 
        out.load_state_dict(self.state_dict()) 
        return out 

    def forward(self, x): 
        x = x.reshape([-1, self.input_dim * self.short_term_memory_length]) 
        x = self.fc1(x)
        x = self.fc1_bn(x) 
        x = torch.relu(x) 
        x = self.fc2(x) 
        x = self.fc2_bn(x)  
        x = torch.relu(x) 
        x = self.fc3(x) 
        x = x*x 
        return x 
    
    def get_action(self, env_state): 
        env_state = torch.tensor(env_state).float() 
        env_state = env_state.reshape([1, -1]) 
        predicted_reward_per_action_idx = self.forward(env_state) 
        return int(predicted_reward_per_action_idx.argmax()) 
    
    def store_observation(self, observation): 
        if len(observation) > self.max_sample: 
            observation = observation[1:] 
        self.observations.append(observation) 
        pass 
    
    def clear_observations(self): 
        self.observations = [] 
        pass 

    def convert_observations_to_memory(self, n_eigenvectors = None): 
        ## convert current observations to a Hessian matrix 
        target_model = self.copy() 
        for obs in self.observations: 
            ## update hessian  
            predicted, target, _ = self.__memory_replay(target_model=target_model, batch_size=None, fit=False, batch=[obs]) 
            loss = F.smooth_l1_loss(predicted, target) 
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
        ## center quadratic form on current estimate 
        self.hessian_center = target_model.get_parameter_vector().detach() 
        ## wipe observations, and use memory going forward instead 
        self.clear_observations() 
        ## for simulation purposes only - no computational benefit is derived from using this feature 
        if n_eigenvectors is not None: 
            eigs = torch.linalg.eig(self.hessian_sum) 
            ## extract and truncate 
            vecs = eigs.eigenvectors.real
            vals = eigs.eigenvalues.real
            vecs[:,n_eigenvectors:] = 0 
            vals[n_eigenvectors:] = 0  
            self.hessian_sum = vecs.matmul(torch.diag(vals)).matmul(vecs.transpose(0,1)) 
        pass 
    
    def __memory_replay(self, target_model, batch_size=None, fit=True, batch=None): 
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
        if self.hessian_sum is not None and self.hessian_denominator is not None and self.hessian_center is not None: 
            t = self.get_parameter_vector() 
            #t0 = target_model.get_parameter_vector().detach() 
            t0 = self.hessian_center 
            ## quadratic form around matrix estimating fisher information 
            regularizer = (t - t0).reshape([1, -1]).matmul(self.hessian_sum) 
            regularizer = regularizer.matmul((t - t0).reshape([-1, 1]))
            regularizer *= .5 / self.hessian_denominator 
            regularizer = regularizer.reshape([])
        return prediction, target, regularizer 
    
    def get_parameter_vector(self): 
        return nn.utils.parameters_to_vector(self.parameters()) 
    
    def optimize(self, max_iter=None, batch_size=None): 
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
            loss = F.smooth_l1_loss(predicted, target) 
            if regularizer is not None: 
                #n = predicted.shape[0] 
                #loss = (n*loss + regularizer)/(n + self.) ## TODO balance by sample size 
                loss += regularizer 
            loss_f = float(loss) 
            loss.backward() 
            if not self.lbfgs: 
                ## lbfgs really doesn't like this 
                nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip) 
            ## lbfgs must re-evaluate target, hence lambda 
            if self.lbfgs: 
                self.optimizer.step(lambda: float(F.smooth_l1_loss(predicted, target))) 
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
    
    def simulate(self, fit=True, total_iters=10000, plot_rewards=True, plot_prob_func=True, tqdm_seconds=10): 
        if plot_prob_func: 
            plt.plot([self.explore_probability_func(idx) for idx in range(total_iters)]) 
            plt.show() 
            pass 
        env = gym.make(self.env_name) 
        env_state = env.reset() 
        env_state_list = [torch.tensor(env_state) for _ in range(self.short_term_memory_length)] 
        env_state = torch.cat(env_state_list) 
        last_start = 0 
        last_total_reward = 0 
        n_restarts = 0 
        self.total_rewards = [] 
        simulation_results = [] 
        ## run experiment 
        iters = tqdm(range(total_iters), disable=False, mininterval=tqdm_seconds, maxinterval=tqdm_seconds) 
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
            env_state, reward, done, info = env.step(action) 
            env_state_list = env_state_list[1:] + [torch.tensor(env_state)] 
            env_state = torch.cat(env_state_list) 
            if done: 
                reward = 0 
                pass 
            last_total_reward += reward 
            observation = env_state, reward, done, info, prev_env_state, action 
            ## store for model fitting 
            self.store_observation(observation) 
            ## store for output 
            self.total_iters += 1 
            simulation_results.append((reward, done, self.total_iters)) 

            if iter_idx > 30 and iter_idx % 1 == 0: 
                _ = self.optimize(max_iter=1, batch_size=self.batch_size) 
                #loss, halt_method, mean_reward = self.optimize(max_iter=1, batch_size=self.batch_size) 
                pass 
                #loss = float(loss) 
                #mean_reward = float(mean_reward) 
                #param_nan = self.get_parameter_vector().isnan().sum() 
                ## too many print statements 
                #iters.set_description(f'n_restarts: {n_restarts}, last_total_reward: {last_total_reward}, '+\
                #    f'loss: {round(loss,4)}, halt: {halt_method}, mean_reward: {round(mean_reward,2)}, action: {action}') 
                #pass 

            if done: 
                env_state = env.reset() 
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