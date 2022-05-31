## Define model 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import randrange 
import gym 
from tqdm import tqdm 
import numpy as np 
import matplotlib.pyplot as plt 
import copy 
import os 
from PIL import Image 

INPUT_DIM = 4
N_ACTIONS = 2
MAX_SAMPLE = 100000
DISCOUNT = .95 
EPS = 1e-5
EXPLORE_PROBABILITY_FUNC = lambda idx: 0.99**idx 
BATCH_SIZE = 30 
LEARNING_RATE = 0.001 
GRAD_CLIP = 10.0 
SHORT_TERM_MEMORY_LENGTH = 40 
LBFGS = False 
ENV_NAME = 'CartPole-v1' 

## necessary for rgb_array renders while without videodrivers 
os.environ["SDL_VIDEODRIVER"] = "dummy" 

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
            hessian_denominator=None, 
            hessian_center=None, 
            observations=None,
            total_iters=0,
            regularizing_lambda_function=None, 
            lstm_state_dim=32): 
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
        self.total_iters = total_iters
        self.regularizing_lambda_function = regularizing_lambda_function 
        ## init CNN + LSTM net  
        ## CCNs 
        ## input: (-1, 3, 40, 60) 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1) ## to (-1, 32, 38, 58) 
        self.conv1_bn = nn.BatchNorm2d(32) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2) ## to (-1, 64, 18, 28) 
        self.conv2_bn = nn.BatchNorm2d(64) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2) ## to (-1, 128, 8, 13) 
        self.conv3_bn = nn.BatchNorm2d(128) 
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2) ## to (-1, 256, 3, 6) 
        self.conv4_bn = nn.BatchNorm2d(256) 
        self.conv5 = nn.Conv2d(256, 32, kernel_size=3, stride=2) ## to (-1, 32, 1, 2) 
        self.conv5_bn = nn.BatchNorm2d(32) 
        ## LSTM  
        self.lstm = nn.LSTM(32*1*2, self.lstm_state_dim)  
        ## FCs 
        self.fc1 = nn.Linear(self.lstm_state_dim, 32) 
        self.fc1_bn = nn.BatchNorm1d(32) 
        self.fc2 = nn.Linear(32, n_actions) 
        ## init data structures 
        if observations is None: 
            self.observations = [] 
        else: 
            self.observations = observations 
            pass 
        ## set LSTM hidden values to zeros 
        self.clear_short_term_memory() 
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
                hessian_sum_low_rank_half=self.hessian_sum_low_rank_half, 
                hessian_denominator=self.hessian_denominator, ## this is an int or None, so it is copied via '=' 
                hessian_center=self.hessian_center.detach().clone() if self.hessian_center is not None else None, 
                observations=self.observations.copy(), 
                total_iters=self.total_iters, 
                regularizing_lambda_function=self.regularizing_lambda_function, 
                lstm_state_dim=self.lstm_state_dim) 
        out.load_state_dict(self.state_dict()) 
        return out 

    def forward(self, x): 
        ## x should be shape [series_idx, sample_idx, channel_idx, height_idx, width_idx] 
        ## we'll use this shorthand: [L, N, C, H, W] <-- names should align to PyTorch's CNN and LSTM documentation  
        ## first, we must reshape for CNNs 
        L, N, C, H, W = tuple(x.shape) 
        x = x.reshape([L*N, C, H, W]) 
        x = self.conv1(x) 
        x = self.conv1_bn(x) 
        x = torch.relu(x) 
        x = self.conv2(x) 
        x = self.conv2_bn(x) 
        x = torch.relu(x) 
        x = self.conv3(x) 
        x = self.conv3_bn(x) 
        x = torch.relu(x) 
        x = self.conv4(x) 
        x = self.conv4_bn(x) 
        x = torch.relu(x) 
        x = self.conv5(x) 
        x = self.conv5_bn(x) 
        x = torch.relu(x) 
        ## reshape for LSTM 
        x = x.reshape([L, N, -1]) 
        x, self.hidden = self.lstm(x, self.hidden) 
        ## extract last of series 
        x = x[L-1, :, :] ## has shape [N, -1] 
        x = torch.relu(x) 
        x = self.fc1(x)
        x = self.fc1_bn(x) 
        x = torch.relu(x) 
        x = self.fc2(x) 
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
    
    def clear_short_term_memory(self): 
        self.hidden = (torch.zeros([self.lstm_state_dim]), torch.zeros([self.lstm_state_dim])) 
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

    def sample_observations(self, batch_size): 
        ## sample indices between second observation and last, inclusive  
        # observation structure: 0: reward, 1: done, 2: info, 3: env_state, 4: action, 5: hidden 
        out = []
        for _ in range(batch_size): 
            ## find starting position 
            ## note: observations stores a series of time-sorted episodes  
            ## iterate backward until you hit a halt, index 0, or too much distance 
            idx = randrange(1, len(self.observations)) 
            fail_count = 0 
            while self.observations[idx-1][1]: 
                ## must not be done 
                idx = randrange(1, len(self.observations)) 
                fail_count += 1 
                if fail_count > 1000: 
                    ## should almost-surely never occur 
                    raise Exception('ERROR: Could not sample valid series of observations!') 
                pass 
            start = idx 
            continue_search = True 
            while continue_search: 
                start -= 1 
                if start == 0:
                    continue_search = False 
                if self.observations[start][2]: 
                    ## done == True 
                    ## hit previous finish 
                    continue_search = False 
                    start += 1 
                if idx - start == self.short_term_memory_length: 
                    continue_search = False 
                pass 
            ## extract 
            ## sequences are +1 length for target network 
            subsequence = [self.observations[i] for i in range(start, idx+1)] 
            if len(subsequence) < self.short_term_memory_length + 1: 
                ## pad as needed 
                subsequence = [subsequence[0]]*(self.short_term_memory_length - len(sequence) + 1) + sequence 
                pass 
            out.append(subsequence) 
            pass 
        return out 

    def __memory_replay(self, target_model, batch_size=None, fit=True, batch=None): 
        ## random sample of memory sequences 
        if batch_size is not None: 
            if batch_size < len(self.observations): 
                sample = self.sample_observations(batch_size) 
        if batch is not None: 
            sample = batch
        ## construct target and prediction vectors 
        ## observation structure: 
        ##  0: reward, 1: done, 2: info, 3: env_state, 4: action, 5: hidden 
        ## build state tensor 
        state_series_list = [] 
        for series in sample: 
            series_list = [] 
            for observation in series: 
                env_state = observation[3] 
                series_list.append(env_state) 
                pass 
            series_tensor = torch.stack(series_list) 
            state_series_list.append(series_tensor) 
            pass 
        state = torch.stack(state_series_list) 
        ## state dimensions: [series_index, timestep_index, channel_index, height_index, width_index] 
        ## model expects otherwise, permute to [timestep_index, series_index, channel_index, height_index, width_index] 
        state = state.permute([1, 0, 2, 3, 4]) 
        ## build initial hidden state tensors 
        prev_hidden_list = [] 
        hidden_list = [] 
        for series in sample: 
            ## all series have at least 2 actual observations 
            prev_hidden = series[0][5] 
            hidden = series[1][5] 
            prev_hidden_list.append(prev_hidden) 
            hidden_list.append(hidden) 
            pass 
        prev_hidden = torch.stack(prev_hidden_list) 
        hidden = torch.stack(hidden_list) 
        ## build reward tensor 
        reward_list = [] 
        for series in sample: 
            reward = torch.Tensor(series[-1][0]) 
            reward_list.append(reward) 
            pass 
        reward = torch.stack(reward_list) 
        ## build done tensor 
        done_list = [] 
        for series in sample: 
            done = torch.Tensor(int(series[-1][1])) 
            done_list.append(done) 
            pass 
        done = torch.stack(done_list) 
        ## build action tensor 
        action_list = [] 
        for series in sample: 
            action = torch.Tensor(int(series[-1][4])) 
            action_list.append(action) 
            pass 
        action = torch.stack(action) 
        ## set hidden states 
        self.hidden = prev_hidden 
        target_model.hidden = hidden 
        ## run predictions 
        prediction = self.forward(state[:-1,:,:,:,:]).gather(1, action) 
        t = target_model.forward(state[1:,:,:,:,:]) 
        t = torch.max(t, dim=1, keepdim=True).values.reshape([-1]) 
        target = reward + (1 - done) * self.discount * t 
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
    
    def optimize(self, max_iter=None, batch_size=None, l2_regularizer=None): 
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
        self.train() 
        while continue_iterating: 
            prev_theta = self.get_parameter_vector() 
            predicted, target, regularizer = self.__memory_replay(target_model=target_model, batch_size=batch_size) 
            mean_reward = predicted.mean() 
            loss = F.smooth_l1_loss(predicted, target) ## avg loss  
            if regularizer is not None: 
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
    
    def simulate(self, fit=True, total_iters=10000, plot_rewards=True, plot_prob_func=True, tqdm_seconds=10, l2_regularizer=None): 
        if plot_prob_func: 
            plt.plot([self.explore_probability_func(idx) for idx in range(total_iters)]) 
            plt.show() 
            pass 
        env = gym.make(self.env_name) 
        env_state = env.reset() 
        env_state = env.render(mode='rgb_array') 
        env_state = np.asarray(Image.fromarray(env_state).resize((40,60))) 
        env_state = torch.tensor(env_state).reshape([1, 40, 60, 3]).permute(0, 3, 1, 2)/255. - .5 
        sequence = [env_state]*self.short_term_memory_length 
        self.clear_short_term_memory 
        ## store initial observation 
        observation = 0, False, {}, env_state, 0, self.hidden 
        self.store_observation(observation) 
        last_start = 0 
        last_total_reward = 0 
        n_restarts = 0 
        self.total_rewards = [] 
        simulation_results = [] 
        ## run experiment 
        iters = tqdm(range(total_iters), disable=False, mininterval=tqdm_seconds, maxinterval=tqdm_seconds) 
        for iter_idx in iters: 
            prev_env_state = env_state 
            prev_hidden = (self.hidden[0].detach(), self.hidden[1].detach())  
            if self.explore_probability_func(iter_idx) > np.random.uniform():  
                ## explore 
                action = env.action_space.sample()
            else: 
                ## exploit 
                self.eval() 
                sequence_tensor = torch.cat(sequence).reshape([self.short_term_memory_length, 1, 3, 40, 60]) 
                action = self.get_action(sequence_tensor) 
                pass 
            env_state, reward, done, info = env.step(action) 
            ## get rgb array of state 
            env_state = env.render(mode='rgb_array') 
            ## compress 
            env_state = np.asarray(Image.fromarray(env_state).resize((40,60))) 
            ## hidden 
            hidden = (self.hidden[0].detach(), self.hidden[1].detach()) 
            ## do not reward failure 
            if done: 
                reward = 0 
                pass 
            last_total_reward += reward 
            ## format env_states for pytorch 
            env_state = torch.tensor(env_state).reshape([1, 40, 60, 3]).permute([0, 3, 1, 2])/255. - .5 
            ## add to sequence 
            sequence = sequence[1:] 
            sequence.append(env_state) 
            observation = reward, done, info, env_state, action, hidden  
            ## store for model fitting 
            self.store_observation(observation) 
            ## store for output 
            self.total_iters += 1 
            simulation_results.append((reward, done, self.total_iters)) 

            if iter_idx > 30 and iter_idx % 1 == 0: 
                _ = self.optimize(max_iter=1, batch_size=self.batch_size, l2_regularizer=l2_regularizer) 
                self.hidden = hidden 
                self.eval() 
                pass 

            if done: 
                env_state = env.reset() 
                env_state = env.render(mode='rgb_array')
                env_state = np.asarray(Image.fromarray(env_state).resize((40,60))) 
                env_state = torch.Tensor(env_state).reshape([1, 40, 60, 3]).permute([0, 3, 1, 2])/255. - .5   
                sequence = [env_state]*self.short_term_memory_length 
                self.clear_short_term_memory() 
                ## store initial observation 
                obesrvation = 0, False, {}, env_state, 0, self.hidden 
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
