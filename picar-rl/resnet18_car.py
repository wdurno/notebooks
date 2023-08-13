import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import randrange, sample 
import gym 
from tqdm import tqdm 
import numpy as np 
import matplotlib.pyplot as plt 
import copy 
import os 
from PIL import Image 
from car_env.car_client import PiCarEnv 
from car_env.constants import N_ACTIONS, N_CAMERA_DIRECTIONS  

MAX_SAMPLE = 100000
DISCOUNT = .5 # .5 # .95 
EPS = 1e-5
#EXPLORE_PROBABILITY_FUNC = lambda idx: 0.999**idx ## lambda doesn't pickle 
def EXPLORE_PROBABILITY_FUNC (idx):
    return 0.99**idx 
BATCH_SIZE = 25  
LEARNING_RATE = 0.001 # 0.01 # 0.001  
GRAD_CLIP = 10.0 
SHORT_TERM_MEMORY_LENGTH = 40 
LBFGS = False 
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
#DEVICE = torch.device('cpu') 

## necessary for rgb_array renders while without videodrivers 
#os.environ["SDL_VIDEODRIVER"] = "dummy" 

class Model(nn.Module): 
    def __init__(self, 
            n_actions=N_ACTIONS, 
            n_camera_directions=N_CAMERA_DIRECTIONS, 
            max_sample=MAX_SAMPLE, 
            discount=DISCOUNT, 
            eps=EPS, 
            explore_probability_func=EXPLORE_PROBABILITY_FUNC, 
            batch_size=BATCH_SIZE, 
            learning_rate=LEARNING_RATE, 
            grad_clip=GRAD_CLIP, 
            short_term_memory_length=SHORT_TERM_MEMORY_LENGTH, 
            lbfgs=LBFGS, 
            memorized_hessian_approximation=None, 
            memorized_parameters=None, 
            observations=None, 
            total_iters=0,
            regularizing_lambda_function=None): 
        super(Model, self).__init__() 
        ## store config 
        self.n_actions = n_actions 
        self.n_camera_directions = n_camera_directions 
        self.max_sample = max_sample 
        self.discount = discount 
        self.eps = eps 
        self.explore_probability_func = explore_probability_func 
        self.batch_size = batch_size 
        self.learning_rate = learning_rate 
        self.grad_clip = grad_clip 
        self.short_term_memory_length = short_term_memory_length 
        self.lbfgs = lbfgs 
        self.memorized_hessian_approximation = memorized_hessian_approximation 
        self.memorized_parameters = memorized_parameters 
        self.total_iters = total_iters 
        self.regularizing_lambda_function = regularizing_lambda_function 
        ## init resnet 
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True) 
        ## shrink via dense net before adding camera embedding 
        self.fc1  = nn.Linaer(1000, 128) 
        self.fc1_bn = nn.BatchNorm1d(128) 
        ## camera embedding 
        self.embedding = nn.Embedding(self.n_camera_directions, 128) 
        ## FCs 
        self.fc2 = nn.Linear(128+128, 64) ## + 64 for camera embedding vec 
        self.fc2_bn = nn.BatchNorm1d(64) 
        self.fc3 = nn.Linear(64, n_actions) 
        ## init data structures 
        if observations is None: 
            self.observations = [] 
        else: 
            self.observations = observations 
            pass 
        if self.lbfgs: 
            ## Misbehavior observed with large `history_size`, ie. >20 
            ## RAM req = O(history_size * model_dim) 
            self.optimizer = optim.LBFGS(self.named_parameters(), history_size=5) 
        else: 
            ## LBFGS was giving nan parameters 
            self.optimizer = optim.Adam(self.named_parameters(), lr=self.learning_rate) 
            pass
        self.to(DEVICE) 
        pass 
    
    def copy(self): 
        out = Model(n_actions=self.n_actions, 
                n_camera_directions=self.n_camera_directions, 
                max_sample=self.max_sample, 
                discount=self.discount, 
                eps=self.eps, 
                explore_probability_func=self.explore_probability_func, 
                batch_size=self.batch_size, 
                learning_rate=self.learning_rate, 
                grad_clip=self.grad_clip, 
                short_term_memory_length=self.short_term_memory_length, 
                lbfgs=self.lbfgs, 
                memorized_hessian_approximation = self.memorized_hessian_approximation
                memorized_parameters = self.memorized_parameters
                observations=self.observations.copy(), 
                total_iters=self.total_iters, 
                regularizing_lambda_function=self.regularizing_lambda_function)
        out.load_state_dict(self.state_dict()) 
        return out 

    def forward(self, x): 
        ## extract pov and camera 
        pov = x['pov'] 
        camera = x['camera'] 
        ## pov should be shape [series_idx, sample_idx, channel_idx, height_idx, width_idx] 
        ## we'll use this shorthand: [L, N, C, H, W] <-- names should align to PyTorch's CNN and LSTM documentation  
        ## first, we must reshape for CNNs 
        L, N, C, H, W = tuple(pov.shape) 
        L_batches_in = pov.unbind() 
        L_batches_out = [] 
        for batch in L_batches_in: ## TODO: optimize! current form is not efficient, but guaranteed to work 
            pov = self.conv1(batch) 
            pov = self.conv1_bn(pov) 
            pov = torch.relu(pov) 
            #pov = self.mp2d1(pov)  
            pov = self.conv2(pov) 
            pov = self.conv2_bn(pov) 
            pov = torch.relu(pov) 
            #pov = self.mp2d2(pov) 
            pov = self.conv3(pov) 
            pov = self.conv3_bn(pov) 
            pov = torch.relu(pov) 
            pov = pov.reshape([N, -1]) 
            L_batches_out.append(pov) 
        ## reshape for CNNs over time  
        pov = torch.stack(L_batches_out) ## [L, N, -1] 
        pov = pov.permute(0, 2, 1) # [L, -1, N] 
        pov = self.conv6(pov)
        pov = self.conv6_bn(pov) 
        pov = torch.relu(pov) 
        ## Dense 
        pov = pov.reshape([L, -1]) 
        camera_vec = self.embedding(camera).squeeze(dim=1) ## [N, 1, 64] -> [N, 64] 
        x = torch.cat([pov, camera_vec], dim=1) 
        x = self.fc1(x)
        x = self.fc1_bn(x) 
        x = torch.relu(x) 
        x = self.fc2(x) 
        return x 
    
    def load_car_env_data(self, dir_path): 
        '''
        load data generated by the car_env api
        inputs:
        - dir_path: directory containing perhaps several .pkl files 
        outputs:
        - None 
        side-effects: 
        - this agent's memory buffer is populated 
        '''
        ## scan the directory for .pkl files to load 
        candidate_files = os.listdir(dir_path) 
        files_to_load = [f for f in candidate_files if f.endswith('.pkl')] 
        paths = [os.path.join(dir_path, f) for f in files_to_load] 
        for path in paths: 
            ## load and upack data  
            data = PiCarEnv.load_memory(path) 
            image_list = data[0] 
            action_list = data[1] 
            #x_location_list = data[2] ## not used 
            radius_list = data[3] 
            reward_list = [PiCarEnv.get_reward(r) for r in radius_list] ## transform 
            camera_position_list = data[4] 
            ## store in memory 
            n = len(image_list) 
            for i in range(n): 
                env_state = self.__format_observation(image_list[i], camera_position_list[i]) 
                obs = (
                        reward_list[i], ## 0: reward  
                        False,          ## 1: done 
                        {},             ## 2: information 
                        env_state,      ## 3: env_state 
                        action_list[i]  ## 4: action 
                        )
                self.store_observation(obs) 
                pass 
            pass 
        pass 

    def get_action(self, env_state): 
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

    def memorize(self, resnet_multiplier=1., disable_tqdm=True): 
        'Update sufficient statistics. Only apply near a convergence point.' 
        ## init, if needed 
        if self.memorized_hessian_approximation is None: 
            self.memorized_hessian_approximation = 0. 
            pass 
        if self.memorized_parameters is None: 
            self.memorized_parameters = 0. 
            pass 
        ## update hessian appoximation 
        ## approximation is n * diag(hessian) 
        for named_grad_list in self.__get_named_grad_generator(): 
            grad_list = [] 
            for name, grad in named_grad_list: 
                if 'resnet' in name: 
                    grad *= resnet_multiplier 
                    pass 
                grad_list.append(grad_list) 
                pass 
            grad_vec = torch.cat(grad_list) 
            self.memorized_hessian_approximation += grad_vec * grad_vec  
            pass 
        ## update memorized parameters 
        self.memorized_parameters = [] 
        for _, p in self.named_parameters(): 
            self.memorized_parameters.append(p.reshape([-1, 1])) 
            pass 
        self.memorized_parameters = torch.cat(self.memorized_parameters) 
        pass 

    def update_target_model(self): 
        'Enables q-learning convergence. Use several times while hunting for convergence points.' 
        target_model = self.copy() 
        pass 

    def __get_named_grad_generator(): 
        def named_grad_generator(): 
            target_model.eval() 
            self.eval() 
            for obs_idx in range(1, len(self.observations)):
                ## check validity 
                prev_obs = self.observations[obs_idx-1]
                prev_done = prev_obs[1]
                if not prev_done:
                    ## observation is valid, proceed 
                    ## update hessian  
                    predicted, target, _ = self.__memory_replay(target_model=target_model, batch_size=None, fit=False, batch=[obs_idx])
                    loss = F.smooth_l1_loss(predicted, target)
                    loss.backward()
                    named_grad = [(t[0], t[1].grad.reshape([-1])) for t in self.named_parameters()] 
                    yield named_grad 
                    pass 
                pass 
            pass 
        return named_grad_generator  

    def __outer_product_diagonal(self, a, b): 
        'numerically efficient caclulation of diag(a * b^T)' 
        return (a*b).sum(dim=1) 

    def __sample_observations(self, batch_size=30, batch=None): 
        ## sample indices between second observation and last, inclusive  
        # observation structure: 0: reward, 1: done, 2: info, 3: env_state, 4: action  
        out = [] 
        if batch is not None: 
            ## batch should be a list of indices 
            batch_size = len(batch) 
        for i in range(batch_size): 
            ## find starting position 
            ## note: observations stores a series of time-sorted episodes  
            ## iterate backward until you hit a halt, index 0, or too much distance 
            if batch is None: 
                idx = randrange(1, len(self.observations)) 
                fail_count = 0 
                while self.observations[idx-1][1]: 
                    ## prev obs is done! find another 
                    ## must not be done 
                    idx = randrange(1, len(self.observations)) 
                    fail_count += 1 
                    if fail_count > 1000: 
                        ## should almost-surely never occur 
                        raise Exception('ERROR: Could not sample valid series of observations!') 
                    pass 
            else: 
                idx = batch[i] 
                pass 
            start = idx 
            continue_search = True 
            while continue_search: 
                start -= 1 
                if start == 0:
                    continue_search = False 
                if self.observations[start][1]: 
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
                subsequence = [subsequence[0]]*(self.short_term_memory_length - len(subsequence) + 1) + subsequence 
                pass 
            out.append(subsequence) 
            pass 
        return out 

    def __memory_replay(self, target_model, batch_size=None, fit=True, batch=None): 
        ## random sample of memory sequences 
        if batch_size is not None: 
            if batch_size < len(self.observations): 
                sample = self.__sample_observations(batch_size) 
        if batch is not None: 
            sample = self.__sample_observations(batch=batch) 
            pass 
        ## construct target and prediction vectors 
        ## observation structure: 
        ## 0: reward, 1: done, 2: info, 3: env_state, 4: action  
        ## build pov tensor 
        pov_series_list = [] 
        for series in sample: 
            series_list = [] 
            for observation in series: 
                env_state = observation[3] 
                series_list.append(env_state['pov']) 
                pass 
            series_tensor = torch.cat(series_list) 
            pov_series_list.append(series_tensor) 
            pass 
        pov = torch.stack(pov_series_list) 
        ## build camera tensor 
        camera_series_list = [] 
        for series in sample:
            penultimate_observation = series[-2] 
            env_state = penultimate_observation[3] 
            camera_series_list.append(env_state['camera']) 
            pass 
        final_observation = sample[-1][-1] 
        final_camera = final_observation[3]['camera'] 
        camera_series_list.append(final_camera) 
        camera = torch.cat(camera_series_list) 
        ## state dimensions: [series_index, timestep_index, channel_index, height_index, width_index] 
        ## model expects otherwise, permute to [timestep_index, series_index, channel_index, height_index, width_index] 
        ## build reward tensor 
        reward_list = [] 
        for series in sample: 
            reward = torch.tensor([series[-2][0]], device=DEVICE) 
            reward_list.append(reward) 
            pass 
        reward = torch.stack(reward_list) 
        ## build done tensor 
        done_list = [] 
        for series in sample: 
            done = torch.tensor([int(series[-1][1])], device=DEVICE) 
            done_list.append(done) 
            pass 
        done = torch.stack(done_list) 
        ## build action tensor 
        action_list = [] 
        for series in sample: 
            action = torch.tensor([int(series[-2][4])], device=DEVICE) 
            action_list.append(action) 
            pass 
        action = torch.stack(action_list) 
        ## run predictions 
        prediction = self.forward({'pov': pov[:,:-1,:,:,:], 'camera': camera[:-1,:]}) 
        prediction = prediction.gather(1, action) 
        t = target_model.forward({'pov': pov[:,1:,:,:,:], 'camera': camera[1:,:]}) 
        t = torch.max(t, dim=1, keepdim=True).values.reshape([-1, 1]) 
        target = reward + (1 - done) * self.discount * t 
        target = target.detach() 
        ## calculate memory regularizer 
        regularizer = None 
        if memorized_hessian_approximation is not None and memorized_parameters is not None: 
            t = self.get_parameter_vector(t) 
            dif = (t - self.memorized_parameters).reshape([-1, 1]) 
            ## dif^T * diag(v) * dif = (dif * v)^T dif, where dif*v is element-wise 
            regularizer = .5 * (dif * self.memorized_hessian_approximation.reshape([-1, 1])).transpose(0, 1).matmul(dif) 
            regularizer = regularizer.reshape([]) ## to scalar 
            pass 
        return prediction, target, regularizer 
    
    def get_parameter_vector(self): 
        parameter_list = [] 
        for _, p in self.named_parameters(): 
            parameter_list.append(p.reshape([-1, 1])) 
            pass 
        return torch.cat(parameter_list) 
    
    def __format_observation(self, image, camera): 
        out = {} 
        pov = torch.tensor(image.copy()).permute([2, 0, 1])/255. - .5 
        pov = pov.unsqueeze(0) 
        camera = torch.tensor(camera)   
        out['pov'] = pov.float().to(DEVICE)  
        out['camera'] = camera.int().reshape([1, 1]).to(DEVICE) ## make observations stackable 
        return out 

    def optimize(self, max_iter=None, batch_size=None, l2_regularizer=None, log1p_regularizer=False): 
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
            self.zero_grad() 
            prev_theta = self.get_parameter_vector() 
            predicted, target, regularizer = self.__memory_replay(target_model=target_model, batch_size=batch_size) 
            mean_reward = predicted.mean() 
            loss = F.smooth_l1_loss(predicted, target) ## avg loss  
            if regularizer is not None: 
                if self.regularizing_lambda_function is not None:
                    regularizer *= self.regularizing_lambda_function(self) 
                    pass 
                if log1p_regularizer: 
                    regularizer = torch.log1p(regularizer) 
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
    
    def simulate(self, host, fit=True, total_iters=10000, plot_rewards=False, plot_prob_func=False, \
            tqdm_seconds=.1, l2_regularizer=None, fit_freq=10, manual_play=False, log1p_regularizer=False): 
        if manual_play: 
            pygame.init() 
            pygame.display.set_mode((100,100)) 
        if plot_prob_func: 
            plt.plot([self.explore_probability_func(idx) for idx in range(total_iters)]) 
            plt.show() 
            pass
        env = PiCarEnv(host, x_resize=224, y_resize=224) ## resnet requires this input size 
        pov, camera = env.reset() 
        env_state = self.__format_observation(pov, camera) 
        sequence = {'pov': [env_state['pov']]*self.short_term_memory_length, 'camera': env_state['camera']} 
        ## store initial observation 
        observation = 0, False, {}, env_state, 0  
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
            if self.explore_probability_func(iter_idx) > np.random.uniform():  
                ## explore 
                action = sample(list(range(N_ACTIONS)), 1)[0] 
            else: 
                ## exploit 
                self.eval() 
                sequence_obs = {'pov': torch.cat(sequence['pov']).unsqueeze(dim=0), 'camera': sequence['camera']} 
                action = self.get_action(sequence_obs) 
                pass 
            if manual_play: 
                for key in range(0,N_ACTIONS): 
                    if getattr(pygame, f'K_{key}'): 
                        action = key 
                        pass
                    pass
                pass 
            #env_state, reward, done, info = env.step(action) 
            pov, camera, reward = env.step(action) 
            done, info = False, {} 
            env_state = self.__format_observation(pov, camera) 
            last_total_reward += reward 
            ## add to sequence 
            sequence['pov'] = sequence['pov'][1:] 
            sequence['pov'].append(env_state['pov']) 
            sequence['camera'] = env_state['camera'] 
            observation = reward, done, info, env_state, action  
            ## store for model fitting 
            self.store_observation(observation) 
            ## store for output 
            self.total_iters += 1 
            simulation_results.append((reward, done, self.total_iters)) 

            if iter_idx > BATCH_SIZE+50 and iter_idx % fit_freq == 0: 
                loss_f, halt_method, mean_reward = self.optimize(max_iter=10, batch_size=self.batch_size, l2_regularizer=l2_regularizer, log1p_regularizer=log1p_regularizer) 
                update_target_model() 
                self.eval() 
                iters.set_description(f'loss: {loss_f}, mean_rwrd: {mean_reward}, ttl rwd: {last_total_reward}') 
                pass 

            if done: 
                pov, camera = env.reset() 
                env_state = self.__format_observation(pov, camera) 
                sequence = {'pov': [env_state['pov']]*self.short_term_memory_length, 'camera': env_state['camera']}  
                ## store initial observation 
                obesrvation = 0, False, {}, env_state, 0  
                self.total_rewards.append(last_total_reward) 
                last_total_reward = 0 
                n_restarts += 1 
                pass 
            pass 
        #env.close() 

        if manual_play: 
            pygame.display.quit() 
            pygame.quit() 
            pass 

        if plot_rewards: 
            plt.plot(self.total_rewards) 
            plt.show()
        return simulation_results 

if __name__ == '__main__':
    m = Model()
    m.simulate() 
