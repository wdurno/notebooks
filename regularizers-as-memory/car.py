import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import randrange, sample 
import gym 
from tqdm import tqdm 
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.patches import Circle 
import copy 
import os 
from PIL import Image 
from lanczos import l_lanczos, combine_krylov_spaces 
try: 
    from car_env.car_client import PiCarEnv 
    from car_env.constants import N_ACTIONS, N_CAMERA_DIRECTIONS  
except ImportError: 
    ## used in spark-k8s scaling due to deep conflicts 
    print('WARNING: could not load `car_env` - mitigating! Simulation will fail!') 
    import pickle 
    N_ACTIONS = 8
    N_CAMERA_DIRECTIONS = 4 
    SCREEN_HEIGHT = 2*60 
    BALL_SIZE_MIN = SCREEN_HEIGHT/10 
    BALL_SIZE_MAX = SCREEN_HEIGHT/3 
    class PiCarEnv: 
        @staticmethod 
        def load_memory(filepath): 
            with open(filepath, 'rb') as f: 
                data = pickle.load(f) 
                pass 
            unstacked_arrays = PiCarEnv.__unstack(data[0]) 
            unstacked_data = (
                    unstacked_arrays, ## images 
                    data[1], ## actions 
                    data[2], ## ball x locations 
                    data[3], ## ball radii 
                    data[4]  ## camera positions 
                    ) 
            return unstacked_data 
        @staticmethod 
        def __unstack(a, axis = 0):
            return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis = axis)] 
        @staticmethod 
        def get_reward(r): 
            ball_radius = r 
            ## reward being close to the ball 
            if ball_radius >= BALL_SIZE_MIN and ball_radius <= BALL_SIZE_MAX: 
                return(ball_radius - BALL_SIZE_MIN ) / (BALL_SIZE_MAX - BALL_SIZE_MIN) 
            elif ball_radius > BALL_SIZE_MAX: 
                ## slight punishment for being too close 
                return min(1. + (BALL_SIZE_MAX - ball_radius)/20., -.1) 
                pass 
            return 0.
        pass 
    pass 

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
ENV_NAME = '' ## not used 
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
            env_name=ENV_NAME, ## not used 
            hessian_sum=None, 
            hessian_sum_low_rank_half=None, 
            hessian_denominator=None, 
            hessian_center=None, 
            hessian_residual_variances=None, 
            observations=None,
            total_iters=0,
            mean_rewards=None, 
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
        self.env_name = env_name 
        self.hessian_sum = hessian_sum 
        self.hessian_denominator = hessian_denominator
        self.hessian_center = hessian_center 
        self.hessian_residual_variances = hessian_residual_variances 
        self.hessian_sum_low_rank_half = hessian_sum_low_rank_half
        self.total_iters = total_iters
        self.mean_rewards = mean_rewards 
        self.regularizing_lambda_function = regularizing_lambda_function 
        ## init CNNs 
        ## 2D CCNs 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=3) ## to (-1, 32, 38, 58) 
        self.conv1_bn = nn.BatchNorm2d(64) 
        self.mp2d1 = nn.MaxPool2d(3, 2) 
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2) ## to (-1, 64, 18, 28) 
        self.conv2_bn = nn.BatchNorm2d(128) 
        self.mp2d2 = nn.MaxPool2d(3, 2) 
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2) ## to (-1, 128, 8, 13) 
        self.conv3_bn = nn.BatchNorm2d(256) 
        ## 1D CNNs over time 
        self.conv6 = nn.Conv1d(320*16, 256, kernel_size=3, stride=2) ## to (-1, 16, 19) 
        self.conv6_bn = nn.BatchNorm1d(256) 
        self.mp1d1 = nn.MaxPool1d(3, 2) 
        ## camera embedding 
        self.embedding = nn.Embedding(self.n_camera_directions, 64) 
        ## FCs 
        self.fc1 = nn.Linear(256*19 + 64, 64) ## + 64 for camera embedding vec 
        self.fc1_bn = nn.BatchNorm1d(64) 
        self.fc2 = nn.Linear(64, n_actions) 
        ## init data structures 
        if observations is None: 
            self.observations = [] 
        else: 
            self.observations = observations 
            pass 
        if mean_rewards is None: 
            self.mean_rewards = [] 
        else: 
            self.mean_rewards = mean_rewards.copy() 
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
        ## ensure everything is on the same device 
        self.to(DEVICE) 
        if self.hessian_sum is not None: 
            self.hessian_sum = self.hessian_sum.to(DEVICE) 
        if self.hessian_center is not None: 
            self.hessian_center = self.hessian_center.to(DEVICE) 
        if self.hessian_residual_variances is not None: 
            self.hessian_residual_variances = self.hessian_residual_variances.to(DEVICE) 
        if self.hessian_sum_low_rank_half is not None: 
            self.hessian_sum_low_rank_half = self.hessian_sum_low_rank_half.to(DEVICE) 
            pass 
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
                env_name=self.env_name,
                hessian_sum=self.hessian_sum.detach().clone() if self.hessian_sum is not None else None, 
                hessian_sum_low_rank_half=self.hessian_sum_low_rank_half, 
                hessian_denominator=self.hessian_denominator, ## this is an int or None, so it is copied via '=' 
                hessian_center=self.hessian_center.detach().clone() if self.hessian_center is not None else None, 
                hessian_residual_variances=self.hessian_residual_variances.detach().clone() if self.hessian_residual_variances is not None else None, 
                observations=self.observations.copy(), 
                total_iters=self.total_iters, 
                mean_rewards=self.mean_rewards.copy(), 
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
        try: 
            candidate_files = os.listdir(dir_path) 
            files_to_load = [f for f in candidate_files if f.endswith('.pkl')] 
            paths = [os.path.join(dir_path, f) for f in files_to_load] 
        except NotADirectoryError: 
            if dir_path.endswith('.pkl'): 
                ## its just one pickle file 
                paths = [dir_path] 
            else: 
                raise Exception('ERROR: `dir_path` must be a directory or a .pkl file!') 
                pass 
            pass 
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

    def convert_observations_to_memory(self, n_eigenvectors = None, krylov_rank = None, krylov_eps=0., disable_tqdm=True): 
        ## convert current observations to a Hessian matrix 
        target_model = self.copy() 
        get_grad_generator = self.__get_get_grad_generator(target_model) 
        if krylov_rank is not None: 
            ## limited-memory lanczos estimation of Fisher Information 
            p = self.get_parameter_vector().detach().shape[0] 
            if self.hessian_sum_low_rank_half is None: 
                ## approximate hessian eigen-space 
                self.hessian_sum_low_rank_half = l_lanczos(get_grad_generator, krylov_rank, p, \
                        device=DEVICE, eps=krylov_eps, disable_tqdm=disable_tqdm) 
                ## calculate hessian diagonal 
                hessian_sum_diagonal = self.__grad_sum(get_grad_generator()).to(DEVICE)  
                self.hessian_residual_variances = hessian_sum_diagonal - self.__outer_product_diagonal(self.hessian_sum_low_rank_half, self.hessian_sum_low_rank_half)   
                self.hessian_residual_variances = self.hessian_residual_variances.maximum(torch.zeros(size=self.hessian_residual_variances.size(), device=DEVICE)) ## clean-up numerical errors, keep it all >= 0   
                ## store total grads sampled 
                self.hessian_denominator = len(self.observations) 
            else: 
                ## Generate new Krylov space and update existing memory with modified Lanczos, combining both 
                krylov_update = l_lanczos(get_grad_generator, krylov_rank, p, device=DEVICE, eps=krylov_eps, disable_tqdm=disable_tqdm) 
                updated_krylov_space = combine_krylov_spaces(self.hessian_sum_low_rank_half, krylov_update, \
                        device=DEVICE, krylov_eps=krylov_eps) 
                ## calc total diagonal variances 
                total_diagonal_variances = self.hessian_residual_variances + self.__outer_product_diagonal(self.hessian_sum_low_rank_half, self.hessian_sum_low_rank_half) 
                total_diagonal_variances += self.__grad_sum(get_grad_generator()).to(DEVICE) ## additional_total_diagonals # TODO just use combined krylov space to avoid this extra loop  
                ## update 
                self.hessian_sum_low_rank_half = updated_krylov_space 
                self.hessian_residual_variances = total_diagonal_variances - self.__outer_product_diagonal(self.hessian_sum_low_rank_half, self.hessian_sum_low_rank_half) 
                self.hessian_residual_variances = self.hessian_residual_variances.maximum(torch.zeros(size=self.hessian_residual_variances.size()).to(DEVICE)) 
                self.hessian_denominator += len(self.observations) 
                pass 
        else: 
            ## full-rank Fisher Information representation 
            ## can be extrememly memory intensive! 
            grad_generator = get_grad_generator() 
            for grad_vec in grad_generator(): 
                outter_product = grad_vec.reshape([-1,1]).matmul(grad_vec.reshape([1,-1])).detach()   
                if self.hessian_sum is None or self.hessian_denominator is None: 
                    self.hessian_sum = outter_product 
                    self.hessian_denominator = 1 
                else: 
                    self.hessian_sum += outter_product 
                    self.hessian_denominator += 1 
                    pass 
                pass
            pass 
        ## center quadratic form on current estimate 
        self.hessian_center = target_model.get_parameter_vector().detach().to(DEVICE) 
        ## wipe observations, and use memory going forward instead 
        self.clear_observations() 
        ## for simulation purposes only - no computational benefit is derived from using this feature 
        if n_eigenvectors is not None: 
            if self.hessian_sum is None: 
                raise ValueError('Full Rank Hessian sum is `None`! Cannot calculate Eigenvectors! Did you use a Krylov estimate?') 
            eigs = torch.linalg.eig(self.hessian_sum) 
            ## extract and truncate 
            vecs = eigs.eigenvectors.real
            vals = eigs.eigenvalues.real
            vecs[:,n_eigenvectors:] = 0 
            vals[n_eigenvectors:] = 0  
            self.hessian_sum = vecs.matmul(torch.diag(vals, device=DEVICE)).matmul(vecs.transpose(0,1)) 
        pass 

    def __get_get_grad_generator(self, target_model): 
        def get_grad_generator(): 
            def grad_generator(): 
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
                        grad_vec = torch.cat([p.grad.reshape([-1]) for p in self.parameters()]) 
                        yield grad_vec 
                        pass 
                    pass 
                pass 
            return grad_generator  
        return get_grad_generator 

    def __grad_sum(self, grad_generator): 
        ''' 
        returns a sum of gradients. 
        divide by n to get a FIM diagonal estimate 
        ''' 
        out = 0. 
        for g in grad_generator(): 
            out += g 
            pass 
        return out  

    def __outer_product_diagonal(self, a, b): 
        'numerically efficient caclulation of diag(a * b^T)' 
        return (a*b).sum(dim=1) 

    def sample_observations(self, batch_size=30, batch=None): 
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
                sample = self.sample_observations(batch_size) 
        if batch is not None: 
            sample = self.sample_observations(batch=batch) 
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
        if (self.hessian_sum is not None or self.hessian_sum_low_rank_half is not None) and \
                self.hessian_denominator is not None and \
                self.hessian_center is not None: 
            t = self.get_parameter_vector() 
            t0 = self.hessian_center 
            ## quadratic form around matrix estimating fisher information 
            if self.hessian_sum is not None:
                ## p X p hessian 
                regularizer = (t - t0).reshape([1, -1]).matmul(self.hessian_sum) 
            elif self.hessian_sum_low_rank_half is not None: 
                ## low-rank hessian 
                regularizer = ((t - t0).reshape([1, -1]).matmul(self.hessian_sum_low_rank_half)).matmul(self.hessian_sum_low_rank_half.transpose(0,1)) 
                pass
            if self.hessian_residual_variances is not None and regularizer is not None: 
                regularizer += (t - t0) * self.hessian_residual_variances 
                pass 
            regularizer = regularizer.matmul((t - t0).reshape([-1, 1]))
            regularizer *= .5 / self.hessian_denominator 
            regularizer = regularizer.reshape([])
        return prediction, target, regularizer 
    
    def get_parameter_vector(self): 
        return nn.utils.parameters_to_vector(self.parameters()) 
    
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
            self.mean_rewards.append(float(mean_reward)) 
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
            tqdm_seconds=.1, l2_regularizer=None, fit_freq=10, manual_play=False, log1p_regularizer=False, \
            memory_write_location='/tmp', eval_pkl_path=None): 
        if manual_play: 
            pygame.init() 
            pygame.display.set_mode((100,100)) 
        if plot_prob_func: 
            plt.plot([self.explore_probability_func(idx) for idx in range(total_iters)]) 
            plt.show() 
            pass
        env = PiCarEnv(host, memory_length=total_iters-1, memory_write_location=memory_write_location)
        pov, camera = env.reset() 
        env_state = self.__format_observation(pov, camera) 
        sequence = {'pov': [env_state['pov']]*self.short_term_memory_length, 'camera': env_state['camera']} 
        ## store initial observation 
        observation = 0, False, {}, env_state, 0  
        self.store_observation(observation) 
        env.memorize() ## saves observations for writing to disk 
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
            env.memorize() ## saves observations for writing to disk 
            ## store for output 
            self.total_iters += 1 
            simulation_results.append((reward, done, self.total_iters)) 

            if iter_idx > BATCH_SIZE+50 and iter_idx % fit_freq == 0: 
                loss_f, halt_method, mean_reward = self.optimize(max_iter=10, batch_size=self.batch_size, l2_regularizer=l2_regularizer, log1p_regularizer=log1p_regularizer) 
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
