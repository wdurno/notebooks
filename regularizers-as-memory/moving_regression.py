## cartpole with actor crtic via ddpg 

import random 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from ssr_agent import SSRAgent, ReplayBuffer 

class Env(): 
    'Generates Gaussian observations around t/n.' 
    def __init__(self, n=10, state_dim=5): 
        self.t = 0 
        self.n = n 
        means = torch.zeros([state_dim, 1]) 
        self.weights = torch.normal(means, 1.) 
        self.bias = torch.normal(torch.tensor([[0.]]), 1.) 
        pass 
    def sample(self): 
        x = torch.normal(torch.zeros([self.n, state_dim]), 1.) 
        y = x.matmul(x, self.weights) + self.bias 
        return x, y 
    def time_step(self): 
        self.bias += 1. / self.n 
        pass 
    pass 

class MovingRegressor(SSRAgent):
    def __init__(self, state_dim=5, replay_buffer=None, ssr_rank=2): 
        super(Critic, self).__init__(replay_buffer=replay_buffer, ssr_rank=ssr_rank) 
        self.fc1 = nn.Linear(state_dim, 10)
        self.fc2 = nn.Linear(10, 1)
        pass 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        y_hat = self.fc2(x) 
        return y_hat 
    def loss(self, transitions): 
        y_hat = self.forward(transitions.x) 
        return torch.sum((transitions.y - y_hat).pow(2)) ## log lik, not average log lik 
    pass 

def simulate(iters=1000, mem_iters=None, buffer_min=100): 
    'regress a moving target efficiently' 
    ## init a fresh env 
    env = Env() 
    ## for spark dataframes 
    output_tuples = [] ## (cumulative_reward, iter_idx)  
    # Create the replay buffer 
    replay_buffer = ReplayBuffer(capacity=100000) 
    # Create the model 
    model = MovingRegressor() 
    # Define the optimizer 
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ## init fitting loop 
    episode_idx = 0 
    iter_idx = 0 
    # fit the model 
    while iter_idx < iters: 
        episode_idx += 1 
        state, _ = env.reset() ## TODO pick-up here  
        target_actor.load_state_dict(actor.state_dict()) 
        target_critic.load_state_dict(critic.state_dict()) 
        
        cumulative_reward = 0 
        while True: 
            action = actor(torch.tensor(state)) 
            if np.random.binomial(1, max(0,50-episode_idx)/50) > 0: 
                ## random action 
                action = torch.tensor(np.random.uniform(low=-1., high=1.)).reshape([1]) 
                pass 
            
            action_p = action.item() * .5 + .5 
            action_int = np.random.binomial(1, action_p) ## must be 0 or 1 
            next_state, reward, done, _, _ = env.step(action_int) 
    
            replay_buffer.add(state, action, reward, next_state, done) 
            
            cumulative_reward += reward 
            output_tuples.append((cumulative_reward, iter_idx)) 
    
            if len(replay_buffer) > 256: 
                # Sample a batch of transitions from the replay buffer 
                transitions = replay_buffer.sample(batch_size=256) 
    
                # Calculate the critic loss 
                critic_loss = critic.loss(transitions) + critic.ssr() 
    
                # Update the critic network 
                critic_optimizer.zero_grad() 
                critic_loss.backward() 
                critic_optimizer.step() 
                
                # Calculate the actor loss 
                ## TODO cannot simply de-activate this SSR, o.w. data is lost. MUST repair it. 
                actor_loss = actor.loss(transitions) + actor.ssr()  
    
                # Update the actor network 
                actor_optimizer.zero_grad() 
                actor_loss.backward() 
                actor_optimizer.step() 

                if mem_iters is not None: 
                    if len(replay_buffer) >= buffer_min + mem_iters: 
                        actor.memorize(mem_iters) 
                        critic.memorize(mem_iters) 
                        replay_buffer.clear(mem_iters) 
                        pass 
                    pass 
                pass
    
            state = next_state 
            iter_idx += 1 
    
            if done or iter_idx > iters:
                ## start new episode 
                break
            pass 
        pass 
    return output_tuples  
