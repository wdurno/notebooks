## cartpole with actor crtic via ddpg 

import random 
import numpy as np 
import gym 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from ssr_agent import SSRAgent, ReplayBuffer, Object 

GAMMA = .99 

class Actor(SSRAgent):
    def __init__(self, state_dim, action_dim, target_critic=None, replay_buffer=None, ssr_rank=2): 
        super(Actor, self).__init__(replay_buffer=replay_buffer, ssr_rank=ssr_rank) 
        self.fc1 = nn.Linear(state_dim, 256) 
        self.fc2 = nn.Linear(256, 128) 
        self.fc3 = nn.Linear(128, action_dim) 
        self.buffer = Object() 
        self.buffer.target_critic = target_critic ## buffer stops param sharing (ie. in `state_dict`) 
        ssr_param_iterable = [p for p in self.parameters()]
        if target_critic is not None: 
            ssr_param_iterable.extend([p for p in self.buffer.target_critic.parameters()]) 
            pass 
        self.ssr_param_iterable = ssr_param_iterable 
        pass 
    def forward(self, state): 
        x = torch.relu(self.fc1(state)) 
        x = torch.relu(self.fc2(x)) 
        action = torch.tanh(self.fc3(x)) 
        return action
    def loss(self, transitions, target_critic=None): 
        if target_critic is None: 
            target_critic = self.buffer.target_critic 
        if target_critic is None: 
            raise ValueError('ERROR: this model has no associated critic!') 
        return -torch.sum(target_critic(transitions.state, self(transitions.state))) ## log lik, not average log lik 
    pass 

class Critic(SSRAgent):
    def __init__(self, state_dim, action_dim, target_actor=None, target_critic=None, replay_buffer=None, ssr_rank=2): 
        super(Critic, self).__init__(replay_buffer=replay_buffer, ssr_rank=ssr_rank) 
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.buffer = Object() 
        self.buffer.target_actor = target_actor 
        self.buffer.target_critic = target_critic 
        self.ssr_param_iterable = [p for p in self.parameters()]  
        pass 
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value
    def loss(self, transitions, target_actor=None, target_critic=None): 
        if target_actor is None: 
            target_actor = self.buffer.target_actor 
        if target_critic is None: 
            target_critic = self.buffer.target_critic 
        if target_actor is None or target_critic is None: 
            raise ValueError('ERROR: target model missing!') 
        # Calculate the target Q-values
        target_Q = target_critic(transitions.next_state, target_actor(transitions.next_state))
        target_Q = (1 - transitions.done.int()) * target_Q.clone().detach() * GAMMA + transitions.reward 
        # Calculate the current Q-values
        current_Q = self(transitions.state, transitions.action) 
        # Calculate the critic loss
        return torch.sum((target_Q - current_Q).pow(2)) ## log lik, not average log lik 
    pass 

# Define the environment 
env = gym.make('CartPole-v1') 

def simulate(iters=5000, mem_iters=None, buffer_min=1000): 
    '''Runs an Actor Crtiic experiment for `iters` iterations. 
    Returns a list of cumulative rewards. 
    Memorizes every `mem_iters` iterations, or never if `None`. 
    ''' 
    output_tuples = [] ## (cumulative_reward, iter_idx)  

    # Create the replay buffer 
    replay_buffer = ReplayBuffer(capacity=100000) 
    
    # Create the actor and critic networks
    target_critic = Critic(state_dim=4, action_dim=1) 
    target_actor = Actor(state_dim=4, action_dim=1) 
    critic = Critic(state_dim=4, action_dim=1, target_critic=target_critic, target_actor=target_actor, replay_buffer=replay_buffer) 
    actor = Actor(state_dim=4, action_dim=1, target_critic=critic, replay_buffer=replay_buffer) 
    
    # Define the optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-5)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
    
    episode_idx = 0 
    iter_idx = 0 
    # Train the agent
    while iter_idx < iters: 
        episode_idx += 1 
        state, _ = env.reset() 
        target_actor.load_state_dict(actor.state_dict()) 
        target_critic.load_state_dict(critic.state_dict()) 
        
        cumulative_reward = 0 
        while True: 
            actor.eval() 
            critic.eval() 
            target_actor.eval() 
            target_critic.eval() 

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
                critic.train() 
                pi_B = 1. - critic.optimal_lambda() 
                critic_loss = pi_B * critic.loss(transitions)/256 + critic.ssr() 
    
                # Update the critic network 
                critic_optimizer.zero_grad() 
                critic_loss.backward() 
                critic_optimizer.step() 
                
                # Calculate the actor loss 
                critic.eval() 
                actor.train() 
                pi_B = 1. - actor.optimal_lambda() 
                actor_loss = pi_B * actor.loss(transitions)/256 #+ actor.ssr() ## TODO can I use this? 
    
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
