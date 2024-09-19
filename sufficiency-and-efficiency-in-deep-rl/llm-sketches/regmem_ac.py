import random 
import numpy as np 
from math import prod 
from tqdm import tqdm 
import gc 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from typing import Optional, Tuple  
from ssr_agent import SSRAgent 
from replay_buffer import Object, ReplayBuffer 
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline 

## Hyperparameters 
## RL 
GAMMA = .99 
## LLM 
MAX_LENGTH=1024
MAX_RESPONSE=100
AVG_TOKENS_PER_WORD=3
MODEL='gpt2' 
GPU = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
CPU = torch.device('cpu') 

## CONSTANTS 
ACTION_DIM = 768 # == token dimension 
TOKENIZER_EOS_TOKEN_ID = 50257 - 1 

class Actor(SSRAgent):
    def __init__(self, target_critic=None, replay_buffer=None, ssr_rank=2, device=GPU): 
        super(Actor, self).__init__(replay_buffer=replay_buffer, ssr_rank=ssr_rank) 
        self.device = device 
        self.llm = GPT2LMHeadModel.from_pretrained(MODEL, pad_token_id=TOKENIZER_EOS_TOKEN_ID).to(device) ## final hidden layer are action space  
        self.buffer = Object() 
        self.buffer.target_critic = target_critic ## buffer stops param sharing (ie. in `state_dict`) 
        self.p_gpt2_loss = -1. ## proportional weight given to gpt2 loss relative to actor critic loss ## TODO move to SSRAgent.loss and .memorize with **kwargs 
        pass 
    def forward(self, state): 
        action = last_hidden(self.llm, state) ## [n, 768] 
        return action
    def loss(self, transitions, target_critic=None): 
        if transitions.state.device != self.device: 
            transitions.state = transitions.state.to(self.device) 
            transitions.next_state = transitions.next_state.to(self.device)
            transitions.reward = transitions.reward.to(self.device) 
            transitions.done = transitions.done.to(self.device) 
            transitions.action = transitions.action.to(self.device) 
        if target_critic is None: 
            target_critic = self.buffer.target_critic 
        if target_critic is None: 
            raise ValueError('ERROR: this model has no associated critic!') 
        loss = -torch.sum(target_critic(transitions.state, self(transitions.state))) ## log lik, not average log lik 
        if self.p_gpt2_loss > 0.: 
            ## the actor must retain meaningful text generation capabilities 
            loss = (1 - self.p_gpt2_loss) * loss + self.p_gpt2_loss * self.llm(transitions.state, labels=transitions.state).loss 
        return loss  
    pass 

class Critic(SSRAgent):
    def __init__(self, action_dim=ACTION_DIM, target_actor=None, target_critic=None, replay_buffer=None, ssr_rank=2, device=GPU): 
        super(Critic, self).__init__(replay_buffer=replay_buffer, ssr_rank=ssr_rank) 
        self.device = device 
        self.llm = GPT2LMHeadModel.from_pretrained(MODEL, pad_token_id=TOKENIZER_EOS_TOKEN_ID).to(device) ## last hidden state + action -> Q 
        self.fc1 = nn.Linear(768 + ACTION_DIM, 32, device=device) ## last_hidden dim + action dim  
        self.fc2 = nn.Linear(32, 16, device=device) 
        self.fc3 = nn.Linear(16, 1, device=device) 
        self.buffer = Object() 
        self.buffer.target_actor = target_actor 
        self.buffer.target_critic = target_critic 
        pass 
    def forward(self, state, action):
        x = last_hidden(self.llm, state) 
        x = torch.cat([x, action], dim=1) 
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value
    def loss(self, transitions, target_actor=None, target_critic=None): 
        if transitions.state.device != self.device: 
            transitions.state = transitions.state.to(self.device)
            transitions.next_state = transitions.next_state.to(self.device)
            transitions.reward = transitions.reward.to(self.device)
            transitions.done = transitions.done.to(self.device) 
            transitions.action = transitions.action.to(self.device) 
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
        loss = torch.sum((target_Q - current_Q).pow(2)) ## log lik, not average log lik 
        return loss 
    pass 

class GPT2ActorCritic(): 
    def __init__(self, device=GPU, ssr_rank=2): 
        self.device = device 
        self.replay_buffer = ReplayBuffer() 
        self.actor = Actor(replay_buffer=self.replay_buffer, ssr_rank=ssr_rank, device=self.device) 
        self.critic = Critic(replay_buffer=self.replay_buffer, ssr_rank=ssr_rank, device=self.device) 
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5) 
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3) 
        pass 
    def save(self, path): 
        self.actor.save(path+'.actor') 
        self.critic.save(path+'.critic') 
        pass 
    def load(self, path): 
        self.actor.load(path+'.actor') 
        self.critic.load(path+'.critic') 
        pass 
    def load_transitions(self, path): 
        self.replay_buffer.load(path) 
        pass 
    def fit_iter_critic(self, batch_size=256, p_gpt2_loss=-1.): 
        if len(self.replay_buffer) < batch_size: 
            warnings.warn('skipping fit_iter due to short replay_buffer!') 
            pass 
        critic_grad = 0. 
        for _ in range(batch_size): 
            ## Sample a batch of transitions from the replay buffer 
            transitions = self.replay_buffer.sample(batch_size=1, device=GPU) ## 1 at a time, then average 
            ## Calculate the critic loss 
            self.critic.train()
            pi_B = 1. - self.critic.optimal_lambda()
            critic_loss = pi_B * self.critic.loss(transitions)/batch_size + self.critic.ssr() 
            ## Update the critic network 
            self.critic_optimizer.zero_grad()
            critic_loss.backward() 
            critic_grad += self.__vectorize_grad(self.critic) 
            pass 
        ## average grads 
        critic_grad /= batch_size 
        ## apply grads 
        self.__insert_grad_vec(self.critic, critic_grad) 
        self.critic_optimizer.step() 
        pass 
    def fit_iter_actor(self, batch_size=256, p_gpt2_loss=-1.):
        if len(self.replay_buffer) < batch_size:
            warnings.warn('skipping fit_iter due to short replay_buffer!')
            pass
        actor_grad = 0.
        for _ in range(batch_size):
            ## Sample a batch of transitions from the replay buffer
            transitions = self.replay_buffer.sample(batch_size=1, device=GPU) ## 1 at a time, then average
            ## Calculate the actor loss
            self.critic.eval()
            self.actor.train()
            pi_B = 1. - self.actor.optimal_lambda()
            actor_loss = pi_B * self.actor.loss(transitions)/batch_size + self.actor.ssr()
            if p_gpt2_loss > 0.:
                self.actor.p_gpt2_loss = p_gpt2_loss
                pass
            ## Update the actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad += self.__vectorize_grad(self.actor)
            pass
        ## average grads
        actor_grad /= batch_size
        ## apply grads
        self.__insert_grad_vec(self.actor, actor_grad)
        self.actor_optimizer.step()
        pass
    def fit_loop(self, n_iters=100, batch_size=256, p_gpt2_loss=-1, progress_bar=True): 
        ## set target models 
        self.critic.buffer.target_actor = Actor(replay_buffer=self.replay_buffer, device=self.device)
        self.critic.buffer.target_critic = Critic(replay_buffer=self.replay_buffer, device=self.device) 
        self.critic.buffer.target_actor.eval() 
        self.critic.buffer.target_critic.eval() 
        self.actor.buffer.target_critic = self.critic 
        ## loop over fit iters 
        for _ in tqdm(range(n_iters), disable=not progress_bar): 
            ## move targets to GPU 
            self.critic.buffer.target_actor = self.critic.buffer.target_actor.to(self.device) 
            self.critic.buffer.target_critic = self.critic.buffer.target_critic.to(self.device) 
            ## update target models 
            self.critic.buffer.target_actor.load_state_dict(self.actor.state_dict()) 
            self.critic.buffer.target_critic.load_state_dict(self.critic.state_dict()) 
            ## iterate critic 
            self.fit_iter_critic(batch_size=batch_size, p_gpt2_loss=p_gpt2_loss) ## scope end activates garbage collection 
            ## move targets back to CPU, clearing space from GPU 
            self.critic.buffer.target_actor = self.critic.buffer.target_actor.to(CPU) 
            self.critic.buffer.target_critic = self.critic.buffer.target_critic.to(CPU) 
            ## iterate actor 
            self.fit_iter_actor(batch_size=batch_size, p_gpt2_loss=p_gpt2_loss) 
            pass 
        ## clear memory of target models  
        self.actor.buffer.target_critic = None 
        self.critic.buffer.target_actor = None 
        self.critic.buffer.target_critic = None 
        pass 
    def memorize(self): 
        ## actor 
        self.actor.buffer.target_critic = self.critic 
        self.actor.buffer.target_critic.eval() 
        self.actor.train() 
        self.actor.memorize() 
        self.actor.buffer.target_critic = None 
        ## critic 
        self.critic.buffer.target_actor = self.actor 
        self.critic.buffer.target_critic = self.critic 
        self.critic.buffer.target_actor.eval() 
        self.critic.train() 
        self.critic.memorize() 
        self.critic.buffer.target_actor = None 
        self.critic.buffer.target_critic = None 
        ## clear buffer of memorized transitions 
        self.replay_buffer.clear() 
        pass 
    def pad_gpt_memory(self, info=100., samples=100.): 
        self.__pad_gpt_memory_for_model(self.actor, info=info, samples=samples) 
        self.__pad_gpt_memory_for_model(self.critic, info=info, samples=samples) 
        pass 
    def __pad_gpt_memory_for_model(self, model, info=100., samples=100.): 
        ## build padded diagonal 
        vec = [] 
        for p in model.named_parameters(): 
            if p[0].startswith('llm.transformer'): 
                vec.append(torch.ones(p[1].reshape([-1,1]).shape)) 
            else: 
                vec.append(torch.zeros(p[1].reshape([-1,1]).shape)) 
                pass 
            pass 
        ## get remaining values 
        if model.ssr_center is None: 
            ## init vals 
            model.ssr_residual_diagonal = info * torch.cat(vec).to(self.device)   
            model.ssr_n = samples 
            model.ssr_model_dimension = model.ssr_residual_diagonal.shape[0] 
            model.ssr_low_rank_matrix = torch.zeros([model.ssr_model_dimension, model.ssr_rank]).to(self.device) 
        else: 
            ## add info 
            model.ssr_residual_diagonal += info * torch.cat(vec).to(self.device) 
            model.ssr_n += samples 
            pass 
        model.ssr_center = model.get_param().clone().detach() 
        pass 
    @staticmethod 
    def __vectorize_grad(model): 
        return torch.cat([p.grad.reshape([-1]) for p in model.parameters()]) 
    @staticmethod 
    def __insert_grad_vec(model, vec): 
        ptr = 0 
        for p in model.parameters(): 
            n = prod(p.shape) 
            p.grad = vec[ptr:(ptr+n)].reshape(p.shape) 
            ptr += n 
            pass 
        pass 
    pass 

def last_hidden( 
    model,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None):
    r"""Get the last hidden state after processing a sequence. 
    COPIED AND MODIFIED FROM `https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/gpt2/modeling_gpt2.py#L1049`.
    labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
        config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
        `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    return_dict = return_dict if return_dict is not None else True
    transformer_outputs = model.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    return transformer_outputs['last_hidden_state'][:,-1,:] ## dim: [n, 768] 

