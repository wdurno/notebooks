
import torch
import torch.nn as nn

from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline 

from replay_buffer import ReplayBuffer 
from lanczos import l_lanczos, combine_krylov_spaces

class Object(object):
    pass

class GPT2SSRAgent(GPT2LMHeadModel): 
    '''
    A GPT2 language model modified to optimally learn how to keep a user happy. 
    '''
    def __init__(self, config, replay_buffer=None, ssr_rank=2): 
        super(GPT2SSRAgent, self).__init__(config) 
        self.ssr_rank = ssr_rank
        self.ssr_low_rank_matrix = None ## =: A
        self.ssr_residual_diagonal = None ## =: resid
        self.ssr_center = None
        self.ssr_prev_center = None
        self.ssr_n = None
        ## N * Fisher Information \approx AA^T + resid
        self.ssr_model_dimension = None 
        self.replay_buffer = replay_buffer  
        pass 
    def __actor_critic_forward(
        self,
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
        r"""
        COPIED AND MODIFIED FROM `https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/gpt2/modeling_gpt2.py#L1049`.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else True
        transformer_outputs = self.transformer(
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
        return transformer_outputs['last_hidden_state'] ## [-1, 1024, 768] = [n, sequence length, embedding dim] 
    def lm_loss(self, transitions): 
        ## TODO get tokenized transcripts t 
        n = t.shape[0] 
        return n * self(t, labels=t).loss ## log lik, not average log lik 
    def loss(self, transitions): ## = rl_loss
        raise NotImplementedError('ERROR: loss not implemented!') 
    def memorize(self, n=None): 
        'memorize oldest `n` transitions, or all if `n is None`' 
        self.ssr_prev_center = self.ssr_center 
        self.ssr_center = self.__get_param() ## elliptical centroid 
        if self.ssr_model_dimension is None: 
            self.ssr_model_dimension = self.ssr_center.shape[0] 
            pass 
        ssr_low_rank_matrix, ssr_residual_diagonal = l_lanczos(self.__get_get_grad_generator(n), self.ssr_rank, self.ssr_model_dimension, calc_diag=True) 
        if self.ssr_low_rank_matrix is None: 
            ## first memorization 
            self.ssr_low_rank_matrix = ssr_low_rank_matrix 
            self.ssr_residual_diagonal = ssr_residual_diagonal 
            self.ssr_n = n 
        else: 
            ## combine with previous memories 
            self.ssr_low_rank_matrix = combine_krylov_spaces(self.ssr_low_rank_matrix, ssr_low_rank_matrix) 
            self.ssr_residual_diagonal += ssr_residual_diagonal 
            self.ssr_n += n 
            pass 
        pass 
    def ssr(self, lmbda=None): 
        '''Get the ssr regularizer. If `lmbda is None`, `lmbda` will be set to 1 when `self.ssr_prev_center is None`, 
        otherwise `lmbda` will be the approximately optimal `n_A` value.'''
        if self.ssr_low_rank_matrix is None: 
            return 0. 
        if lmbda is None: 
            lmbda = 1. 
            pass 
        p = self.__get_param() 
        p0 = self.ssr_center 
        d = p - p0 
        A = self.ssr_low_rank_matrix 
        res = self.ssr_residual_diagonal 
        dTA = d.transpose(0,1).matmul(A) 
        ATd = dTA.transpose(0,1) 
        dTresd = (d.transpose(0,1) * res).matmul(d) 
        ssr_sum = dTA.matmul(ATd) + dTresd 
        if lmbda is None: 
            if self.ssr_prev_center is None: 
                lmbda = 1. 
            else: 
                ## approximately optimal lambda 
                dt = p0 - self.ssr_prev_center 
                dtTA = dt.transpose(0,1).matmul(A) 
                ATdt = dtTA.transpose(0,1) 
                dtTresdt = (dt.transpose(0,1) * res).matmul(dt) 
                lmbda = dtTA.matmul(ATdt) + dtTresdt 
                pass 
            pass 
        ## no average, because we're using the log lik  
        ## also, `ssr_n` becomes vague as we iteratively apply optimal lambda 
        return lmbda * .5 * ssr_sum 
    def __get_param(self): 
        return torch.cat([p.reshape([-1, 1]) for p in self.parameters()], dim=0) 
    def __get_get_grad_generator(self, n=None): 
        ## The double get hides `self` in a function context,  
        ## packaging `get_grad_generator` for calling without 
        ## the SSRAgent instance.  
        if n is None: 
            n = self.replay_buffer.n 
            pass 
        def get_grad_generator(): 
            'l-Lanczos alg uses grad at least `ssr_rank` times' 
            def grad_generator(): 
                self.eval() 
                for idx in range(n): 
                    transition = self.replay_buffer.sample(idx_list=[idx]) 
                    loss = self.loss(transition) 
                    loss.backward() 
                    grad_vec = torch.cat([p.grad.reshape([-1]) for p in self.parameters()]) 
                    yield grad_vec 
                    pass
                pass 
            return grad_generator 
        return get_grad_generator 
    pass
    pass 

class GPT2Actor(GPT2SSRAgent): ## transformer output is the action  
    def __init__(self, config, critic=None):
        super().__init__(config)
        self.buffer = Object() 
        self.buffer.critic = critic ## buffer stops param sharing (ie. in `state_dict`) 
        pass
    def rl_mu(self, tokenized_transcripts):  
        'final transformer output is the action' 
        x = self.__actor_critic_forward(tokenized_transcripts) 
        ## extract last action  
        return x[:, -1, :] ## [-1, 768] 
    def loss(self, transitions):
        if self.buffer.critic is None: 
            raise ValueError('ERROR: this model has no associated critic!') 
        return -torch.sum(self.buffer.critic.rl_q(transitions.state, self.rl_mu(transitions.state))) ## log lik, not average log lik 
    pass 

class GPT2Critic(GPT2SSRAgent):  
    def __init__(self, config, target_actor=None, target_critic=None, gamma=.99): 
        super().__init__(config) 
        self.gamma = gamma 
        self.buffer = Object()
        self.buffer.target_actor = target_actor
        self.buffer.target_critic = target_critic
        ## hidden state inputs are shaped [-1, 1024, 768]
        self.rl_conv1 = nn.Conv1d(768, 256, kernel_size=7, stride=5) ## shape is [-1, 256, 204]
        self.rl_conv2 = nn.Conv1d(256, 16, kernel_size=7, stride=5) ## shape is [-1, 16, 40]
        self.rl_linear = nn.Linear(16*40, 1)
        pass

    def rl_q(self, tokenized_transcripts, action):  
        x = self.__actor_critic_forward(tokenized_transcripts) 
        x[:, -1, :] = action ## overwrite final transformer output with action 
        x = torch.relu(self.rl_conv1(x))
        x = torch.relu(self.rl_conv2(x))
        x = x.flatten(start_dim=1)
        return self.rl_linear(x) 

    def loss(self, transitions): 
        if self.buffer.target_actor is None or self.buffer.target_critic is None:
            raise ValueError('ERROR: target model missing!')
        # Calculate the target Q-values
        target_Q = self.buffer.target_critic.rl_q(transitions.next_state, self.buffer.target_actor.rl_mu(transitions.next_state))
        target_Q = (1 - transitions.done.int()) * target_Q.clone().detach() * self.gamma + transitions.reward
        # Calculate the current Q-values
        current_Q = self.rl_q(transitions.state, transitions.action) 
        # Calculate the critic loss
        return torch.sum((target_Q - current_Q).pow(2)) ## log lik, not average log lik 
    pass 


