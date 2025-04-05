## We'll use an advantage-weighted approach because it allows shared parameters
## between the policy net and the advantage function unlike Deterministic Policy Gradients (DPG). 
## I've enjoyed DPG for its theoretical elegance, but RAM is expensive. 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers.models.llama.modeling_llama import LlamaForCausalLM 

## Name of the pretrained model 
MODEL_NAME = "meta-llama/Meta-Llama-3-8B" 

## Load config 
CONFIG = AutoConfig.from_pretrained(MODEL_NAME) 
CONFIG.output_hidden_states = True  ## important for value head! 

## Initialize tokenizer for future use 
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME) 

## TODO I can't just save state dicts, I need to use these functions to get configs 
## model.save_pretrained("dualhead-llama3") 
## tokenizer.save_pretrained("dualhead-llama3") 

class LlamaForCausalLMWithValueHead(LlamaForCausalLM): 
    def __init__(self, config=CONFIG):
        super().__init__(config)
        self.value_head = nn.Linear(config.hidden_size, 1)  ## regression target 
        self.post_init()
        pass 
    @staticmethod
    def load_pretrained(): 
        '''Use this to init with pre-trained parameters from Hugging Face.
        '''
        model = LlamaForCausalLMWithValueHead.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            ignore_mismatched_sizes=True  ## prevents crashing due to new head
            ) 
        model.value_head.reset_parameters() ## re-init value_head params 
        return model 
    def forward(
            self,
            transitions, 
            attention_mask=None, 
            old_model=None, ## I can't just cache old_log_probs because I'm running off-policy experiments 
            value_coef=.5, 
            **kwargs
            ):
        '''Predicts probability logits and expected RL value. 
        If a `transitions` is a full tuple and (`old_model`, `value_coef`) is provided, then it also calculates a loss. 
        inputs: 
        - transisions: can be one of two things... 
          - if just a tensor, then its a matrix of input IDs of shape [batch_size, sequence length] 
          - if a tuple, then it has these entries... 
            1. `state`: a matrix of input IDs 
            2. `next_state`: a matrix of input IDs immediately following `state` 
            3. `action`: action probability logits that originally lead to `next_state`, shaped [batch_size, 1] 
            4. `reward`: a matrix of rewards, shaped [batch_size, 1]
            5. `done`: a matrix of boolean `done` flags, shaped [batch_size, 1] 
        - `old_model`: a similar-but-different `LlamaForCausalLMWithValueHead` instance 
        - `value_coef`: the loss weight given to RL value. For example, 0. implies a pure LLM loss. 
        outputs: 
        - `p_logits`: soft max logits encoding the probability of which token comes next 
        - `values`: the predicted RL value of the next state 
        - `loss`: a combined LLM and RL loss 
        '''
        ## unpack transitions 
        if type(transitions) == tuple: 
            state = input_ids = transitions[0] 
            next_state = transitions[1] 
            action = transitions[2] 
            reward = transitions[3] 
            done = transitions[4] 
        else: 
            state = input_ids = transitions 
            next_state = None 
            action = None 
            reward = None 
            done = None 
            pass 
        ## predict on current state 
        p_logits, values = self.__get_logits_and_values(input_ids=input_ids, attention_mask=attention_mask, **kwargs) 
        ## if viable, calculate the loss 
        loss = None 
        if next_state is not None and old_model is not None and value_coef is not None:  
            old_p_logits, next_values = old_model.__get_logits_and_values(input_ids=next_state, attention_mask=None, **kwargs) 
            llm_loss, rl_loss = LlamaForCausalLMWithValueHead.__compute_ppo_loss_nonsequential(p_logits, values, action, reward, next_values, old_p_logits, done) 
            loss = llm_loss + value_coef * rl_loss 
            pass 
        # return (probability logits, expected values, loss) 
        return p_logits, values, loss 

    def __get_logits_and_values(self, input_ids, attention_mask, **kwargs): 
        ## apply backbone and get probability logits 
        output = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs) 
        ## Reuse last hidden states
        hidden_states = output.hidden_states[-1] if self.config.output_hidden_states else output[0]
        ## Use the same final hidden state for value prediction
        values = self.value_head(hidden_states).squeeze(-1)  ## shape: [batch, seq] 
        ## returns (probability logits, expected values) 
        return output.logits, values 
    @staticmethod
    def __compute_ppo_loss_nonsequential(
            logits, values, actions, rewards, next_values, old_log_probs, done, 
            gamma=0.99, clip_epsilon=0.2, value_coef=0.5
            ):
        '''Computes a PPO loss for a dual head but without the regularizer. 
        inputs: ## TODO 
        outputs: ## TODO 
        '''
        ## break differentiation graph 
        old_log_probs = old_log_probs.clone().detach() 
        ## Compute current log probs 
        log_probs = F.log_softmax(logits, dim=-1) 
        action_log_probs = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1) 
        ## Advantage estimate: TD(0) 
        target_values = rewards + gamma * next_values * (1. - done) 
        advantages = (target_values - values).detach()  ## no gradient through targets 
        ## Policy loss (PPO clip) 
        ratios = torch.exp(action_log_probs - old_log_probs) 
        unclipped = ratios * advantages 
        clipped = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages 
        policy_loss = -torch.mean(torch.min(unclipped, clipped))  ## maximize advantage 
        ## Value loss (TD error)
        value_loss = F.mse_loss(values, target_values)
        ## Total loss
        total_loss = policy_loss + value_coef * value_loss
        return total_loss, policy_loss, value_loss, advantages 
