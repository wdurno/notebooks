## Implemented with PPO advantage-weighted loss but with SSR 

import json 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch import quantization 
import torch.nn.functional as F 
import torch.distributed as dist 
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, LlamaConfig 
from transformers.models.llama.modeling_llama import LlamaForCausalLM 
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP 

from fsdp_ssr_module import AbstractFsdpSsrModule 
from efficient_replay_buffer import EfficientReplayBuffer as ReplayBuffer 

## Name of the pretrained model 
MODEL_NAME = "meta-llama/Meta-Llama-3-8B" 

class FsdpSsrLlama8B(AbstractFsdpSsrModule): 
    def __init__(self, load_path=None, ssr_rank=2, dt_mean_N=10, learning_rate=1e-4, config=None, rl_coef=None): 
        if load_path is not None and config is None: 
            with open(f"{load_path}/config.json") as f: 
                config = LlamaConfig.from_dict(json.load(f))
                pass 
            pass 
        if config is None: 
            ## contacts Hugging Face for the config 
            config = AutoConfig.from_pretrained(MODEL_NAME) 
            config.output_hidden_states = True  ## important for value head! 
            pass 
        ## default `rl_coef` value is `0.` if not already set 
        config.rl_coef = getattr(config, 'rl_coef', 0.) 
        if rl_coef is not None: 
            ## override with provided value 
            config.rl_coef = rl_coef 
            pass 
        replay_buffer = ReplayBuffer(capacity=1_000_000) 
        if load_path is None: 
            module = LlamaForCausalLMWithValueHead.load_pretrained() ## get a fresh model 
        else: 
            module = LlamaForCausalLMWithValueHead(config=config) 
            pass 
        super(FsdpSsrLlama8B, self).__init__(module=module, replay_buffer=replay_buffer, ssr_rank=ssr_rank, dt_mean_N=dt_mean_N) 
        if load_path is not None: 
            self.load(load_path) 
            pass 
        ## TODO consider saving space with SGD because I have true natural gradients 
        self.optimizer = optim.AdamW([p for p in self.module.parameters() if p.requires_grad], lr=learning_rate) 
        pass 
    @staticmethod 
    def get_tokenizer(): 
        ## TODO add `save` and `load` for tokenizer to avoid duplicative downloads 
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) 
        tokenizer.pad_token = tokenizer.eos_token 
        tokenizer.padding_side = "right" 
        return tokenizer 
    def save_quantized(self, path): 
        'Load model from FSDP cluster and write quantized on rank 0 disk for data generation' 
        path = f'{path}/quantized_model' 
        def _save_quantized(): 
            ## full model is saved prior to loading with quantization 
            self.module.save_pretrained(path) 
            pass 
        with FSDP.summon_full_params(self.module, offload_to_cpu=True, rank0_only=True, writeback=False): 
            if dist.get_rank() == 0: 
                print(f'Writing quantized model to "{path}"...') 
                _save_quantized() 
            pass 
        pass 
    @staticmethod 
    def load_quantized(path): 
        '''Do not run with FSDP! 
        This is for off-cluster data generation. 
        Example data generation: 
        ``` 
        inputs = tokenizer("Hello", return_tensors="pt").to("cuda") 
        outputs = model.generate(**inputs, max_new_tokens=50) 
        print(tokenizer.decode(outputs[0])) 
        ``` 
        ''' 
        ## need that superior quantization for cheap GPUs 
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,  # use `load_in_8bit=True` for 8-bit instead
        #     bnb_4bit_use_double_quant=False,
        #     bnb_4bit_quant_type="nf4",  # nf4 or fp4  
        #     bnb_4bit_compute_dtype=torch.float16
        #     )
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True, 
            llm_int8_has_fp16_weight=False 
            )
        ## Loading into parent class bypasses massive parameters inits and disregards subclass-specific parameters 
        model = LlamaForCausalLM.from_pretrained(
            path, 
            quantization_config=bnb_config,
            device_map="auto"
            )
        model.eval() 
        return model 
    def loss(self, transitions): 
        _, _, loss = self.module(transitions, rl_coef=self.module.config.rl_coef) 
        return loss 
    def save(self, path): 
        if dist.get_rank() == 0: 
            self.module.config.save_pretrained(path) 
            pass 
        super(FsdpSsrLlama8B, self).save(path) 
        pass 
    def load(self, path): 
        'requires correctly-configured modules before loading' 
        super(FsdpSsrLlama8B, self).load(path) 
        self.module.config.from_pretrained(path) 
        pass 
    pass 

class LlamaForCausalLMWithValueHead(LlamaForCausalLM): 
    def __init__(self, config=None): 
        super().__init__(config)
        self.value_head = nn.Linear(config.hidden_size, 1) ## regression target 
        self.post_init() 
        pass 
    @staticmethod 
    def load_pretrained(): 
        '''Use this to init with pre-trained parameters from Hugging Face. 
        '''
        ## contacts Hugging Face for the pre-fit parameters 
        model = LlamaForCausalLMWithValueHead.from_pretrained( ## TODO this'll hammer Hugging Face - pull once, then distribute 
            "meta-llama/Meta-Llama-3-8B", 
            ignore_mismatched_sizes=True  ## prevents crashing due to new head 
            ) 
        model.value_head.reset_parameters() ## re-init value_head params 
        return model 
    def forward(
            self,
            transitions, 
            attention_mask=None, 
            rl_coef=.5, 
            **kwargs
            ):
        '''Predicts probability logits and expected RL value. 
        If a `transitions` is a full tuple and (`old_model`, `value_coef`) is provided, then loss combines LLM & RL loss. 
        inputs: 
        - transisions: can be one of two things... 
          - if just a tensor, then its a matrix of input IDs of shape [batch_size, sequence length] 
          - if a tuple, then it has these entries... 
            1. `state`: a matrix of input IDs 
            2. `next_state`: a matrix of input IDs immediately following `state` 
            3. `action`: action probability logits that originally lead to `next_state`, shaped [batch_size, 1] ### TODO this is wrong! Actions are now currently sequences! 
            4. `reward`: a matrix of rewards, shaped [batch_size, 1] 
            5. `done`: a matrix of boolean `done` flags, shaped [batch_size, 1] 
        - `rl_coef`: the loss weight given to RL value. For example, 0. implies a pure LLM loss. 
        outputs: 
        - `p_logits`: soft max logits encoding the probability of which token comes next 
        - `values`: the predicted RL value of the next state 
        - `loss`: a combined LLM and RL loss OR just LLM loss, depending on inputs 
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
        d = self.__get_logits_and_values(input_ids=input_ids, attention_mask=attention_mask, **kwargs) 
        ## if this loss gets returned, it's just the LLM loss 
        p_logits, values, loss = d['logits'], d['values'], d['loss'] 
        print(f'DEBUG 0: p_logits.shape: {p_logits.shape}, values.shape: {values.shape}') ## TODO verifying... if p_logits 2-rank, below log-softmax likely needs to be a sum. O.w., expect 3-rank. Also, expecting values to be 1-rank. 
        ## if viable, calculate the RL loss 
        if next_state is not None and rl_coef is not None: 
            ## I don't use a previous model here because we use symbolic differentiation. 
            ## Numerical differentiation would require we store several values of theta at great memory cost. 
            ## Instead, it's just sufficient to break the differentiation graph in the right spots. 
            ## This is equivalent to updating from theta_old every time we run optimizer.step(). 
            d = self.__get_logits_and_values(input_ids=next_state, attention_mask=None, **kwargs) 
            old_p_logits, next_values = d['logits'], d['values'] 
            llm_loss, rl_loss = LlamaForCausalLMWithValueHead.__compute_ppo_loss_nonsequential(p_logits, values, action, reward, next_values, old_p_logits, done) 
            rl_loss = llm_loss + .5*rl_loss 
            loss += rl_coef * rl_loss 
            pass 
        # return (probability logits, expected values, loss) 
        return p_logits, values, loss 

    def __get_logits_and_values(self, input_ids, attention_mask, **kwargs): 
        ## apply backbone and get probability logits 
        output = self.model.forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs) 
        ## Reuse last hidden states 
        hidden_states = output.hidden_states[-1] if self.config.output_hidden_states else output[0] 
        ## Use the same final hidden state for value prediction 
        values = self.value_head(hidden_states).squeeze(-1)  ## shape: [batch, seq] 
        ## returns (probability logits, expected values, LLM loss) as dictionary to meet `generate` interface 
        return {'logits': output.logits, 'values': values, 'loss': output.loss} 
    @staticmethod 
    def __compute_ppo_loss_nonsequential( 
            logits, values, actions, rewards, next_values, old_log_probs, done, 
            gamma=0.99, clip_epsilon=0.2 
            ):
        """
        Compute the PPO loss for a batch of assistant responses without assuming sequential generation.

        This function compares the current policy's log probabilities of the sampled actions (assistant responses)
        against a reference policy's frozen log probabilities using the PPO clipped surrogate objective. 
        Each response is treated independently and not as a step in a trajectory.

        Args:
            states (Tensor): Tokenized input prompt sequences (context), shape (batch_size, seq_len). ## TODO this doc string is out-of-date! 
            actions (Tensor): Tokenized assistant response sequences (actions), shape (batch_size, action_len).
            advantages (Tensor): Advantage estimates for each sample, shape (batch_size,).
            old_logprobs (Tensor): Log probabilities of actions under the reference (old) policy, shape (batch_size, action_len).

        Returns: 
            policy_loss (Tensor): The mean PPO policy loss over the batch (scalar).
            value_loss (Tensor): The mean value function loss over the batch (scalar).
        """
        ## break differentiation graph 
        old_log_probs = old_log_probs.clone().detach() 
        ## Compute current log probs 
        log_probs = F.log_softmax(logits, dim=-1) 
        action_log_probs = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1) ## TODO this likely needs a sum since actions are sequences 
        ## TODO Use GAE instead. Use masks to avoid learning from padding tokens. 
        ## Advantage estimate: TD(0) with sequences (not tokens) as actions 
        target_values = rewards + gamma * next_values * (1. - done) 
        advantages = (target_values - values).detach()  ## no gradient through targets 
        ## Policy loss (PPO clip) 
        ratios = torch.exp(action_log_probs - old_log_probs) 
        unclipped = ratios * advantages 
        clipped = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages 
        policy_loss = -torch.mean(torch.min(unclipped, clipped))  ## maximize advantage 
        ## Value loss (TD error)
        value_loss = F.mse_loss(values, target_values)
        return policy_loss, value_loss 
