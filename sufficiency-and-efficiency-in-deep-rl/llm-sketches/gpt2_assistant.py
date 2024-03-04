
import torch
import torch.nn as nn

from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline 

class GPT2Assistant(GPT2LMHeadModel): 
    '''
    A GPT2 language model modified to optimally learn how to keep a user happy. 
    '''
    def __init__(self, config): 
        super().__init__(config) 
        self.sentiment_analysis_model = None ## `pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")` avoid unnecessary loads 
        self.target_model = None ## for RL loss function 
        ## RL nets 
        ## hidden state inputs are shaped [-1, 1024, 768] 
        self.rl_conv1 = nn.Conv1d(768, 256, kernel_size=7, stride=5) ## shape is [-1, 256, 204] 
        self.rl_conv2 = nn.Conv1d(256, 16, kernel_size=7, stride=5) ## shape is [-1, 16, 40] 
        self.rl_linear = nn.Linear(16*40, 1) ## TODO this needs to be actor-critic. I need to verify my method before attempting without a proper validation set.   
        pass 

    def rl_pred(self, tokenized_transcriptions): ## TODO add non-linearities  
        x = tokenized_transcriptions.transpose(1,2) ## produce [series, channel, observation] 
        x = self.rl_conv1(x) 
        pass 
    pass 

