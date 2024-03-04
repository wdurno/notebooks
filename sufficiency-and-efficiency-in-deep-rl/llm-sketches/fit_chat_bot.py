import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from time import time
import json
import os 
from typing import Optional, Tuple, Union 

MAX_LENGTH=1024
MAX_RESPONSE=100
AVG_TOKENS_PER_WORD=3
MODEL='gpt2'

tokenizer = GPT2Tokenizer.from_pretrained(MODEL)
model = GPT2LMHeadModel.from_pretrained(MODEL, pad_token_id=tokenizer.eos_token_id)
sentiment_analysis_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")

def load_data(data_dir='data'): 
    '''
    Loads files into dict mapping files names to lists of statement objects. 
    So, it'll look like `{'filename-1': [<statments objects>], 'filename-2': ...}`. 
    Statement objects alternate between user and assistant statements, like this: 
    ``` 
    {'User': <str>, 'score': <float>} 
    {'AI Assistant': <str>} 
    ``` 
    Users statements have sentiment scores. 
    ''' 
    out = {} 
    files = os.listdir(data_dir) 
    for f in [f for f in files if f.endswith('.json')]: 
        with open(os.path.join(data_dir, f), encoding='utf-8') as f_ptr: 
            lines = f_ptr.readlines() # it's not a actually json file 
            pass 
        out[f] = [] 
        for line in lines: 
            statement_object = json.loads(line) # each line is json 
            out[f].append(statement_object) 
            pass 
        pass 
    return out ## TODO json isn't handling utf-8 characters correctly. I need to redo-experiments anyways (after fixing ": "), so I'll just pickle next time.  

def get_transcripts(data_dir='data'): 
    data = load_data(data_dir) 
    transcripts =  [] 
    for f in data: # f is a key pointing to a list of statement objects  
        transcript = '' 
        for so in data[f]: # so = statement object 
            if 'User' in so: 
                transcript += '\n\n' + 'User:' + so['User'] 
            if 'AI Assistant' in so: 
                transcript += '\n\n' + 'AI Assistant:' + so['AI Assistant']  
                pass 
            pass 
        transcripts.append(transcript) 
        pass 
    return transcripts 

def rl_forward(
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
    return transformer_outputs ## extends dict with keys 'last_hidden_state', 'past_key_values' 

def fit_iter(model, transcripts, alpha): 
    '''
    Linearly combine an language model (LM) loss with new RL loss. 

    inputs:
    - model: an instance of GPT2LMHeadModel 
    - transcripts: a list of strings making-up tuning data 
    - alpha: the linear combining term, between 0 and 1. At 0, loss is entirely the LM loss. At 1, loss is RL. 
    Side-effects: 
    - model is updated by 1 training iteration 

    Design inspired by this Huggingface documentation: 
    - handling loss functions in pytorch: `https://huggingface.co/docs/transformers/en/training#train-in-native-pytorch`.
    - label index shifting: `https://discuss.huggingface.co/t/gpt-2-shift-logits-and-labels/812`
    ''' 
    ## strategy: 
    ## 1. Get the LM loss with `model.forward` 
    ## 2. Get the hidden state `model.transformer(...)[0]` <-- will require slight change to model.forward 
    ## 3. Apply RL regression on with hidden state as input 
    ## 4. Combined losses and optimize 
    ## implementation: 
    ## sample 
    t = tokenizer.encode(transcripts[0], return_tensors='pt')[:, :1024] ## TODO actually build a simple random sample here 
    ## get LM loss 
    lm_loss = model.forward(t, labels=t).loss ## TODO double-check `forward` shifts label indices 
    ## get RL loss 
    rl_loss = rl_forward(model, t) ## TODO need to actually write loss function portion 
    pass 
















