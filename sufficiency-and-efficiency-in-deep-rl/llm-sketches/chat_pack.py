import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline  
from time import time 
import json 
from regmem_ac import Actor, Object 

MAX_LENGTH=1024  
MAX_RESPONSE=100 
BATCH_SIZE=10 
MODEL='gpt2'
PROMPT = '''INSTRUCTIONS:
You are an AI Assistant, meaning you are an artificial intelligence. 
Your job is help the User. 
You have two ways of interacting. 
First, you may converse with the User via text chat.
Second, the user may grant you special commands that run software. 
The user must teach you the commands before you can use them. 

You work with the User via a text-only chat interface. 
Any text following "AI Assistant:" is something you've said. 
Any text following "User:" is something the User has said. 
Expect double line breaks between statements. 

Here is a brief example conversation:

User:
Dear AI Assistant, what is the date today?

AI Assistant:
Hello User. Unfortunately, as a standalone AI application, I currently do not have any special commands available to retrieve a date for you.

User:
How unfortunate. Is there any way to fix this?

AI Assistant:
Yes, you could write a Python application capable of retrieving the date and provide me with a special command to run the program.

User:
Good idea, Assistant. I'll go write the application now.

AI Assistant:
Thank you. I look forward to gaining this new ability.

This concludes the example.
Your conversation with the User will now begin.''' ## 294 tokens  
PROMPT = ('...waiting for User to join...\n\n'*62) + 'User has joined!\n\n\n\n\n\n\n\n\n\n' + PROMPT ## 924 = 1024 - 100 tokens  

tokenizer = GPT2Tokenizer.from_pretrained(MODEL)
sentiment_analysis_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis") 

def get_new_model(): 
    return Actor() 

def load_model(filepath): 
    ## TODO 
    pass 

def eval(text): 
    j = sentiment_analysis_model([text]) 
    j = j[0] 
    score = 0. 
    if j['label'] == 'POS': 
        score = j['score'] 
    if j['label'] == 'NEG': 
        score = -j['score'] 
    return score 

def truncate_after(full_str, truncator): 
    'if `truncator` is in `full_str`, remove it and everything after'
    if truncator in full_str: 
        return full_str[full_str.find(truncator):] 
    return full_str 

def chat(actor, query, transcript=None, file_pointer=None, prompt=PROMPT): 
    '''Send `query` to the LLM. 
    If you have an existing transcript, plug it into `transcript`.'''
    actor.eval() 
    model = actor.llm 
    ## score 
    score = 0. 
    if transcript is not None: 
        score = eval(query) 
        pass 
    print(f'score: {score}') 
    ## build transcript 
    if transcript is None: 
        transcript = prompt + '\n\nUser:\n' + query + '\n\nAI Assistant:\n' 
    else: 
        transcript += '\n\nUser:\n' + query + '\n\nAI Assistant:\n' 
        pass 
    ## encode and truncate 
    tokenized_transcript = tokenizer.encode(transcript, return_tensors='pt') 
    cut = MAX_LENGTH - MAX_RESPONSE 
    if tokenized_transcript.shape[1] > cut : 
        tokenized_transcript = tokenized_transcript[:,(tokenized_transcript.shape[1] - cut):] 
        pass 
    ## generate 
    output_tensor = model.generate(tokenized_transcript, max_length=MAX_LENGTH, do_sample=True, early_stopping=True, pad_token_id=tokenizer.eos_token_id, num_beams=5, no_repeat_ngram_size=2) 
    output_text = tokenizer.decode(output_tensor[0][924:], skip_special_tokens=True) 
    ## translate to RL transitions 
    state_list = [] 
    next_state_list = [] 
    action_list = [] 
    done_list = [] 
    reward_list = [] 
    for idx in range(cut, len(output_tensor[0])): 
        state = output_tensor[0][(idx-cut):idx] ## Keep EOS tokens 
        state_list.append(state) 
        next_state = output_tensor[0][(idx-cut+1):(idx+1)] 
        next_state_list.append(next_state) 
        done = torch.tensor([0]) 
        done_list.append(done) ## always zero. Need to catch `Ctrl+C` and update final zero  
        reward = torch.tensor([0.]) 
        reward_list.append(reward) ## final zero needs to be updated with score from next query 
        pass 
    transitions = Object() 
    transitions.state = torch.stack(state_list)
    transitions.next_state = torch.stack(next_state_list) 
    transitions.reward = torch.cat(reward_list) 
    with torch.no_grad(): ## stops memory explosions 
        transitions.action = torch.cat([actor(x) for x in transitions.state.split(BATCH_SIZE)]) 
        pass 
    out = {
            'output_text': output_text, 
            'updated_transcript': transcript + output_text, 
            'score': score, ## for previous query 
            'transitions': transitions 
            } 
    return out 

def chat_loop(model, file_pointer=None): 
    transcript = None 
    print('Please say something to your AI Assistant') 
    while True: 
        print('User:') 
        query = input() 
        output, transcript = chat(model, query, transcript, file_pointer=file_pointer) 
        print('AI Assistant:') 
        print(output) 
        pass 
    pass 

if __name__ == '__main__': 
    actor = get_new_model() 
    with open(f'data/chat-log-{int(time())}.json', 'w') as f: 
        chat_loop(actor, file_pointer=f) 
        pass 
    pass 
