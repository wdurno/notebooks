import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline  
from time import time 
import json 
from replay_buffer import ReplayBuffer 
from regmem_ac import Actor, Object 
import argparse 

MAX_LENGTH=512  
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
PROMPT = ('...waiting for User to join...\n\n'*11) + 'User has joined!\n\n\n\n' + PROMPT ## 412 = 512 - 100 tokens  

tokenizer = GPT2Tokenizer.from_pretrained(MODEL)
sentiment_analysis_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis") 
gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
cpu = torch.device('cpu') 

def get_new_model(): 
    return Actor() 

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
    if tokenized_transcript.shape[1] > cut: 
        tokenized_transcript = tokenized_transcript[:,(tokenized_transcript.shape[1] - cut):] 
        pass 
    ## to gpu 
    tokenized_transcript = tokenized_transcript.to(gpu) 
    ## generate 
    output_tensor = model.generate(tokenized_transcript, max_length=MAX_LENGTH, do_sample=True, early_stopping=True, pad_token_id=tokenizer.eos_token_id, num_beams=5, no_repeat_ngram_size=2) 
    ## punish rambling 
    rambling = 0. 
    if output_tensor[:,-1] != tokenizer.eos_token_id: 
        output_tensor[:,-1] = tokenizer.eos_token_id 
        rambling = -2. 
        print('`score -= 2.` due to rambling') 
        pass 
    ## decode 
    output_text = tokenizer.decode(output_tensor[0][(MAX_LENGTH - MAX_RESPONSE):], skip_special_tokens=True) 
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
        done_list.append(done) ## always zero. Need to catch `Ctrl+C` and update final zero, done below  
        reward = torch.tensor([0.]) 
        if idx + 1 == len(output_tensor[0]): 
            reward += rambling 
            pass 
        reward_list.append(reward) ## final zero needs to be updated with score from next query 
        pass 
    transitions = Object() 
    transitions.state = torch.stack(state_list) ## .to(cpu) ## keep on GPU for fast action calculation 
    transitions.next_state = torch.stack(next_state_list).to(cpu) 
    transitions.reward = torch.cat(reward_list).to(cpu) 
    transitions.done = torch.cat(done_list).to(cpu) 
    with torch.no_grad(): ## stops memory explosions 
        transitions.action = torch.cat([actor(x).to(cpu) for x in transitions.state.split(BATCH_SIZE)]) ## to cpu   
        pass 
    transitions.state = transitions.state.to(cpu) ## finally get it on the cpu  
    ## package outputs 
    out = {
            'output_text': output_text, 
            'updated_transcript': transcript + output_text, 
            'score': score, ## for previous query 
            'transitions': transitions 
            } 
    return out 

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Converse with GPT2 or RL-modified-GPT2 to generate data for reinforcement learning.') 
    parser.add_argument('--path', dest='path', help='Load an Actor model with a prefix path. Example: models/mv1', \
            default=None) 
    args = parser.parse_args() 
    ## init 
    actor = get_new_model().to(gpu) 
    if args.path is not None: 
        actor.load(args.path+'.actor') 
        pass 
    transcript = None  
    replay_buffer = ReplayBuffer() 
    print('End this conversation and save data with Ctrl-C.') 
    print('Please say something to your AI Assistant.') 
    try: 
        ## chat loop 
        while True: 
            ## get user query 
            print('User:') 
            query = input() 
            ## process 
            out  = chat(actor, query, transcript) 
            out_text, transcript, tr, score = out['output_text'], out['updated_transcript'], out['transitions'], out['score'] 
            if replay_buffer.n > 0: 
                replay_buffer.reward_storage[-1] += score 
                pass 
            replay_buffer.add(tr.state, tr.action, tr.reward, tr.next_state, tr.done) 
            ## print output 
            print('AI Assistant:') 
            print(out_text) 
            pass 
    except: ## KeyboardInterrupt: ## input() throws funny errors on escape 
        ## save data before exit on Ctrl-C 
        if transcript is not None: 
            replay_buffer.done_storage[-1] = 1 ## register done 
            t = int(time()) 
            print(f'Saving data with timestamp {t}...') 
            with open(f'data/chat-log-{t}.txt', 'w') as f: 
                f.write(transcript) 
                pass 
            replay_buffer.save(f'data/chat-game-data-{t}.pt') 
            pass 
        pass 
    pass 
