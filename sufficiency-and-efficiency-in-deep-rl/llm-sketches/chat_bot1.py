import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline  
from time import time 
import json 

MAX_LENGTH=1024  
MAX_RESPONSE=100 
AVG_TOKENS_PER_WORD=3  
MODEL='gpt2'
#MODEL='gpt2-xl' ## too slow for local testing 

tokenizer = GPT2Tokenizer.from_pretrained(MODEL)
model = GPT2LMHeadModel.from_pretrained(MODEL, pad_token_id=tokenizer.eos_token_id)
sentiment_analysis_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis") 

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

def chat(query, transcript=None, file_pointer=None): 
    ## score 
    score = 0. 
    if transcript is not None: 
        score = eval(query) 
        pass 
    print(f'score: {score}') 
    ## buid transcript 
    if transcript is None: 
        transcript = '\n\nUser: ' + query + '\n\nAI Assistant: ' 
    else: 
        transcript += '\n\nUser: ' + query + '\n\nAI Assistant: ' 
        pass 
    ## heuristic truncation 
    transcript_chunks = transcript.split(' ') 
    cut = (MAX_LENGTH - MAX_RESPONSE) // AVG_TOKENS_PER_WORD - 100 
    if len(transcript_chunks) > cut : 
        transcript_chunks = transcript_chunks[cut:] 
        pass 
    transcript = ' '.join(transcript_chunks) 
    ## generate 
    input_tensor = tokenizer.encode(transcript, return_tensors='pt')  
    n_input_tokens = input_tensor.shape[1] 
    response_max = n_input_tokens + MAX_RESPONSE 
    if response_max > MAX_LENGTH: 
        response_max = MAX_LENGTH 
        pass 
    output_tensors = model.generate(input_tensor, max_length=response_max, do_sample=True, early_stopping=True, pad_token_id=tokenizer.eos_token_id, num_beams=5, no_repeat_ngram_size=2)
    output_text = tokenizer.decode(output_tensors[0], skip_special_tokens=True) 
    new_text = output_text[len(transcript):] 
    ## curate
    new_text = truncate_after(new_text, 'AI Assistant:') 
    new_text = truncate_after(new_text, 'User:') 
    if file_pointer is not None: 
        ## save data 
        user_data = json.dumps({'User': query, 'score': score}) 
        ai_data = json.dumps({'AI Assistant': new_text}) 
        file_pointer.write(user_data + '\n' + ai_data + '\n') 
        file_pointer.flush() 
        pass 
    return new_text, transcript + new_text 

def chat_loop(file_pointer=None): 
    transcript = None 
    print('Please say something to your AI Assistant') 
    while True: 
        print('User:') 
        query = input() 
        output, transcript = chat(query, transcript, file_pointer=file_pointer) 
        print('AI Assistant:') 
        print(output) 
        pass 
    pass 

with open(f'data/chat-log-{int(time())}.json', 'w') as f: 
    chat_loop(file_pointer=f) 
    pass 
