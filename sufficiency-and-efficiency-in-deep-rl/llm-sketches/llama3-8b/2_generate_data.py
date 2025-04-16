## TODO put an interface on this for reusability over different model versions 

import torch 
from transformers import LlamaForCausalLM, AutoTokenizer

print('loading model...')
model = LlamaForCausalLM.from_pretrained("models/model_v0_quantized/quantized_model") 
print('loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B") 
tokenizer.pad_token = tokenizer.eos_token 

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f'found device: {device}')
print('sending model to device...')
model.to(device) 

print('tokenizing...')
input_ids = tokenizer("Hello there!", return_tensors="pt", padding=True)
input_ids = {k: v.to(device) for k, v in input_ids.items()}
print('generating...')
outputs = model.generate(**input_ids, 
        do_sample=True,
        temperature=0.95,
        top_p=0.95,
        max_new_tokens=1000, 
        pad_token_id=tokenizer.eos_token_id, 
        eos_token_id=tokenizer.eos_token_id
        )
print('decoding...')
print(tokenizer.decode(outputs[0]))