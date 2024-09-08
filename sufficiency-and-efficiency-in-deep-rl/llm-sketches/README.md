
To create an initial dataset, run `python3 chat_pack.py`. This'll load a blank GPT2 instance and you'll converse with it, generating data. Data will be written to the `data/` directory upon `Ctrl-C`. 

To perform an initial fit to data, again load a GPT2 instance and apply the dataset you've generated. Use interactive Python to achieve this like follows. Notice `mv1` is a prefix, as several files must be created per model. 
```
from regmem_ac import GPT2ActorCritic;
m = GPT2ActorCritic()
m.pad_gpt_memory() 
m.replay_buffer.load('data/chat-game-data-1725764004.pt')
m.fit_loop(n_iters=2, batch_size=256, p_gpt2_loss=.95)
m.save('models/mv1')
```



