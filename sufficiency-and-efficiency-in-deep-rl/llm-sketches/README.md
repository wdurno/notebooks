
To create an initial dataset, run `python3 chat_pack.py`. This'll load a blank GPT2 instance and you'll converse with it, generating data. Data will be written to the `data/` directory upon `Ctrl-C`. 

To perform an initial fit to data, again load a GPT2 instance and apply the dataset you've generated. Use interactive Python to achieve this like follows. Notice `mv1` is a prefix, as several files must be created per model. 
```
from regmem_ac import GPT2ActorCritic;
m = GPT2ActorCritic(ssr_rank=1) ## ssr_rank=1 fits this into 12GB GPU RAM. Use a higher value for better Hessian approximations 
m.pad_gpt_memory() 
m.replay_buffer.load('data/chat-game-data-1725764004.pt')
m.fit_loop(n_iters=2, batch_size=256, p_gpt2_loss=.95) ## use many more n_iters in application 
m.memorize() 
m.save('models/mv1')
```

```
python3 chat_pack.py 
python3 initial_fit.py 
python3 chat_pack.py --path models/mv1   
python3 add_data.py --load-model-path models/mv1 --save-model-path models/mv2 --load-data-path data/chat-game-data-1727407562.pt --pi-min .3 --pi-max .7 
python3 chat_pack.pu --path models/mv2 
python3 add_data.py --load-model-path models/mv2 --save-model-path models/mv3 --load-data-path data/chat-game-data-1727547144.pt --pi-min .3 --pi-max .7 
```

