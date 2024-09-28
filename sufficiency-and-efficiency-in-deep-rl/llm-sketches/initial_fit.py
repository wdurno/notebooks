from regmem_ac import GPT2ActorCritic;

## ssr_rank=1 fits this into 12GB GPU RAM. Use a higher value for better Hessian approximations 
m = GPT2ActorCritic(ssr_rank=1) 
## Gpt2 was fit on 80Gb of text, so 20B token regressions assuming 4 bytes per token. 
## Arbitrarily average information per sample and dimension is 1. per regressed token. 
## Setting samples value smaller to increase the effect of new data. 
N = 10.**9 
m.pad_gpt_memory(info=20.*N, samples=N) 
## load a small amount if data to add 
m.replay_buffer.load('data/chat-game-data-1726351690.pt') 
## SSR-regularized tuning allows us to add small amounts of data and slowly transform the distribution over `p_gpt2_loss`  
pi_min = 1. - N/(N+1) 
pi_max = .5 
m.fit_loop(n_iters=100, batch_size=256, p_gpt2_loss=.99, pi_min=pi_min, pi_max=pi_max) 
## integrate new data into the SSR 
m.memorize()
m.save('models/mv1')
