from regmem_ac import GPT2ActorCritic;

## TODO OOM! Slow memory leak 
m = GPT2ActorCritic(ssr_rank=1) ## ssr_rank=1 fits this into 12GB GPU RAM. Use a higher value for better Hessian approximations
m.pad_gpt_memory()
m.replay_buffer.load('data/chat-game-data-1725764004.pt')
m.fit_loop(n_iters=100, batch_size=256, p_gpt2_loss=.95) ## use many more n_iters in application
m.memorize()
m.save('models/mv1')
