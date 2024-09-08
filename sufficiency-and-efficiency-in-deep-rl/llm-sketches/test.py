
from regmem_ac import GPT2ActorCritic; 
m = GPT2ActorCritic()
m.replay_buffer.load('data/chat-game-data-1725764004.pt') 
m.fit_loop(n_iters=2, batch_size=10, p_gpt2_loss=.9) 

