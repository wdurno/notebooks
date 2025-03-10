from regmem_ac import GPT2ActorCritic;

parser = argparse.ArgumentParser(description='Add initial dataset to a GPT2 model. Pad sufficient statistic before fitting.') 
parser.add_argument('--data-path', dest='data_path', help='Fit GPT2 to this model and update its sufficient statistic.', \
        default=None, required=True, type=str) 
args = parser.parse_args()
data_path = args.data_path 

## ssr_rank=1 fits this into 12GB GPU RAM. Use a higher value for better Hessian approximations 
m = GPT2ActorCritic(ssr_rank=1) 
## Gpt2 was fit on 80Gb of text, so 20B token regressions assuming 4 bytes per token. 
## Arbitrarily average information per sample and dimension is 1. per regressed token. 
## Setting samples value smaller to increase the effect of new data. 
N = 10.**9 
m.pad_gpt_memory(info=20.*N, samples=N) 
## load a small amount if data to add 
m.replay_buffer.load('data/chat-game-data-1728172436.pt') 
## SSR-regularized tuning allows us to add small amounts of data and slowly transform the distribution over `p_gpt2_loss`  
pi_min = .7 
pi_max = .3 
m.fit_loop(n_iters=100, batch_size=256, p_gpt2_loss=.99, pi_min=pi_min, pi_max=pi_max) 
## integrate new data into the SSR 
m.memorize()
m.save('models/mv1')
