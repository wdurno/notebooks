from regmem_ac import GPT2ActorCritic;
import argparse 

parser = argparse.ArgumentParser(description='Add a small amount of data to your model.') 
parser.add_argument('--load-model-path', dest='load_model_path', help='Load an Actor model with a prefix path. Example: models/mv1') 
parser.add_argument('--save-model-path', dest='save_model_path', help='Save the updated model at this prefix path. Example: models/mv2') 
parser.add_argument('--load-data-path', dest='load_data_path', \
        help='Load a .pt file of data to add to your model. Example: data/chat-game-data-1726580965.pt') 
parser.add_argument('--p-gpt2-loss', dest='p_gpt2_loss', default=.99, type=float, \
        help='proportion of loss derived from the GPT2 loss.')
args = parser.parse_args()
load_model_path = args.load_model_path 
save_model_path = args.save_model_path 
load_data_path = args.load_data_path 
p_gpt2_loss = args.p_gpt2_loss 

## load model 
m = GPT2ActorCritic() 
m.load(load_model_path) 
## load data 
m.replay_buffer.load(load_data_path) 
## update model 
m.fit_loop(n_iters=100, batch_size=256, p_gpt2_loss=p_gpt2_loss) 
## memorize 
m.memorize() 
## save 
m.save(save_model_path) 

