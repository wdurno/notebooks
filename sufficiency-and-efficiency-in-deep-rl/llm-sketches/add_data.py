from regmem_ac import GPT2ActorCritic;
import argparse 

def add_data(load_model_path, save_model_path, load_data_path, p_gpt2_loss=.99, pi_min=-1., pi_max=-1., progress_bar=True): 
    '''Add data to an existing GPT2ActorCrtic model. 
    inputs: 
    - load_model_path: If a string is provided, it is assumed to be a file path and the model is loaded from here. If it's a GPT2ActorCritic instance, updates will be made in-place. 
    - save_model_path: If a string is provided, it is assumed to be a file path and a model is saved to this location. Otherwise, no save will occur. 
    - load_data_path: If a string is provided, it is assumed to be a file path to a `.pt`-formatted dataset, and the data will be loaded. If a non-string is provided, no data load will occur. 
    - p_gpt2_loss: proportion of loss dedicated to the standard GPT2 generative loss. 
    - pi_min: pi-form regularizer lower bound 
    - pi_max: pi-form regularizer upper bound 
    - progress_bar: If True, show a tmux progress bar. 
    outputs: 
    - pi_actor: the Actor model's pi estimate 
    - pi_critic: the Critic model's pi estimate 
    '''
    if type(load_model_path) == str: 
        ## load model 
        m = GPT2ActorCritic()
        m.load(load_model_path)
    elif str(type(load_model_path)) == "<class 'regmem_ac.GPT2ActorCritic'>": 
        m = load_model_path 
    else: 
        raise ValueError('ERROR: `load_model_path` must be a string or type GPT2ActorCritic!') 
    if type(load_data_path) == str: 
        ## load data 
        m.replay_buffer.load(load_data_path)
    pass 
    ## update model 
    N = 10.**9
    if pi_min < 0.:
        pi_min = 1. - N/(N+1)
        pass
    if pi_max < 0.:
        pi_max = .001
        pass
    pi_actor, pi_critic = m.fit_loop(n_iters=100, batch_size=256, p_gpt2_loss=p_gpt2_loss, pi_min=pi_min, pi_max=pi_max, progress_bar=progress_bar)
    ## memorize 
    m.memorize() 
    if type(save_model_path) == str: 
        ## save 
        m.save(save_model_path)
        pass 
    return pi_actor, pi_critic  

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Add a small amount of data to your model.') 
    parser.add_argument('--load-model-path', dest='load_model_path', help='Load an Actor model with a prefix path. Example: models/mv1') 
    parser.add_argument('--save-model-path', dest='save_model_path', help='Save the updated model at this prefix path. Example: models/mv2') 
    parser.add_argument('--load-data-path', dest='load_data_path', \
            help='Load a .pt file of data to add to your model. Example: data/chat-game-data-1726580965.pt') 
    parser.add_argument('--p-gpt2-loss', dest='p_gpt2_loss', default=.99, type=float, \
            help='proportion of loss derived from the GPT2 loss.') 
    parser.add_argument('--pi-min', dest='pi_min', default=-1., type=float, help='minimum proportion of new data') 
    parser.add_argument('--pi-max', dest='pi_max', default=-1., type=float, help='maximum proportion of new data') 
    args = parser.parse_args()
    load_model_path = args.load_model_path 
    save_model_path = args.save_model_path 
    load_data_path = args.load_data_path 
    p_gpt2_loss = args.p_gpt2_loss 
    pi_min = args.pi_min 
    pi_max = args.pi_max 
    ## add data to model 
    add_data(load_model_path=load_model_path, save_model_path=save_model_path, load_data_path=load_data_path, \
            p_gpt2_loss=p_gpt2_loss, pi_min=pi_min, pi_max=pi_max) 
    pass 
