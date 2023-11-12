
## libraries 

import sys 
sys.path.append('../') 
from car import Model, DEVICE 
from tqdm import tqdm 
from time import time 
from matplotlib import pyplot as plt 
import os 
import pickle 
import torch 
from data_farm_constants import HOST

## constants 

API_HOST = HOST 
MODEL_DIR = 'models/'
OPTIMIZATION_ITERS = 200 
MEMORIZATION_CYCLES = 3 

## find latest model 
available_models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')] 
INPUT_MODEL_FILE = available_models[0] ## example: 'model-stage-2-v1-t1699814976.pkl' 
current_timestamp = int(INPUT_MODEL_FILE.split('.')[3][1:].replace('.pkl', '')) 
for f in available_models: 
    ## get latest model 
    candidate_timestamp = int(f.split('.')[3][1:].replace('.pkl', '')) 
    if candidate_timestamp > current_timestamp: 
        current_timestamp = candidate_timestamp 
        INPUT_MODEL_FILE = f 
        pass 
    pass 

#def lambda_func(x): 
#    return 10.0
lambda_func = None 

## code 

input_model_path = os.path.join(MODEL_DIR, INPUT_MODEL_FILE)
with open(input_model_path, 'rb') as f: 
    loaded_model = pickle.load(f)  
    ## pickle saves old class definitions 
    ## by copying essential content into a new instance, we can debug faster 
    model = Model(n_actions=loaded_model.n_actions,
                n_camera_directions=loaded_model.n_camera_directions,
                max_sample=loaded_model.max_sample,
                discount=loaded_model.discount,
                eps=loaded_model.eps,
                explore_probability_func=loaded_model.explore_probability_func,
                batch_size=loaded_model.batch_size,
                learning_rate=loaded_model.learning_rate,
                grad_clip=loaded_model.grad_clip,
                short_term_memory_length=loaded_model.short_term_memory_length,
                lbfgs=loaded_model.lbfgs,
                env_name=loaded_model.env_name,
                hessian_sum=loaded_model.hessian_sum.detach().clone() if loaded_model.hessian_sum is not None else None,
                hessian_sum_low_rank_half=loaded_model.hessian_sum_low_rank_half,
                hessian_denominator=loaded_model.hessian_denominator, ## this is an int or None, so it is copied via '='
                hessian_center=loaded_model.hessian_center.detach().clone() if loaded_model.hessian_center is not None else None,
                hessian_residual_variances=loaded_model.hessian_residual_variances.detach().clone() if loaded_model.hessian_residual_variances is not None else None,
                observations=loaded_model.observations.copy(),
                total_iters=loaded_model.total_iters,
                regularizing_lambda_function=loaded_model.regularizing_lambda_function) 
    pass 

def save_model(): 
    filepath = os.path.join(f'{MODEL_DIR}', f'model-stage-2-v1-t{int(time())}.pkl') 
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
        pass
    print(f'model saved as {filepath}') 
    pass 

model.regularizing_lambda_function = lambda_func ## real lambdas don't pickle

for _ in range(MEMORIZATION_CYCLES): 
    print('experimenting...') 
    model.simulate(host=API_HOST, total_iters=OPTIMIZATION_ITERS, model.memory_write_location=os.getcwd()+'/data_with_rl') 
    print('memorizing...')  
    model.convert_observations_to_memory(krylov_rank=10, disable_tqdm=False) 
    save_model() 
    torch.cuda.empty_cache() 
    pass 

