
## libraries 

import sys 
sys.path.append('../') 
from car import Model 
from tqdm import tqdm 
from time import time 
import pickle 
import os

## constants 

DATA_DIR = 'data_without_rl/' 
MODEL_DIR = 'models/'

## code 

model = Model() 

model.load_car_env_data(DATA_DIR) 

def fit(n=100, sub_iters=10): 
    losses = [] 
    rewards = [] 
    pbar = tqdm(range(n)) 
    for _ in pbar: 
        loss, _, mean_reward = model.optimize(max_iter=sub_iters, batch_size=25) 
        loss = float(loss) 
        mean_reward = float(mean_reward) 
        pbar.set_description(f'loss: {round(loss, 3)}, mean_reward: {round(mean_reward, 3)}') 
        losses.append(loss) 
        rewards.append(mean_reward) 
        pass 
    return losses, rewards   

print('fitting model...') 
result = fit(n=100) 

print('converting observations to memory...') 
model.convert_observations_to_memory(krylov_rank=10, disable_tqdm=False) 

## save to disk 
filepath = f'{MODEL_DIR}/model-stage-1-v1-t{int(time())}.pkl' 
with open(filepath, 'wb') as f: 
    pickle.dump(model, f) 
    pass 
print(f'model written to {filepath}') 

