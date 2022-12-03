import pandas as pd

SAMPLE_SIZE = 8 #100  
ITERS = 3000 
FIT_FREQ = 1 
LEARNING_RATE = 0.001 
BATCH_SIZE = 50 
LAMBDA = 1. 
KRYLOV_EPS = 0. 

task_idx=1

from miner import Model 
from az_blob_util import upload_to_blob_store 
import os 
import pickle 
condition_0_model = Model(env_name='MineRLNavigateDense-v0', learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE) 
## condition 0 (control): No use of memory, no discarding of data 
condition_0_result_tuples_before = condition_0_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False, \
        fit_freq=FIT_FREQ, manual_play=False) 
## copy, creating other models before continuing 
condition_0_model = condition_0_model.copy() 
condition_1_model = condition_0_model.copy() 
condition_0_model.env_name = 'MineRLNavigateExtremeDense-v0' 
condition_1_model.env_name = 'MineRLNavigateExtremeDense-v0'
## continue condition 0 (control), without application of memory and without discarding data 
condition_0_result_tuples_after = condition_0_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False, fit_freq=FIT_FREQ) 
## condition 1 (control): Use of memory, do discard data 
condition_1_model.convert_observations_to_memory(krylov_rank=10, krylov_eps=KRYLOV_EPS) 
condition_1_model.regularizing_lambda_function = lambda x: (LAMBDA) 
condition_1_result_tuples_after = condition_1_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False, fit_freq=FIT_FREQ) 
## merge before & after results 
condition_0_result_tuples = condition_0_result_tuples_before + condition_0_result_tuples_after 
condition_1_result_tuples = condition_0_result_tuples_before + condition_1_result_tuples_after 
## append condition codes 
condition_0_result_tuples = [(x[0], x[1], x[2], 0) for x in condition_0_result_tuples] 
condition_1_result_tuples = [(x[0], x[1], x[2], 1) for x in condition_1_result_tuples] 
## format output 
out = [] 
def append_results(result_tuples, out=out): 
    cumulative_reward = 0 
    for r in result_tuples: 
        reward = r[0] 
        done = r[1] 
        if done: 
            cumulative_reward = 0 
        else: 
            cumulative_reward += reward 
            pass 
        iter_idx = r[2] 
        condition = r[3] 
        ## return in-place 
        out.append((cumulative_reward, done, iter_idx, task_idx, condition))
        pass 
    pass 
append_results(condition_0_result_tuples) 
append_results(condition_1_result_tuples) 

