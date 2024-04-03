## Experiment 26: Actually optimal lambda  
## 
## Hypothesis: I found some math elluding to a decent lambda estimate ...then fixed my math.  

import pandas as pd
from az_blob_util import upload_to_blob_store, download_from_blob_store, ls_blob_store, create_container 
from mnist_model import download_mnist_to_blob_storage 
import os 
from time import time 
from pyspark import SparkContext
from pyspark.sql import SparkSession 
sc = SparkContext() 
spark = SparkSession(sc) 

EXPERIMENT_ID = 26  
N_EXPERIMENTAL_ITERATIONS = 1000  
ITERS = 500  
EXPERIMENT_TIME = int(time())
TEMP_CONTAINER_NAME = f'tmp-{EXPERIMENT_ID}-{EXPERIMENT_TIME}' 
create_container(os.environ['STORAGE_KEY'] , TEMP_CONTAINER_NAME)
print(f'Working from container: {TEMP_CONTAINER_NAME}') 

## download mnist once, then distribute via blob storage 
## this isn't just a courtesy, because the download is rate-limited 
download_mnist_to_blob_storage() 

def get_regularizing_lambda_function(model, prior_parameter_vector, multiplier=1.0): 
    current_parameter_vector = model.get_parameter_vector() 
    param_delta = (current_parameter_vector - prior_parameter_vector).reshape([-1, 1])  
    hessian_estimate = model.hessian_sum / model.hessian_denominator ## does not scale well with model size! 
    lambda_estimate = param_delta.transpose(0,1).matmul(hessian_estimate).matmul(param_delta) 
    lambda_estimate = multiplier * lambda_estimate.reshape([]).abs().detach() 
    lambda_estimate *= 2. ## cancelling hard-coded term in regmem.py 
    return lambda x: lambda_estimate  

def map1(task_idx):
    import traceback 
    try:
        task_idx = int(task_idx) 
        ## run experiment
        from regmem import Model
        from az_blob_util import upload_to_blob_store
        import os
        import pickle
        from tqdm import tqdm
        condition_0_model = Model() 
        ## condition 0 (control): No use of memory, no discarding of data
        condition_0_result_tuples_before = condition_0_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False, silence_tqdm=True)
        ## copy, creating other models before continuing 
        condition_1_model = condition_0_model.copy()
        condition_2_model = condition_0_model.copy()
        condition_3_model = condition_0_model.copy() 
        condition_4_model = condition_0_model.copy() 
        condition_5_model = condition_0_model.copy() 
        ## continue condition 0 (control) Use memory, do discard data 
        condition_0_model.convert_observations_to_memory() 
        condition_0_result_tuples_after_1 = condition_0_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        condition_0_model.convert_observations_to_memory() 
        condition_0_result_tuples_after_2 = condition_0_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        condition_0_model.convert_observations_to_memory() 
        condition_0_result_tuples_after_3 = condition_0_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        ## condition 1 (experimental): Use memory, do dicard data, optimal lambda 
        condition_1_model.convert_observations_to_memory() 
        p = condition_1_model.get_parameter_vector().detach() 
        condition_1_result_tuples_after_1 = condition_1_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        condition_1_model.regularizing_lambda_function = get_regularizing_lambda_function(condition_1_model, p, multiplier=ITERS) 
        condition_1_model.convert_observations_to_memory()
        p = condition_1_model.get_parameter_vector().detach() 
        condition_1_result_tuples_after_2 = condition_1_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        condition_1_model.regularizing_lambda_function = get_regularizing_lambda_function(condition_1_model, p, multiplier=ITERS) 
        condition_1_model.convert_observations_to_memory()
        condition_1_result_tuples_after_3 = condition_1_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False)
        ## condition 2 (experimental): Use memory, do discard data, hueristically optimal lambda = p   
        condition_2_model.convert_observations_to_memory()
        p = condition_2_model.get_parameter_vector().detach() 
        condition_2_result_tuples_after_1 = condition_2_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        #condition_2_model.regularizing_lambda_function = get_regularizing_lambda_function(condition_2_model, p, multiplier=1.0) 
        condition_2_model.regularizing_lambda_function = lambda model: p.shape[0] 
        condition_2_model.convert_observations_to_memory() 
        p = condition_2_model.get_parameter_vector().detach() 
        condition_2_result_tuples_after_2 = condition_2_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        #condition_2_model.regularizing_lambda_function = get_regularizing_lambda_function(condition_2_model, p, multiplier=2.0) 
        condition_2_model.regularizing_lambda_function = lambda model: p.shape[0] 
        condition_2_model.convert_observations_to_memory()
        condition_2_result_tuples_after_3 = condition_2_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False)
        ## condition 3 (experimental): Use memory, do discard data, hueristically optimal lambda = p n / n_B
        condition_3_model.convert_observations_to_memory()
        p = condition_3_model.get_parameter_vector().detach()
        condition_3_result_tuples_after_1 = condition_3_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False)
        #condition_3_model.regularizing_lambda_function = get_regularizing_lambda_function(condition_3_model, p, multiplier=1.0) 
        condition_3_model.regularizing_lambda_function = lambda model: p.shape[0] * 2. 
        condition_3_model.convert_observations_to_memory()
        p = condition_3_model.get_parameter_vector().detach()
        condition_3_result_tuples_after_2 = condition_3_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False)
        #condition_3_model.regularizing_lambda_function = get_regularizing_lambda_function(condition_3_model, p, multiplier=2.0) 
        condition_3_model.regularizing_lambda_function = lambda model: p.shape[0] * 3. 
        condition_3_model.convert_observations_to_memory()
        condition_3_result_tuples_after_3 = condition_3_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        ## continue condition 4 (control) No memory, do not discard data
        condition_4_result_tuples_after_1 = condition_4_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False)
        condition_4_result_tuples_after_2 = condition_4_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False)
        condition_4_result_tuples_after_3 = condition_4_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False)
        ## continue condition 5 (control) Use memory, do discard data
        condition_5_model.clear_observations()
        condition_5_result_tuples_after_1 = condition_5_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False)
        condition_5_model.clear_observations()
        condition_5_result_tuples_after_2 = condition_5_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False)
        condition_5_model.clear_observations()
        condition_5_result_tuples_after_3 = condition_5_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False)
        ## merge before & after results 
        condition_0_result_tuples = condition_0_result_tuples_before + \
                condition_0_result_tuples_after_1 + \
                condition_0_result_tuples_after_2 + \
                condition_0_result_tuples_after_3
        condition_1_result_tuples = condition_0_result_tuples_before + \
                condition_1_result_tuples_after_1 + \
                condition_1_result_tuples_after_2 + \
                condition_1_result_tuples_after_3
        condition_2_result_tuples = condition_0_result_tuples_before + \
                condition_2_result_tuples_after_1 + \
                condition_2_result_tuples_after_2 + \
                condition_2_result_tuples_after_3
        condition_3_result_tuples = condition_0_result_tuples_before + \
                condition_3_result_tuples_after_1 + \
                condition_3_result_tuples_after_2 + \
                condition_3_result_tuples_after_3
        condition_4_result_tuples = condition_0_result_tuples_before + \
                condition_4_result_tuples_after_1 + \
                condition_4_result_tuples_after_2 + \
                condition_4_result_tuples_after_3
        condition_5_result_tuples = condition_0_result_tuples_before + \
                condition_5_result_tuples_after_1 + \
                condition_5_result_tuples_after_2 + \
                condition_5_result_tuples_after_3
        ## append condition codes
        condition_0_result_tuples = [(x[0], x[1], x[2], 0) for x in condition_0_result_tuples]
        condition_1_result_tuples = [(x[0], x[1], x[2], 1) for x in condition_1_result_tuples]
        condition_2_result_tuples = [(x[0], x[1], x[2], 2) for x in condition_2_result_tuples]
        condition_3_result_tuples = [(x[0], x[1], x[2], 3) for x in condition_3_result_tuples]
        condition_4_result_tuples = [(x[0], x[1], x[2], 4) for x in condition_4_result_tuples]
        condition_5_result_tuples = [(x[0], x[1], x[2], 5) for x in condition_5_result_tuples]
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
                out.append((cumulative_reward, iter_idx, task_idx, condition))
                pass
            pass
        append_results(condition_0_result_tuples)
        append_results(condition_1_result_tuples)
        append_results(condition_2_result_tuples)
        append_results(condition_3_result_tuples)
        append_results(condition_4_result_tuples)
        append_results(condition_5_result_tuples)
        ## write out 
        filename = f'experiment-{EXPERIMENT_ID}-result-{task_idx}.pkl'
        sas_key = os.environ['STORAGE_KEY']
        output_container_name = TEMP_CONTAINER_NAME
        upload_to_blob_store(pickle.dumps(out), filename, sas_key, output_container_name)  
    except Exception as e: 
        ## increase verbosity before failing 
        print(f'ERROR!\n{e}\n{traceback.format_exc()}')
        raise e 
    return filename   

def map2(filename): 
    import os 
    import pickle 
    from az_blob_util import download_from_blob_store 
    ## returns a list, use flatmap 
    sas_key = os.environ['STORAGE_KEY']  
    container_name = TEMP_CONTAINER_NAME 
    x = download_from_blob_store(filename, sas_key, container_name) 
    return pickle.loads(x) 

def phase_1(): 
    x = sc.parallelize(list(range(N_EXPERIMENTAL_ITERATIONS)), N_EXPERIMENTAL_ITERATIONS) 
    y = x.map(map1) 
    return y.collect() 

def phase_2(): 
    import pandas as pd
    import matplotlib.pyplot as plt
    ## config 
    sas_key = os.environ['STORAGE_KEY'] 
    input_container_name = TEMP_CONTAINER_NAME  
    output_container_name = 'data' 
    ## get data 
    filenames = ls_blob_store('', sas_key, input_container_name) 
    filenames = [f for f in filenames if f.startswith(f'experiment-{EXPERIMENT_ID}-result-') and f.endswith('.pkl')]
    filenames = sc.parallelize(filenames, len(filenames)) 
    y = filenames.flatMap(map2) 
    ## process 
    schema = ['score', 'iter', 'task', 'condition'] 
    z = y.toDF(schema=schema) 
    w = z.groupBy('iter', 'condition').mean('score') 
    df = w.toPandas()
    scores0 = df.loc[df['condition'] == 0].sort_values('iter')['avg(score)'].tolist() 
    scores1 = df.loc[df['condition'] == 1].sort_values('iter')['avg(score)'].tolist() 
    scores2 = df.loc[df['condition'] == 2].sort_values('iter')['avg(score)'].tolist() 
    scores3 = df.loc[df['condition'] == 3].sort_values('iter')['avg(score)'].tolist() 
    scores4 = df.loc[df['condition'] == 4].sort_values('iter')['avg(score)'].tolist()
    scores5 = df.loc[df['condition'] == 5].sort_values('iter')['avg(score)'].tolist()
    ## save data 
    FILENAME = f'df-experiment-{EXPERIMENT_ID}'
    df_to_save = pd.DataFrame({'scores0': scores0, 
                               'scores1': scores1,
                               'scores2': scores2,
                               'scores3': scores3,
                               'scores4': scores4,
                               'scores5': scores5}) 
    df_data = df_to_save.to_csv().encode() 
    upload_to_blob_store(df_data, FILENAME+'.csv', sas_key, output_container_name) 
    ## save plot 
    plt.plot(scores0, label='lambda = 1') 
    plt.plot(scores1, label='lambda = optimal') 
    plt.plot(scores2, label='lambda = p') 
    plt.plot(scores3, label='lambda = p*n/n_B') 
    plt.plot(scores4, label='No memory, discard data') 
    plt.plot(scores5, label='No memory, keep data') 
    plt.legend() 
    plt.savefig(FILENAME+'.png') 
    with open(FILENAME+'.png', 'rb') as f: 
        upload_to_blob_store(f.read(), FILENAME+'.png', sas_key, output_container_name) 
        pass 
    pass 

if __name__ == '__main__': 
    phase_1() 
    phase_2() 
