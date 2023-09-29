## Experiment 20: Hard subspacing 
## 
## Hypothesis: fixing high-info dimensions has a similar effect to a large lambda value 

import pandas as pd
from az_blob_util import upload_to_blob_store, download_from_blob_store, ls_blob_store, create_container 
from mnist_model import download_mnist_to_blob_storage 
import os 
from time import time 
from pyspark import SparkContext
from pyspark.sql import SparkSession 
sc = SparkContext() 
spark = SparkSession(sc) 

EXPERIMENT_ID = 20  
N_EXPERIMENTAL_ITERATIONS = 1000  
LAMBDA = 1.   
ACC_FREQ=10
ITERS = 500  
PHASE_1_SAMPLE_SIZE = 100 
PHASE_2_SAMPLE_SIZE = 20 
KRYLOV_EPS = 0. 
L2_REG = None  
KRYLOV_RANK = 10
EXPERIMENT_TIME = int(time())
TEMP_CONTAINER_NAME = f'tmp-{EXPERIMENT_ID}-{EXPERIMENT_TIME}' 
create_container(os.environ['STORAGE_KEY'] , TEMP_CONTAINER_NAME)
print(f'Working from container: {TEMP_CONTAINER_NAME}') 

## download mnist once, then distribute via blob storage 
## this isn't just a courtesy, because the download is rate-limited 
download_mnist_to_blob_storage() 

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
        ## continue condition 0 trial, without application of memory and without discarding data
        condition_0_result_tuples_after = condition_0_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False, silence_tqdm=True)
        ## condition 1 (control): No use of memory, do discard data
        condition_1_model.clear_observations()
        condition_1_result_tuples_after = condition_1_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False, silence_tqdm=True)
        ## condition 2 (experimental): Use memory, do discard data
        condition_2_model.convert_observations_to_memory()
        condition_2_result_tuples_after = condition_2_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False, silence_tqdm=True)
        ## condition 3 (experimental): No use of memory, do discard data, drop high-info dims  
        condition_3_model.convert_observations_to_memory()  
        condition_3_result_tuples_after = condition_3_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False, silence_tqdm=True, high_info_proportion=.01) 
        ## condition 4 (experimental): No use of memory, do discard data, drop high-info dims 
        condition_4_model.convert_observations_to_memory()
        condition_4_result_tuples_after = condition_4_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False, silence_tqdm=True, high_info_proportion=.1)
        ## condition 5 (experimental): No use of memory, do discard data, drop high-info dims 
        condition_5_model.convert_observations_to_memory() 
        condition_5_result_tuples_after = condition_5_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False, silence_tqdm=True, high_info_proportion=.3)
        ## merge before & after results
        condition_0_result_tuples = condition_0_result_tuples_before + condition_0_result_tuples_after
        condition_1_result_tuples = condition_0_result_tuples_before + condition_1_result_tuples_after
        condition_2_result_tuples = condition_0_result_tuples_before + condition_2_result_tuples_after
        condition_3_result_tuples = condition_0_result_tuples_before + condition_3_result_tuples_after
        condition_4_result_tuples = condition_0_result_tuples_before + condition_4_result_tuples_after
        condition_5_result_tuples = condition_0_result_tuples_before + condition_5_result_tuples_after
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
    plt.plot(scores0, label='0') 
    plt.plot(scores1, label='1') 
    plt.plot(scores2, label='2') 
    plt.plot(scores3, label='3') 
    plt.plot(scores4, label='4') 
    plt.plot(scores5, label='5') 
    plt.legend() 
    plt.savefig(FILENAME+'.png') 
    with open(FILENAME+'.png', 'rb') as f: 
        upload_to_blob_store(f.read(), FILENAME+'.png', sas_key, output_container_name) 
        pass 
    pass 

if __name__ == '__main__': 
    phase_1() 
    phase_2() 
