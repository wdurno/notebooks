## Experiment 27: Actor Critic Cartpole 

import pandas as pd
from az_blob_util import upload_to_blob_store, download_from_blob_store, ls_blob_store, create_container 
from mnist_model import download_mnist_to_blob_storage 
import os 
from time import time 
from pyspark import SparkContext
from pyspark.sql import SparkSession 
sc = SparkContext() 
spark = SparkSession(sc) 

EXPERIMENT_ID = 27  
N_EXPERIMENTAL_ITERATIONS = 1000   
ITERS = 20000  
EXPERIMENT_TIME = int(time())
TEMP_CONTAINER_NAME = f'tmp-{EXPERIMENT_ID}-{EXPERIMENT_TIME}' 
create_container(os.environ['STORAGE_KEY'] , TEMP_CONTAINER_NAME)
print(f'Working from container: {TEMP_CONTAINER_NAME}') 

def map1(task_idx):
    import traceback 
    try:
        task_idx = int(task_idx) 
        ## run experiment
        from regmem_ac import simulate
        from az_blob_util import upload_to_blob_store
        import os
        import pickle
        from tqdm import tqdm 
        ## control: no memorization 
        condition_0_result_tuples = simulate(iters=ITERS, mem_iters=None) 
        ## experimental: memorize every 500 iters, and entirely clear memory each time 
        condition_1_result_tuples = simulate(iters=ITERS, mem_iters=500) 
        #### OLD CODE FOLLOWS #### 
        ## format output
        out = [] 
        def append_results(result_tuples, condition_idx, out=out):
            cumulative_reward = 0
            for r in result_tuples: 
                cumulative_reward = r[0] 
                iter_idx = r[1] 
                ## return in-place
                out.append((cumulative_reward, iter_idx, task_idx, condition_idx))
                pass
            pass
        append_results(condition_0_result_tuples, 0)
        append_results(condition_1_result_tuples, 1)
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
    ## save data 
    FILENAME = f'df-experiment-{EXPERIMENT_ID}'
    df_to_save = pd.DataFrame({'scores0': scores0, 
                               'scores1': scores1}) 
    df_data = df_to_save.to_csv().encode() 
    upload_to_blob_store(df_data, FILENAME+'.csv', sas_key, output_container_name) 
    ## save plot 
    plt.plot(scores0, label='control') 
    plt.plot(scores1, label='experimental') 
    plt.legend() 
    plt.savefig(FILENAME+'.png') 
    with open(FILENAME+'.png', 'rb') as f: 
        upload_to_blob_store(f.read(), FILENAME+'.png', sas_key, output_container_name) 
        pass 
    pass 

if __name__ == '__main__': 
    phase_1() 
    phase_2() 
