## Experiment 1: use frontal lobe norms to clear a long-term memory cache between batches 

import pandas as pd
from az_blob_util import create_container, upload_to_blob_store, download_from_blob_store, ls_blob_store 
from mnist_model import download_mnist_to_blob_storage 
import os 
from time import time 
from pyspark import SparkContext
from pyspark.sql import SparkSession 
sc = SparkContext() 
spark = SparkSession(sc) 

EXPERIMENT_ID = 1 
N_EXPERIMENTAL_ITERATIONS = 100  
LAMBDA = 1.   
ACC_FREQ=10
FIT_ITERS = 1000  
SUBSAMPLE_SIZE = 400 
N = 1000  
KRYLOV_EPS = 0. 
L2_REG = None  
KRYLOV_RANK = 10 
BATCH_PREFIXES = ['01', '23', '45', '67', '89'] 
EXPERIMENT_TIME = int(time()) 
TEMP_CONTAINER_NAME = f'tmp-{EXPERIMENT_ID}-{EXPERIMENT_TIME}' 
create_container(os.environ['STORAGE_KEY'] , TEMP_CONTAINER_NAME) 

## download mnist once, then distribute via blob storage 
## this isn't just a courtesy, because the download is rate-limited 
download_mnist_to_blob_storage() 

def map1(task_idx):
    import traceback 
    try:
        task_idx = int(task_idx) 
        ## run experiment 
        from mnist_model import Model, get_datasets 
        from az_blob_util import upload_to_blob_store 
        import os 
        import pickle 
        from tqdm import tqdm 
        ## get data 
        datasets = get_datasets(n=N) 
        ## define initial model  
        case_0_model = Model(net_type='cnn', batch_norm=False, log1p_reg=False) ## control: don't memorize 
        case_1_model = case_0_model.copy() ## control: memorize 
        case_2_model = case_0_model.copy() ## experimental: memorize and clear frontal lobe 
        ## fit 
        for idx, prefix in enumerate(BATCH_PREFIXES): 
            _ = case_0_model.fit(datasets[f'{prefix}_train'], datasets[f'{prefix}_test'], n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ) 
            idx_batch = case_1_model.fit(datasets[f'{prefix}_train'], datasets[f'{prefix}_test'], n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                    ams=LAMBDA*N*idx) 
            case_1_model.memorize(datasets[f'{prefix}_train'], memorization_size=FIT_ITERS, silence_tqdm=True, krylov_rank=KRYLOV_RANK, \
                    krylov_eps=KRYLOV_EPS, idx_batch=idx_batch)
            idx_batch = case_2_model.fit(datasets[f'{prefix}_train'], datasets[f'{prefix}_test'], n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                    ams=LAMBDA*N*idx) 
            case_2_model.memorize(datasets[f'{prefix}_train'], memorization_size=FIT_ITERS, silence_tqdm=True, krylov_rank=KRYLOV_RANK, \
                    krylov_eps=KRYLOV_EPS, idx_batch=idx_batch) 
            _ = case_2_model.fit(datasets[f'{prefix}_train'], datasets[f'{prefix}_test'], n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                    ams=LAMBDA*N*idx, fl_reg=LAMBDA*N) 
            ## re-memorize to reflect information-reduced frontal lobe 
            case_2_model.memorize(datasets[f'{prefix}_train'], memorization_size=FIT_ITERS, silence_tqdm=True, krylov_rank=KRYLOV_RANK, \
                    krylov_eps=KRYLOV_EPS, idx_batch=idx_batch) 
            pass 
        ## gather results 
        metric_0 = case_0_model.accs
        metric_1 = case_1_model.accs 
        metric_2 = case_2_model.accs
        ## append condition codes 
        metric_0_tuples = [(x, 0) for x in metric_0] 
        metric_1_tuples = [(x, 1) for x in metric_1] 
        metric_2_tuples = [(x, 2) for x in metric_2] 
        ## format output 
        out = [] 
        def append_results(result_tuples, out=out): 
            cumulative_reward = 0 
            for iter_idx, r in enumerate(result_tuples): 
                acc = r[0] 
                condition = r[1] 
                ## return in-place 
                out.append((acc, iter_idx, task_idx, condition))
                pass 
            pass 
        append_results(metric_0_tuples) 
        append_results(metric_1_tuples) 
        append_results(metric_2_tuples) 
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
    ## save data 
    FILENAME = f'df-experiment-{EXPERIMENT_ID}'
    df_to_save = pd.DataFrame({'scores0': scores0, 
                               'scores1': scores1,
                               'scores2': scores2}) 
    df_data = df_to_save.to_csv().encode() 
    upload_to_blob_store(df_data, FILENAME+'.csv', sas_key, output_container_name) 
    ## save plot 
    plt.plot(scores0, label='0') 
    plt.plot(scores1, label='1') 
    plt.plot(scores2, label='2') 
    plt.legend() 
    plt.savefig(FILENAME+'.png') 
    with open(FILENAME+'.png', 'rb') as f: 
        upload_to_blob_store(f.read(), FILENAME+'.png', sas_key, output_container_name) 
        pass 
    pass 

if __name__ == '__main__': 
    phase_1() 
    phase_2() 
