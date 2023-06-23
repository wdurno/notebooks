## Experiment 1: use frontal lobe norms to clear a long-term memory cache between batches 

import pandas as pd
from az_blob_util import create_container, upload_to_blob_store, download_from_blob_store, ls_blob_store 
from mnist_model import download_mnist_to_blob_storage 
import os 
from time import time 
from pyspark import SparkContext
from pyspark.sql import SparkSession 
from pyspark.sql.functions import mean, stddev 
if __name__ == '__main__':
    sc = SparkContext() 
    spark = SparkSession(sc) 
    pass 

EXPERIMENT_ID = 1 
N_EXPERIMENTAL_ITERATIONS = 10 # 100  
LAMBDA = 10000.   
ACC_FREQ=10
FIT_ITERS = 100 # 1000 
SUBSAMPLE_SIZE = 400 
N = 1000  
KRYLOV_EPS = 0. 
L2_REG = None  
KRYLOV_RANK = 2 
BATCH_PREFIXES = ['01', '23', '45', '67', '89']*2 
BATCH_PREFIXES = BATCH_PREFIXES[:2] ## TODO REMOVE! This just speeds-up testing! 
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
        case_2_model_fl_param_names = [p[0] for p in case_2_model.named_fl_params] 
        case_2_model_non_fl_params = [p[1] for p in case_2_model.named_parameters() if p[0] not in case_2_model_fl_param_names] 
        ## test datasets are cumulative
        ## so we test against prior batches as well, thereby evaluating memory 
        cumulative_test_dataset = [] 
        ## fit 
        for idx, prefix in enumerate(BATCH_PREFIXES): 
            ## update test dataset 
            print(f'{idx}/{len(BATCH_PREFIXES)} extending test dataset...')
            cumulative_test_dataset += datasets[f'{prefix}_test'] 
            ## run experiment 
            print(f'{idx}/{len(BATCH_PREFIXES)} fitting case_0_model...') 
            _ = case_0_model.fit(datasets[f'{prefix}_train'], cumulative_test_dataset, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ) 
            print(f'{idx}/{len(BATCH_PREFIXES)} fitting case_1_model...') 
            idx_batch = case_1_model.fit(datasets[f'{prefix}_train'], cumulative_test_dataset, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                    ams=LAMBDA*N*idx) 
            print(f'{idx}/{len(BATCH_PREFIXES)} case_1_model memorizing...') 
            case_1_model.memorize(datasets[f'{prefix}_train'], memorization_size=FIT_ITERS, silence_tqdm=True, krylov_rank=KRYLOV_RANK, \
                    krylov_eps=KRYLOV_EPS, idx_batch=idx_batch) 
            print(f'{idx}/{len(BATCH_PREFIXES)} case_2_model first fit...') 
            idx_batch = case_2_model.fit(datasets[f'{prefix}_train'], cumulative_test_dataset, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                    ams=LAMBDA*N*idx, parameters=case_2_model_non_fl_params) ## TODO "parameters" name is terrible. only applies to memory  
            print(f'{idx}/{len(BATCH_PREFIXES)} case_2_model first memorization...') ## TODO remove unneeded experimental cases  
            case_2_model.memorize(datasets[f'{prefix}_train'], memorization_size=FIT_ITERS, silence_tqdm=True, krylov_rank=KRYLOV_RANK, \
                    krylov_eps=KRYLOV_EPS, idx_batch=idx_batch, parameters=case_2_model_non_fl_params) 
            print(f'{idx}/{len(BATCH_PREFIXES)} case_2_model second fit...') 
            _ = case_2_model.fit(datasets[f'{prefix}_train'], cumulative_test_dataset, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                    ams=LAMBDA*N*idx/100, fl_reg=LAMBDA*N*100, parameters=case_2_model_non_fl_params) 
            ## re-memorize to reflect information-reduced frontal lobe 
            print(f'{idx}/{len(BATCH_PREFIXES)} case_2_model second memorization...') 
            case_2_model.memorize(datasets[f'{prefix}_train'], memorization_size=FIT_ITERS, silence_tqdm=True, krylov_rank=KRYLOV_RANK, \
                    krylov_eps=KRYLOV_EPS, idx_batch=idx_batch, parameters=case_2_model_non_fl_params) 
            ## pad accs 
            case_2_accs_n = len(case_2_model.accs) 
            case_0_model.accs.extend([case_0_model.accs[-1]]*(case_2_accs_n - len(case_0_model.accs))) 
            case_1_model.accs.extend([case_1_model.accs[-1]]*(case_2_accs_n - len(case_1_model.accs))) 
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
        ## write-out accuracies  
        filename = f'experiment-{EXPERIMENT_ID}-result-{task_idx}.pkl'
        sas_key = os.environ['STORAGE_KEY'] 
        output_container_name = TEMP_CONTAINER_NAME 
        upload_to_blob_store(pickle.dumps(out), filename, sas_key, output_container_name)  
        ## write-out information diagonal 
        filename = f'experiment-{EXPERIMENT_ID}-info-vec-{task_idx}.pkl'  
        case_1_model_info_vec = case_1_model.get_information_diagonal() ## just using control for now... will do something more-impressive later 
        info_df_rows = [] 
        for param_idx, info_val in enumerate(case_1_model_info_vec.tolist()): 
            info_val = float(info_val[0]) ## column vec is a list of single-element lists 
            info_df_rows.append((param_idx, info_val, task_idx)) 
            pass 
        upload_to_blob_store(pickle.dumps(info_df_rows), filename, sas_key, output_container_name) 
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
    'fit models' 
    x = sc.parallelize(list(range(N_EXPERIMENTAL_ITERATIONS)), N_EXPERIMENTAL_ITERATIONS) 
    y = x.map(map1) 
    return y.collect() 

def phase_2(): 
    'aggregate accuracies' 
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
    fig, fig_ax = plt.subplots() 
    fig_ax.plot(scores0, label='0') 
    fig_ax.plot(scores1, label='1') 
    fig_ax.plot(scores2, label='2') 
    fig_ax.legend() 
    fig.savefig(FILENAME+'.png') 
    with open(FILENAME+'.png', 'rb') as f: 
        upload_to_blob_store(f.read(), FILENAME+'.png', sas_key, output_container_name) 
        pass 
    pass 

def phase_3(): 
    'aggregate info vecs into mean and std vecs' 
    ## get data 
    import pandas as pd
    import matplotlib.pyplot as plt
    ## config 
    sas_key = os.environ['STORAGE_KEY']
    input_container_name = TEMP_CONTAINER_NAME
    output_container_name = 'data'
    ## get data 
    filenames = ls_blob_store('', sas_key, input_container_name)
    filenames = [f for f in filenames if f.startswith(f'experiment-{EXPERIMENT_ID}-info-vec-') and f.endswith('.pkl')] 
    filenames = sc.parallelize(filenames, len(filenames))
    y = filenames.flatMap(map2) 
    schema = ['param_idx', 'info_val', 'task_idx'] 
    y = y.toDF(schema=schema) 
    param_groups = y.groupBy('param_idx') 
    aggs = param_groups.agg(mean('info_val'), stddev('info_val'))  
    df = aggs.toPandas() 
    means = df['avg(info_val)'].tolist() 
    stds = (df['stddev_samp(info_val)'] + df['avg(info_val)']).tolist() 
    fig, fig_ax = plt.subplots() 
    fig_ax.plot(stds, label='means + stds') 
    fig_ax.plot(means, label='means') 
    fig_ax.legend() 
    FILENAME = f'df-infos-{EXPERIMENT_ID}' 
    fig.savefig(FILENAME+'.png') 
    with open(FILENAME+'.png', 'rb') as f:
        upload_to_blob_store(f.read(), FILENAME+'.png', sas_key, output_container_name)
        pass
    pass 

if __name__ == '__main__': 
    phase_1() 
    phase_2() 
    DEBUG = phase_3() 
    print(DEBUG) ## TODO remove this! 
