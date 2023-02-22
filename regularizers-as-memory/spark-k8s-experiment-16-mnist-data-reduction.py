## Experiment 16: memory as regularization 

import pandas as pd
from az_blob_util import upload_to_blob_store, download_from_blob_store, ls_blob_store 
from mnist_model import download_mnist_to_blob_storage 
import os 
from pyspark import SparkContext
from pyspark.sql import SparkSession 
sc = SparkContext() 
spark = SparkSession(sc) 

EXPERIMENT_ID = 16 
N_EXPERIMENTAL_ITERATIONS = 1000  
LAMBDA = 1.  
ACC_FREQ=10
FIT_ITERS = 1000 
SUBSAMPLE_SIZE = 1000 
SAMPLING_STRIDE = 100 
SAMPLING_WINDOW = 100 
KRYLOV_EPS = 0. 
L2_REG = 0. 

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
        mnist_train_evens, mnist_test_evens, mnist_train_odds, mnist_test_odds = get_datasets(n=SUBSAMPLE_SIZE) 
        ## define models 
        control_0_model = Model(net_type='dense', batch_norm=True, log1p_reg=False) ## only fit to windowed data, and don't memorize 
        control_1_model = control_0_model.copy() ## cumulative datasets, and don't memorize 
        experimental_model = control_0_model.copy() ## only fit to windowed datasets and do memorize  
        ## define sampling process 
        sampling_indices = tqdm([idx*SAMPLING_STRIDE for idx in range(len(mnist_train_evens) // SAMPLING_STRIDE)]) 
        N = len(sampling_indices)  
        for idx in sampling_indices: 
            ## define training datasets 
            windowed_train_evens = mnist_train_evens[idx:(idx+SAMPLING_WINDOW)] 
            windowed_train_odds = mnist_train_odds[idx:(idx+SAMPLING_WINDOW)] 
            cumulative_train_evens = mnist_train_evens[:(idx+SAMPLING_WINDOW)] 
            cumulative_train_odds = mnist_train_odds[:(idx+SAMPLING_WINDOW)] 
            ## fit models 
            control_0_model.fit(windowed_train_evens, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                    even_label_test_set=mnist_test_evens, odd_label_test_set=mnist_test_odds) 
            control_0_model.fit(windowed_train_odds, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                    even_label_test_set=mnist_test_evens, odd_label_test_set=mnist_test_odds) 
            control_1_model.fit(cumulative_train_evens, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                    even_label_test_set=mnist_test_evens, odd_label_test_set=mnist_test_odds) 
            control_1_model.fit(cumulative_train_odds, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                    even_label_test_set=mnist_test_evens, odd_label_test_set=mnist_test_odds)
            idx_batch_even = experimental_model.fit(windowed_train_evens, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                    even_label_test_set=mnist_test_evens, odd_label_test_set=mnist_test_odds) 
            idx_batch_odd = experimental_model.fit(windowed_train_odds, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                    even_label_test_set=mnist_test_evens, odd_label_test_set=mnist_test_odds)
            ## memorize 
            experimental_model.memorize(windowed_train_evens, memorization_size=FIT_ITERS, silence_tqdm=True, krylov_rank=10, \
                    krylov_eps=KRYLOV_EPS, idx_batch=idx_batch_even) 
            experimental_model.memorize(windowed_train_odds, memorization_size=FIT_ITERS, silence_tqdm=True, krylov_rank=10, \
                    krylov_eps=KRYLOV_EPS, idx_batch=idx_batch_odd) 
            pass 
        ## gather results 
        metric_0 = control_0_model.accs_low
        metric_1 = control_0_model.accs_high 
        metric_2 = control_1_model.accs_low
        metric_3 = control_1_model.accs_high
        metric_4 = experimental_model.accs_low 
        metric_5 = experimental_model.accs_high 
        ## append condition codes 
        metric_0_tuples = [(x, 0) for x in metric_0] 
        metric_1_tuples = [(x, 1) for x in metric_1] 
        metric_2_tuples = [(x, 2) for x in metric_2] 
        metric_3_tuples = [(x, 3) for x in metric_3] 
        metric_4_tuples = [(x, 4) for x in metric_4]
        metric_5_tuples = [(x, 5) for x in metric_5] 
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
        append_results(metric_3_tuples) 
        append_results(metric_4_tuples) 
        append_results(metric_5_tuples) 
        ## write out 
        filename = f'experiment-{EXPERIMENT_ID}-result-{task_idx}.pkl'
        sas_key = os.environ['STORAGE_KEY']
        output_container_name = 'tmp'
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
    container_name = 'tmp' 
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
    input_container_name = 'tmp' 
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
    plt.plot(scores3, label='4') 
    plt.plot(scores3, label='5') 
    plt.legend() 
    plt.savefig(FILENAME+'.png') 
    with open(FILENAME+'.png', 'rb') as f: 
        upload_to_blob_store(f.read(), FILENAME+'.png', sas_key, output_container_name) 
        pass 
    pass 

if __name__ == '__main__': 
    phase_1() 
    phase_2() 
