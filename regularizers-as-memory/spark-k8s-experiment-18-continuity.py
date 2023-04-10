## Experiment 18: experiments involving continuous deformation of sampling distributions, similar to RL  
## 
## ## Experimental conditions: 
## control 0: idealized and theoretically impossible; fit only to odds, use data from future and past. No memorization.  
## control 1: idealized (p=1); only sample odds; but still apply memorization 
## control 2: sub-optimal (p=0); fit exclusively to evens, then to odds  
## experimental 0: fit to p-mixture of evens and odds, then odds exclusively. No memorization.  
## experimental 1: same^^^, but larger p. With memorization
## experimental 2: same^^^, but larger p. With memorization

import pandas as pd
from az_blob_util import upload_to_blob_store, download_from_blob_store, ls_blob_store 
from mnist_model import download_mnist_to_blob_storage 
import os 
from pyspark import SparkContext
from pyspark.sql import SparkSession 
sc = SparkContext() 
spark = SparkSession(sc) 

EXPERIMENT_ID = 18  
N_EXPERIMENTAL_ITERATIONS = 300  
LAMBDA = 1.   
ACC_FREQ=10
FIT_ITERS = 500  
PHASE_1_SAMPLE_SIZE = 100 
PHASE_2_SAMPLE_SIZE = 20 
KRYLOV_EPS = 0. 
L2_REG = None  
KRYLOV_RANK = 10 

## download mnist once, then distribute via blob storage 
## this isn't just a courtesy, because the download is rate-limited 
download_mnist_to_blob_storage() 

def map1(task_idx):
    import traceback 
    try:
        task_idx = int(task_idx) 
        ## run experiment 
        from mnist_model import Model, get_datasets, probability_merge_datasets 
        from az_blob_util import upload_to_blob_store 
        import os 
        import pickle 
        from tqdm import tqdm 
        ## get data 
        mnist_train_1_evens, mnist_train_1_odds, mnist_test_evens, mnist_test_odds = get_datasets(n=PHASE_1_SAMPLE_SIZE) 
        _, mnist_train_2_odds, _, _ = get_datasets(n=PHASE_2_SAMPLE_SIZE) 
        p0_train = probability_merge_datasets(mnist_train_1_evens, mnist_train_1_odds, PHASE_1_SAMPLE_SIZE, .1) 
        p1_train = probability_merge_datasets(mnist_train_1_evens, mnist_train_1_odds, PHASE_1_SAMPLE_SIZE, .5) 
        p2_train = probability_merge_datasets(mnist_train_1_evens, mnist_train_1_odds, PHASE_1_SAMPLE_SIZE, .9) 
        ## define initial model  
        control_0_model = Model(net_type='cnn', batch_norm=False, log1p_reg=False) 
        control_1_model = control_0_model.copy() 
        control_2_model = control_0_model.copy() 
        experimental_0_model = control_0_model.copy() 
        experimental_1_model = control_0_model.copy() 
        experimental_2_model = control_0_model.copy() 
        ## execute conditions 
        ## control 0 
        _ = control_0_model.fit(mnist_train_1_odds + mnist_train_2_odds, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                even_label_test_set=mnist_test_evens, odd_label_test_set=mnist_test_odds) 
        _ = control_0_model.fit(mnist_train_1_odds, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                even_label_test_set=mnist_test_evens, odd_label_test_set=mnist_test_odds, ams=False) 
        ## control 1 
        idx_batch = control_1_model.fit(mnist_train_1_odds, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                even_label_test_set=mnist_test_evens, odd_label_test_set=mnist_test_odds) 
        control_1_model.memorize(mnist_train_1_odds, memorization_size=PHASE_1_SAMPLE_SIZE, silence_tqdm=True, krylov_rank=KRYLOV_RANK, \
                krylov_eps=KRYLOV_EPS, idx_batch=idx_batch) 
        _ = control_1_model.fit(mnist_train_2_odds, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                even_label_test_set=mnist_test_evens, odd_label_test_set=mnist_test_odds, ams=LAMBDA) 
        ## control 2 
        idx_batch = control_2_model.fit(mnist_train_1_evens, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                even_label_test_set=mnist_test_evens, odd_label_test_set=mnist_test_odds)
        control_2_model.memorize(mnist_train_1_evens, memorization_size=PHASE_1_SAMPLE_SIZE, silence_tqdm=True, krylov_rank=KRYLOV_RANK, \
                krylov_eps=KRYLOV_EPS, idx_batch=idx_batch)
        _ = control_2_model.fit(mnist_train_2_odds, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                even_label_test_set=mnist_test_evens, odd_label_test_set=mnist_test_odds, ams=LAMBDA) 
        ## experimental 0 
        idx_batch = experimental_0_model.fit(p0_train, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                even_label_test_set=mnist_test_evens, odd_label_test_set=mnist_test_odds)
        #experimental_0_model.memorize(p0_train, memorization_size=PHASE_1_SAMPLE_SIZE, silence_tqdm=True, krylov_rank=KRYLOV_RANK, \
        #        krylov_eps=KRYLOV_EPS, idx_batch=idx_batch)
        _ = experimental_0_model.fit(mnist_train_2_odds, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                even_label_test_set=mnist_test_evens, odd_label_test_set=mnist_test_odds, ams=LAMBDA)
        ## experimental 1 
        idx_batch = experimental_1_model.fit(p0_train, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                even_label_test_set=mnist_test_evens, odd_label_test_set=mnist_test_odds)
        experimental_1_model.memorize(p1_train, memorization_size=PHASE_1_SAMPLE_SIZE, silence_tqdm=True, krylov_rank=KRYLOV_RANK, \
                krylov_eps=KRYLOV_EPS, idx_batch=idx_batch)
        _ = experimental_1_model.fit(mnist_train_2_odds, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                even_label_test_set=mnist_test_evens, odd_label_test_set=mnist_test_odds, ams=LAMBDA)
        ## experimental 2 
        idx_batch = experimental_2_model.fit(p2_train, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                even_label_test_set=mnist_test_evens, odd_label_test_set=mnist_test_odds)
        experimental_2_model.memorize(p2_train, memorization_size=PHASE_1_SAMPLE_SIZE, silence_tqdm=True, krylov_rank=KRYLOV_RANK, \
                krylov_eps=KRYLOV_EPS, idx_batch=idx_batch)
        _ = experimental_2_model.fit(mnist_train_2_odds, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                even_label_test_set=mnist_test_evens, odd_label_test_set=mnist_test_odds, ams=LAMBDA)
        ## gather results 
        metric_0 = control_0_model.accs_odd
        metric_1 = control_1_model.accs_odd 
        metric_2 = control_2_model.accs_odd
        metric_3 = experimental_0_model.accs_odd
        metric_4 = experimental_1_model.accs_odd 
        metric_5 = experimental_2_model.accs_odd 
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
