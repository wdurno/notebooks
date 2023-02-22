## Experiment 14: online learning 

import pandas as pd
from az_blob_util import upload_to_blob_store, download_from_blob_store, ls_blob_store 
import os 
from pyspark import SparkContext
from pyspark.sql import SparkSession 
sc = SparkContext() 
spark = SparkSession(sc) 

N_EXPERIMENTAL_ITERATIONS = 1000  
LAMBDA = 1.  
ACC_FREQ=10
FIT_ITERS = 1000 
SAMPLING_STRIDE = 10000 
SAMPLING_WINDOW = 10000 
KRYLOV_EPS = 0. 
L2_REG = .00001 ## centered at last memorized point 

def map1(task_idx):
    import traceback 
    try:
        task_idx = int(task_idx) 
        ## run experiment 
        from nlp import Model, EvenOrOddNLPDataset, NLPDataset, nlp_train, nlp_test, shakes_tokens_train, SHAKES_SERIES_LEN, BATCH_SIZE   
        from az_blob_util import upload_to_blob_store 
        import os 
        import pickle 
        from tqdm import tqdm 
        ## artificially reducing dataset size to speed-up experiments 
        shakes_tokens_train = shakes_tokens_train[:100000] 
        ## define models 
        control_0_model = Model(net_type='nlp', batch_norm=False, log1p_reg=False) ## only fit to windowed data, and don't memorize 
        control_1_model = control_0_model.copy() ## only fit to windowed data, and do memorize 
        experimental_model = control_0_model.copy() ## fit to all data observed thus far 
        ## generate test datasets 
        nlp_even_test = EvenOrOddNLPDataset(nlp_test, even=True) 
        nlp_odd_test = EvenOrOddNLPDataset(nlp_test, even=False) 
        ## define sampling process 
        sampling_indices = tqdm([idx*SAMPLING_STRIDE for idx in range(len(shakes_tokens_train) // SAMPLING_STRIDE)]) 
        N = len(sampling_indices)  
        for idx in sampling_indices: 
            ## define training datasets 
            windowed_dataset_train = NLPDataset(token_list=shakes_tokens_train[idx:(idx+SAMPLING_WINDOW)], sample_length=SHAKES_SERIES_LEN) 
            cumulative_dataset_train = NLPDataset(token_list=shakes_tokens_train[:(idx+SAMPLING_WINDOW)], sample_length=SHAKES_SERIES_LEN) 
            ## fit models 
            control_0_model.fit(windowed_dataset_train, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                    nlp_even_test=nlp_even_test, nlp_odd_test=nlp_odd_test) 
            control_1_model.fit(cumulative_dataset_train, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                    nlp_even_test=nlp_even_test, nlp_odd_test=nlp_odd_test)
            idx_batch = experimental_model.fit(windowed_dataset_train, n_iters=FIT_ITERS, silence_tqdm=True, acc_frequency=ACC_FREQ, \
                    nlp_even_test=nlp_even_test, nlp_odd_test=nlp_odd_test, ams=LAMBDA, l2_reg=L2_REG) 
            ## memorize 
            experimental_model.memorize(windowed_dataset_train, memorization_size=FIT_ITERS, silence_tqdm=True, krylov_rank=10, \
                    krylov_eps=KRYLOV_EPS, idx_batch=idx_batch) 
            pass 
        ## gather results 
        metric_0 = [(a+b)*.5 for a,b in zip(control_0_model.accs_low, control_0_model.accs_high)] ## TODO replace this ugly, approximate hack  
        metric_1 = [(a+b)*.5 for a,b in zip(control_1_model.accs_low, control_1_model.accs_high)] 
        metric_2 = [(a+b)*.5 for a,b in zip(experimental_model.accs_low, experimental_model.accs_high)] 
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
        filename = f'result-{task_idx}.pkl'
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
    FILENAME = 'df-experiment-14'
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
