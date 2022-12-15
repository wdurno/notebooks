## Experiment 11: Language model transfer learning from odd to even tokens  

import pandas as pd
from az_blob_util import upload_to_blob_store, download_from_blob_store, ls_blob_store 
import os 
from pyspark import SparkContext
from pyspark.sql import SparkSession 
sc = SparkContext() 
spark = SparkSession(sc) 

SAMPLE_SIZE = 100  
LAMBDA = 1e2 #1e1 
ACC_FREQ=100
RANDOM_LABEL_PROBABILITY=.0
N_ITERS = 310
N_TRANSFER_LEARNING_SIZE = 10000 
EXTRA_FIT_CYCLES=5000  

def map1(task_idx):
    import traceback 
    try:
        task_idx = int(task_idx) 
        ## run experiment 
        from nlp import Model, EvenOrOddNLPDataset, nlp_train, nlp_test  
        from az_blob_util import upload_to_blob_store 
        import os 
        import pickle 
        ## generate datasets 
        nlp_even_train = EvenOrOddNLPDataset(nlp_train, even=True, random_subset_size=N_TRANSFER_LEARNING_SIZE) ## subset demonstrates transfer learning 
        nlp_odd_train = EvenOrOddNLPDataset(nlp_train, even=False)
        nlp_even_test = EvenOrOddNLPDataset(nlp_test, even=True)
        nlp_odd_test = EvenOrOddNLPDataset(nlp_test, even=False) 
        ## fit models 
        model = Model(net_type='nlp', batch_norm=False, log1p_reg=False) 
        model.fit(nlp_odd_train, n_iters=EXTRA_FIT_CYCLES, silence_tqdm=True, acc_frequency=ACC_FREQ, random_label_probability=RANDOM_LABEL_PROBABILITY, nlp_even_test=nlp_even_test, nlp_odd_test=nlp_odd_test) 
        model = model.copy() ## double-copy because I've seen `copy` influence behavior. So, we apply it to all cases.   
        ams_model = model.copy() 
        ams_model.memorize(nlp_odd_train, memorization_size=N_ITERS, silence_tqdm=True, krylov_rank=10, krylov_eps=1.)  
        ams_model = ams_model.copy() 
        ## apply a small dataset 
        model.fit(nlp_even_train, n_iters=5*EXTRA_FIT_CYCLES, silence_tqdm=True, acc_frequency=ACC_FREQ, random_label_probability=RANDOM_LABEL_PROBABILITY, nlp_even_test=nlp_even_test, nlp_odd_test=nlp_odd_test) 
        _ = ams_model.fit(nlp_even_train, n_iters=1*EXTRA_FIT_CYCLES, silence_tqdm=True, ams=75.*LAMBDA, acc_frequency=ACC_FREQ, halt_acc=.0, random_label_probability=RANDOM_LABEL_PROBABILITY, nlp_even_test=nlp_even_test, nlp_odd_test=nlp_odd_test) ## demonstrate retention 
        idx_batch = ams_model.fit(nlp_even_train, n_iters=2*EXTRA_FIT_CYCLES, silence_tqdm=True, ams=1.*LAMBDA, acc_frequency=ACC_FREQ, halt_acc=.0, random_label_probability=RANDOM_LABEL_PROBABILITY, nlp_even_test=nlp_even_test, nlp_odd_test=nlp_odd_test) 
        ## update memory, and continue fitting 
        ams_model.memorize(nlp_even_train, memorization_size=N_ITERS, silence_tqdm=True, krylov_rank=10, krylov_eps=1., idx_batch=idx_batch) 
        ams_model.fit(nlp_even_train, n_iters=2*EXTRA_FIT_CYCLES, silence_tqdm=True, ams=100.*LAMBDA, acc_frequency=ACC_FREQ, halt_acc=.0, random_label_probability=RANDOM_LABEL_PROBABILITY, nlp_even_test=nlp_even_test, nlp_odd_test=nlp_odd_test) 
        ## gather results 
        metric_0 = model.accs_low 
        metric_1 = model.accs_high 
        metric_2 = ams_model.accs_low 
        metric_3 = ams_model.accs_high 
        ## append condition codes 
        metric_0_tuples = [(x, 0) for x in metric_0] 
        metric_1_tuples = [(x, 1) for x in metric_1] 
        metric_2_tuples = [(x, 2) for x in metric_2] 
        metric_3_tuples = [(x, 3) for x in metric_3] 
        ## format output 
        out = [] 
        def append_results(result_tuples, out=out): 
            cumulative_reward = 0 
            for iter_idx, r in enumerate(result_tuples): 
                acc = r[0] 
                condition = r[1] 
                ## return in-place 
                t = (acc, iter_idx, task_idx, condition) 
                if len(t) == 4: 
                    out.append((acc, iter_idx, task_idx, condition))
                else: 
                    ## had 5 once, not sure why 
                    print(f'WARNING: len(t) != 4, t: {t}') 
                pass 
            pass 
        append_results(metric_0_tuples) 
        append_results(metric_1_tuples) 
        append_results(metric_2_tuples) 
        append_results(metric_3_tuples) 
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
    x = sc.parallelize(list(range(SAMPLE_SIZE)), SAMPLE_SIZE) 
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
    scores3 = df.loc[df['condition'] == 3].sort_values('iter')['avg(score)'].tolist() 
    ## save data 
    FILENAME = 'df-experiment-11'
    df_to_save = pd.DataFrame({'scores0': scores0, 
                               'scores1': scores1, 
                               'scores2': scores2,
                               'scores3': scores3}) 
    df_data = df_to_save.to_csv().encode() 
    upload_to_blob_store(df_data, FILENAME+'.csv', sas_key, output_container_name) 
    ## save plot 
    plt.plot(scores0, label='0') 
    plt.plot(scores1, label='1') 
    plt.plot(scores2, label='2') 
    plt.plot(scores3, label='3') 
    plt.legend() 
    plt.savefig(FILENAME+'.png') 
    with open(FILENAME+'.png', 'rb') as f: 
        upload_to_blob_store(f.read(), FILENAME+'.png', sas_key, output_container_name) 
        pass 
    pass 

if __name__ == '__main__': 
    phase_1() 
    phase_2() 
