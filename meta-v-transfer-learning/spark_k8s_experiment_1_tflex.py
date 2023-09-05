## Experiment 1: Transfer learning with flexibility "tflex"  

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
FIT_ITERS = 2*50 ## must be divisible 
SUBSAMPLE_SIZE = 20 
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
        from mnist_model import Classifier, get_datasets, sample 
        from az_blob_util import upload_to_blob_store 
        import os 
        import pickle 
        from tqdm import tqdm 
        ## get data 
        mnist_train, mnist_test = get_datasets() 
        ## for out-of-sample modelling 
        print('sub-sampling data...') 
        mnist_train_small_subset = sample(SUBSAMPLE_SIZE, sub_sample=True, dataset=mnist_train) 
        mnist_train_9 = [(image, label) for (image, label) in mnist_train if label == 9] 
        mnist_train_not_9 = [(image, label) for (image, label) in mnist_train if label != 9] 
        mnist_test_9 = [(image, label) for (image, label) in mnist_test if label == 9] 
        mnist_test_not_9 = [(image, label) for (image, label) in mnist_test if label != 9] 
        ## base_model gets a large dataset, so serves as the transfer learning source 
        ## note the omission of 9s from the dataset, enabling out-of-sample experiments 
        print('fitting base model...') 
        base_model = Classifier() 
        base_accs = [] 
        for _ in range(FIT_ITERS/2): ## burn-in 
            base_accs.append(big_model.fit(batch_size=100, train_data=mnist_train_not_9, eval_dataset=mnist_test_not_9)) 
            pass 
        for _ in range(FIT_ITERS/2): ## consolidate 
            base_accs.append(big_model.fit(batch_size=100, train_data=mnist_train_not_9, eval_dataset=mnist_test_not_9, memorize=True)) 
            pass 
        ## control: infinite lambda (classic transfer learning) 
        print('running control 1...') 
        accs_1 = [] 
        control_model = Classifier(base_layer_transfer=base_model.base_layer, infinite_lambda=True) 
        for _ in range(FIT_ITERS): 
            accs_1.append(control_model.fit(train_data=mnist_train_small_subset, use_memory=True, eval_dataset=mnist_test_9)) 
            pass 
        ## experimental 1: lambda = 1. 
        print('running first experiment...') 
        lmbda = 1. 
        accs_2 = [] 
        experimental_model = Classifier(base_layer_transfer=base_model.base_layer, infinite_lambda=False) 
        for _ in range(FIT_ITERS): 
            accs_2.append(control_model.fit(train_data=mnist_train_small_subset, use_memory=lmbda, eval_dataset=mnist_test_9)) 
            pass 
        ## experimental 2: lambda = .01 
        print('running second experiment...') 
        lmbda = .01 
        accs_3 = [] 
        experimental_model = Classifier(base_layer_transfer=base_model.base_layer, infinite_lambda=False) 
        for _ in range(FIT_ITERS): 
            accs_3.append(control_model.fit(train_data=mnist_train_small_subset, use_memory=lmbda, eval_dataset=mnist_test_9)) 
            pass 
        ## gather results 
        metric_0 = base_accs 
        metric_1 = accs_1 
        metric_2 = accs_2
        metric_3 = accs_3 
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
                out.append((acc, iter_idx, task_idx, condition))
                pass 
            pass 
        append_results(metric_0_tuples) 
        append_results(metric_1_tuples) 
        append_results(metric_2_tuples) 
        append_results(metric_3_tuples) 
        ## write-out accuracies  
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
    scores3 = df.loc[df['condition'] == 3].sort_values('iter')['avg(score)'].tolist() 
    ## save data 
    FILENAME = f'df-experiment-{EXPERIMENT_ID}'
    df_to_save = pd.DataFrame({'scores0': scores0, 
                               'scores1': scores1,
                               'scores2': scores2, 
                               'scores3': scores3}) 
    df_data = df_to_save.to_csv().encode() 
    upload_to_blob_store(df_data, FILENAME+'.csv', sas_key, output_container_name) 
    ## save plot 
    fig, fig_ax = plt.subplots() 
    fig_ax.plot(scores0, label='0') 
    fig_ax.plot(scores1, label='1') 
    fig_ax.plot(scores2, label='2') 
    fig_ax.plot(scores3, label='3') 
    fig_ax.legend() 
    fig.savefig(FILENAME+'.png') 
    with open(FILENAME+'.png', 'rb') as f: 
        upload_to_blob_store(f.read(), FILENAME+'.png', sas_key, output_container_name) 
        pass 
    pass 

if __name__ == '__main__': 
    phase_1() 
    phase_2() 
