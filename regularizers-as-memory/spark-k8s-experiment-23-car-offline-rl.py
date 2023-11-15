## Experiment 23: Offline RL with PiCar data 
## 
## Hypothesis: Optimal learning effects are observable in offline learning in a new game. 

import pandas as pd
from az_blob_util import upload_to_blob_store, download_from_blob_store, ls_blob_store, create_container 
from mnist_model import download_mnist_to_blob_storage 
import os 
from time import time 
from pyspark import SparkContext
from pyspark.sql import SparkSession 
sc = SparkContext() 
spark = SparkSession(sc) 

EXPERIMENT_ID = 23  
CAR_DATA_ZIP = 'car.zip' 
N_EXPERIMENTAL_ITERATIONS = 1000  
LAMBDA = 1.   
MAX_ITERS = 10 
BATCH_SIZE = 25 
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
        from car import Model
        from az_blob_util import upload_to_blob_store
        import os
        import pickle
        import zipfile 
        import random 
        from time import time 
        from tqdm import tqdm
        ## download data, unzip, write to disk  
        sas_key = os.environ['STORAGE_KEY']
        car_data_zip_bytes = download_from_blob_store(CAR_DATA_ZIP, sas_key, 'data') 
        data_dir = f'/tmp/car{task_idx}' 
        zip_path = os.path.join(data_dir, 'car.zip') 
        unzip_path = os.path.join(data_dir, 'unzipped') 
        os.mkdir(data_dir) 
        with open(zip_path, 'wb') as f: 
            f.write(car_data_zip_bytes) 
            pass 

        with zipfile.ZipFile(zip_path) as zip_ref: 
            zip_ref.extractall(unzip_path) 
            pass 

        ## randomly divide into test and train 
        unzip_path = os.path.join(unzip_path, 'car') 
        data_files = os.listdir(unzip_path) 
        data_files = [os.path.join(unzip_path, f) for f in data_files] 
        random.shuffle(data_files) 
        n_train = len(data_files) // 2 
        train = data_files[:n_train] 
        test = data_files[n_train:] 
        ## sort chronologically 
        train.sort() 
        test.sort() 
        ## init models 
        condition_0_model = Model() 
        condition_1_model = condition_0_model.copy() 
        ## or just have 1 big training dataset 
        data_files.sort() 
        ## run experiment 
        for i in range(len(data_files)): 
            print(f'[{time()}] task {task_idx} fitting dataset {i} of {len(data_files)}...') 
            ## load data 
            condition_0_model.load_car_env_data(data_files[i]) 
            condition_1_model.load_car_env_data(data_files[i]) 
            ## fit 
            n = len(data_files[i][0]) ## TODO this is probably wrong 
            for j in range(n//MAX_ITERS+1): 
                print(f'[{time()}] running optimization iteration {j} of {n//MAX_ITERS+1}...')
                _ = condition_0_model.optimize(max_iter=MAX_ITERS, batch_size=BATCH_SIZE) ## consider using an eval dataset  
                _ = condition_1_model.optimize(max_iter=MAX_ITERS, batch_size=BATCH_SIZE) 
                pass 
            ## memorize 
            print(f'[{time()}] memorizing...') 
            condition_1_model.convert_observations_to_memory(krylov_rank=KRYLOV_RANK, disable_tqdm=True) 
            print(f'[{time()}] memorization complete!') 
            pass 
        ## append condition codes
        condition_0_result_tuples = [(r, False, i, 0) for i, r in enumerate(condition_0_model.mean_rewards)] 
        condition_1_result_tuples = [(r, False, i, 1) for i, r in enumerate(condition_1_model.mean_rewards)] 
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
                               'scores5': scores1}) 
    df_data = df_to_save.to_csv().encode() 
    upload_to_blob_store(df_data, FILENAME+'.csv', sas_key, output_container_name) 
    ## save plot 
    plt.plot(scores0, label='0') 
    plt.plot(scores1, label='1') 
    plt.legend() 
    plt.savefig(FILENAME+'.png') 
    with open(FILENAME+'.png', 'rb') as f: 
        upload_to_blob_store(f.read(), FILENAME+'.png', sas_key, output_container_name) 
        pass 
    pass 

if __name__ == '__main__': 
    phase_1() 
    phase_2() 
