## Experiment 15: Extreme dimensional reductions 
## Using optimization program duality to restrict training parameter space to a very low dimensionality  
## WARNING: Requires EMPTY storage container "tmp"!  

import pandas as pd
from az_blob_util import upload_to_blob_store, download_from_blob_store, ls_blob_store 
import os 
from pyspark import SparkContext
from pyspark.sql import SparkSession 
try: 
    sc = SparkContext() 
    spark = SparkSession(sc) 
except: 
    ## sc + spark probably already exist 
    pass  

SAMPLE_SIZE = 1000  
ITERS = 1000

def map1(task_idx):
    import traceback 
    try:
        task_idx = int(task_idx) 
        ## run experiment 
        from regmem import Model 
        from az_blob_util import upload_to_blob_store 
        import os 
        import pickle 
        condition_0_model = Model() 
        ## condition 0 (control): No use of memory, no discarding of data 
        condition_0_result_tuples_before = condition_0_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        ## copy, creating other models before continuing 
        condition_1_model = condition_0_model.copy() 
        condition_2_model = condition_0_model.copy() 
        condition_3_model = condition_0_model.copy() 
        ## continue condition 0 (control) Use memory, do discard data, memorize once  
        condition_0_model.convert_observations_to_memory() 
        condition_0_result_tuples_after_1 = condition_0_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        condition_0_result_tuples_after_2 = condition_0_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        condition_0_result_tuples_after_3 = condition_0_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        ## condition 1 (experimental): Use memory, do dicard data, memorize twice  
        condition_1_model.convert_observations_to_memory() 
        condition_1_result_tuples_after_1 = condition_1_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        condition_1_model.convert_observations_to_memory() 
        condition_1_result_tuples_after_2 = condition_1_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        condition_1_result_tuples_after_3 = condition_1_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        ## condition 2 (experimental): Use memory, do discard data, memorize thrice 
        condition_2_model.convert_observations_to_memory() 
        condition_2_result_tuples_after_1 = condition_2_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        condition_2_model.convert_observations_to_memory() 
        condition_2_result_tuples_after_2 = condition_2_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        condition_2_model.convert_observations_to_memory() 
        condition_2_result_tuples_after_3 = condition_2_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        ## condition 3 (experimental): Use memory, do discard data, memorize thrice, use eta-space dimensional reduction 
        condition_3_model.convert_observations_to_memory(n_eigenvectors=10, n_reparameterized_dims=20) 
        condition_3_result_tuples_after_1 = condition_3_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        condition_3_model.convert_observations_to_memory(n_eigenvectors=10, n_reparameterized_dims=20) 
        condition_3_result_tuples_after_2 = condition_3_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        condition_3_model.convert_observations_to_memory(n_eigenvectors=10, n_reparameterized_dims=20) 
        condition_3_result_tuples_after_3 = condition_3_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        ## merge before & after results 
        condition_0_result_tuples = condition_0_result_tuples_before + \
                condition_0_result_tuples_after_1 + \
                condition_0_result_tuples_after_2 + \
                condition_0_result_tuples_after_3 
        condition_1_result_tuples = condition_0_result_tuples_before + \
                condition_1_result_tuples_after_1 + \
                condition_1_result_tuples_after_2 + \
                condition_1_result_tuples_after_3 
        condition_2_result_tuples = condition_0_result_tuples_before + \
                condition_2_result_tuples_after_1 + \
                condition_2_result_tuples_after_2 + \
                condition_2_result_tuples_after_3 
        condition_3_result_tuples = condition_0_result_tuples_before + \
                condition_3_result_tuples_after_1 + \
                condition_3_result_tuples_after_2 + \
                condition_3_result_tuples_after_3 
        ## append condition codes 
        condition_0_result_tuples = [(x[0], x[1], x[2], 0) for x in condition_0_result_tuples] 
        condition_1_result_tuples = [(x[0], x[1], x[2], 1) for x in condition_1_result_tuples] 
        condition_2_result_tuples = [(x[0], x[1], x[2], 2) for x in condition_2_result_tuples] 
        condition_3_result_tuples = [(x[0], x[1], x[2], 3) for x in condition_3_result_tuples] 
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
                out.append((cumulative_reward, done, iter_idx, task_idx, condition))
                pass 
            pass 
        append_results(condition_0_result_tuples) 
        append_results(condition_1_result_tuples) 
        append_results(condition_2_result_tuples) 
        append_results(condition_3_result_tuples) 
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
    schema = ['score', 'done', 'iter', 'task', 'condition'] 
    z = y.toDF(schema=schema) 
    w = z.groupBy('iter', 'condition').mean('score') 
    df = w.toPandas()
    scores0 = df.loc[df['condition'] == 0].sort_values('iter')['avg(score)'].tolist() 
    scores1 = df.loc[df['condition'] == 1].sort_values('iter')['avg(score)'].tolist() 
    scores2 = df.loc[df['condition'] == 2].sort_values('iter')['avg(score)'].tolist() 
    scores3 = df.loc[df['condition'] == 3].sort_values('iter')['avg(score)'].tolist() 
    ### save data 
    FILENAME = 'df-experiment-15'
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
    pass 

if __name__ == '__main__': 
    phase_1() 
    phase_2() 
