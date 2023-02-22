## Experiment 6: Optimal lambda  
## Observe effect of using mathematically-determined lambda 

import pandas as pd
from az_blob_util import upload_to_blob_store
import os 
from pyspark import SparkContext
from pyspark.sql import SparkSession 
sc = SparkContext() 
spark = SparkSession(sc) 

SAMPLE_SIZE = 1000  
ITERS = 1000

def f(task_idx):
    import traceback 
    try:
        task_idx = int(task_idx) 
        ## run experiment 
        from regmem import Model 
        condition_0_model = Model() 
        ## condition 0 (control): No use of memory, no discarding of data 
        condition_0_result_tuples_before = condition_0_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        ## copy, creating other models before continuing 
        condition_1_model = condition_0_model.copy() 
        condition_2_model = condition_0_model.copy() 
        condition_3_model = condition_0_model.copy() 
        condition_4_model = condition_0_model.copy() 
        ## continue condition 0 (control), without application of memory and without discarding data 
        condition_0_result_tuples_after = condition_0_model.simulate(total_iters=2*ITERS, plot_prob_func=False, plot_rewards=False) 
        ## condition 1 (control): Use memory, clear data, non-optimal lambda   
        condition_1_model.convert_observations_to_memory() 
        condition_1_result_tuples_after = condition_1_model.simulate(total_iters=2*ITERS, plot_prob_func=False, plot_rewards=False) 
        ## condition 2 (experimental): Use memory, clear data, optimal lambda 
        condition_2_model.regularizing_lambda_function = lambda model: (model.hessian_denominator/(model.total_iters - model.hessian_denominator))
        condition_2_model.convert_observations_to_memory() 
        condition_2_result_tuples_after = condition_2_model.simulate(total_iters=2*ITERS, plot_prob_func=False, plot_rewards=False) 
        ## condition 3 (experimental): Use memory, clear data, custom lambda 
        condition_3_model.regularizing_lambda_function = lambda model: (model.hessian_denominator/model.total_iters)
        condition_3_model.convert_observations_to_memory() 
        condition_3_result_tuples_after = condition_3_model.simulate(total_iters=2*ITERS, plot_prob_func=False, plot_rewards=False)
        ## condition 4 (experimental): Use memory, clear data, near-optimal lambda = min(1, n1/n2)
        condition_4_model.regularizing_lambda_function = lambda model: min(1., float(model.hessian_denominator/(model.total_iters - model.hessian_denominator))) 
        condition_4_model.convert_observations_to_memory()
        condition_4_result_tuples_after = condition_4_model.simulate(total_iters=2*ITERS, plot_prob_func=False, plot_rewards=False) 
        ## merge before & after results 
        condition_0_result_tuples = condition_0_result_tuples_before + condition_0_result_tuples_after 
        condition_1_result_tuples = condition_0_result_tuples_before + condition_1_result_tuples_after 
        condition_2_result_tuples = condition_0_result_tuples_before + condition_2_result_tuples_after
        condition_3_result_tuples = condition_0_result_tuples_before + condition_3_result_tuples_after 
        condition_4_result_tuples = condition_0_result_tuples_before + condition_4_result_tuples_after 
        ## append condition codes 
        condition_0_result_tuples = [(x[0], x[1], x[2], 0) for x in condition_0_result_tuples] 
        condition_1_result_tuples = [(x[0], x[1], x[2], 1) for x in condition_1_result_tuples] 
        condition_2_result_tuples = [(x[0], x[1], x[2], 2) for x in condition_2_result_tuples] 
        condition_3_result_tuples = [(x[0], x[1], x[2], 3) for x in condition_3_result_tuples] 
        condition_4_result_tuples = [(x[0], x[1], x[2], 4) for x in condition_4_result_tuples] 
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
        append_results(condition_4_result_tuples) 
    except Exception as e: 
        ## increase verbosity before failing 
        print(f'ERROR!\n{e}\n{traceback.format_exc()}')
        raise e 
    return out 

x = sc.parallelize(list(range(SAMPLE_SIZE)), SAMPLE_SIZE) 
y = x.flatMap(f) 
schema = ['score', 'done', 'iter', 'task', 'condition'] 
z = y.toDF(schema=schema) 
w = z.groupBy('iter', 'condition').mean('score') 
df = w.toPandas()
scores0 = df.loc[df['condition'] == 0].sort_values('iter')['avg(score)'].tolist() 
scores1 = df.loc[df['condition'] == 1].sort_values('iter')['avg(score)'].tolist()
scores2 = df.loc[df['condition'] == 2].sort_values('iter')['avg(score)'].tolist() 
scores3 = df.loc[df['condition'] == 3].sort_values('iter')['avg(score)'].tolist() 
scores4 = df.loc[df['condition'] == 4].sort_values('iter')['avg(score)'].tolist() 

### save data 
FILENAME = 'df-experiment-6.csv'
storage_name = 'databricksdataa'
sas_key = os.environ['STORAGE_KEY']  
output_container_name = 'data'

# Configure blob storage account access key globally

df_to_save = pd.DataFrame({'scores0': scores0, 
                           'scores1': scores1,
                           'scores2': scores2,
                           'scores3': scores3, 
                           'scores4': scores4})
df_data = df_to_save.to_csv().encode() 
upload_to_blob_store(df_data, FILENAME, sas_key, output_container_name) 

