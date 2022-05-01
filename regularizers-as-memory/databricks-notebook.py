# Databricks notebook source
# MAGIC %md
# MAGIC # Azure Databricks notebook
# MAGIC 
# MAGIC I'm using Databricks to avoid architecting and containerization work. 
# MAGIC Default environments seem to have most dependencies, but need init step `pip install gym pygame`. 
# MAGIC Used Spark config:
# MAGIC ```
# MAGIC spark.task.cpus 2
# MAGIC spark.executor.pyspark.memory 12g
# MAGIC ```
# MAGIC 
# MAGIC I'm using a scaled compute environment because my experiments have _noisy_ results and non-trivial compute time per experiment.
# MAGIC By executing at scale, I can average-out noise and derive statistically significant findings.

# COMMAND ----------

## pyspark  
import pandas as pd 

SAMPLE_SIZE = 1000 
ITERS=1000

def f(task_idx):
    import traceback 
    try:
        task_idx = int(task_idx) 
        ## install requirements 
        import os 
        os.system('pip install torch tqdm gym pygame') ## does %pip work on auto-scaled nodes? 
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
        condition_5_model = condition_0_model.copy() 
        ## continue condition 0 trial, without application of memory and without discarding data 
        condition_0_result_tuples_after = condition_0_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        ## condition 1 (control): No use of memory, do discard data 
        condition_1_model.clear_observations() 
        condition_1_result_tuples_after = condition_1_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        ## condition 2 (experimental): Use memory, do discard data 
        condition_2_model.convert_observations_to_memory() 
        condition_2_result_tuples_after = condition_2_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        ## condition 3 (experimental): Use memory, do discard data, use 1 eigenvector 
        condition_3_model.convert_observations_to_memory(n_eigenvectors=1) 
        condition_3_result_tuples_after = condition_3_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        ## condition 4 (experimental): Use memory, do discard data, use 2 eigenvectors 
        condition_4_model.convert_observations_to_memory(n_eigenvectors=2) 
        condition_4_result_tuples_after = condition_4_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        ## condition 5 (experimental): Use memory, do discard data, use 10 eigenvectors 
        condition_5_model.convert_observations_to_memory(n_eigenvectors=10) 
        condition_5_result_tuples_after = condition_5_model.simulate(total_iters=ITERS, plot_prob_func=False, plot_rewards=False) 
        ## merge before & after results 
        condition_0_result_tuples = condition_0_result_tuples_before + condition_0_result_tuples_after 
        condition_1_result_tuples = condition_0_result_tuples_before + condition_1_result_tuples_after 
        condition_2_result_tuples = condition_0_result_tuples_before + condition_2_result_tuples_after 
        condition_3_result_tuples = condition_0_result_tuples_before + condition_3_result_tuples_after 
        condition_4_result_tuples = condition_0_result_tuples_before + condition_4_result_tuples_after 
        condition_5_result_tuples = condition_0_result_tuples_before + condition_5_result_tuples_after 
        ## append condition codes 
        condition_0_result_tuples = [(x[0], x[1], x[2], 0) for x in condition_0_result_tuples] 
        condition_1_result_tuples = [(x[0], x[1], x[2], 1) for x in condition_1_result_tuples] 
        condition_2_result_tuples = [(x[0], x[1], x[2], 2) for x in condition_2_result_tuples] 
        condition_3_result_tuples = [(x[0], x[1], x[2], 3) for x in condition_3_result_tuples] 
        condition_4_result_tuples = [(x[0], x[1], x[2], 4) for x in condition_4_result_tuples] 
        condition_5_result_tuples = [(x[0], x[1], x[2], 5) for x in condition_5_result_tuples] 
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
        append_results(condition_5_result_tuples) 
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
scores5 = df.loc[df['condition'] == 5].sort_values('iter')['avg(score)'].tolist() 

# COMMAND ----------

## analysis 

import matplotlib.pyplot as plt 

plt.plot(scores0, label='0') 
plt.plot(scores1, label='1') 
plt.plot(scores2, label='2') 
plt.plot(scores3, label='3') 
plt.plot(scores4, label='4') 
plt.plot(scores5, label='5') 
plt.legend()
plt.show()

# COMMAND ----------

## save data 
FILENAME = 'df-4.30.22-2'

storage_name = 'databricksdataa'
sas_key = '[REDACTED]' ## TODO CYCLE THIS KEY 
output_container_name = 'data'
output_container_path = ''

# Configure blob storage account access key globally
spark.conf.set(
  "fs.azure.account.key.%s.blob.core.windows.net" % storage_name,
  sas_key)

output_container_path = "wasbs://%s@%s.blob.core.windows.net" % (output_container_name, storage_name) 
output_blob_folder = "%s/" % output_container_path 

df_to_save = pd.DataFrame({'scores0': scores0, 
                           'scores1': scores1, 
                           'scores2': scores2,
                           'scores3': scores3,
                           'scores4': scores4,
                           'scores5': scores5,})
dbutils.fs.put(f'{output_blob_folder}/{FILENAME}.csv', df_to_save.to_csv(), overwrite=True) ## got annoyed with mounts for blob-writing parquets 
