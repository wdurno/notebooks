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

def f(task_idx):
    task_idx = int(task_idx) 
    ## install requirements
    import os
    os.system('pip install torch tqdm gym pygame') ## does %pip work on auto-scaled nodes? 
    ## run experiment 
    from regmem import Model
    condition_0_model = Model() 
    ## condition 0 (control): No use of memory, no discarding of data 
    condition_0_result_tuples_before = condition_0_model.simulate(total_iters=10000, plot_prob_func=False, plot_rewards=False) 
    ## copy, creating other models before continuing 
    condition_1_model = condition_0_model.copy() 
    condition_2_model = condition_0_model.copy() 
    ## continue condition 0 trial, without application of memory and without discarding data 
    condition_0_result_tuples_after = condition_0_model.simulate(total_iters=10000, plot_prob_func=False, plot_rewards=False) 
    ## condition 1 (control): No use of memory, do discard data 
    condition_1_model.clear_observations() 
    condition_1_result_tuples_after = condition_1_model.simulate(total_iters=10000, plot_prob_func=False, plot_rewards=False) 
    ## condition 2 (experimental): Use memory, do discard data 
    condition_2_model.convert_observations_to_memory() 
    condition_2_result_tuples_after = condition_1_model.simulate(total_iters=10000, plot_prob_func=False, plot_rewards=False) 
    ## merge before & after results 
    condition_0_result_tuples = condition_0_result_tuples_before + condition_0_result_tuples_after 
    condition_1_result_tuples = condition_0_result_tuples_before + condition_1_result_tuples_after 
    condition_2_result_tuples = condition_0_result_tuples_before + condition_2_result_tuples_after 
    ## append condition codes 
    condition_0_result_tuples = [(x[0], x[1], x[2], 0) for x in condition_0_result_tuples] 
    condition_1_result_tuples = [(x[0], x[1], x[2], 1) for x in condition_1_result_tuples] 
    condition_2_result_tuples = [(x[0], x[1], x[2], 2) for x in condition_2_result_tuples] 
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
    return out 

x = sc.parallelize(list(range(100)), 50) 
y = x.flatMap(f) 
schema = ['score', 'done', 'iter', 'task', 'condition'] 
z = y.toDF(schema=schema) 
w = z.groupBy('iter', 'condition').mean('score') 
df = w.toPandas()
scores = df.sort_values('iter')['avg(score)'].tolist()

# COMMAND ----------

## analysis 

import matplotlib.pyplot as plt 
import pandas as pd 

plt.plot(scores) 
plt.show()
