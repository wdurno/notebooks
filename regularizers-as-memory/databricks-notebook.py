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

def f(model_idx):
    model_idx = int(model_idx) 
    ## install requirements
    import os
    os.system('pip install torch tqdm gym pygame')
    ## run experiment 
    from regmem import Model
    model = Model() 
    result_tuples = model.simulate(total_iters=10000, plot_prob_func=False, plot_rewards=False) 
    ## format output 
    out = []
    for r in result_tuples:
        reward = r[0]
        done = r[1]
        iter_idx = r[2]
        out.append((reward, done, iter_idx, model_idx))
    return out 

x = sc.parallelize(list(range(100)), 50) 
y = x.map(f).collect() 
y

# COMMAND ----------

import matplotlib.pyplot as plt 
import pandas as pd 

def process_data(data):
    cumulative_score = [] 
    iteration = [] 
    model_idx = [] 
    prev_score = 0. 
    for job_result in data:
        for row in job_result:
            if row[0] > 0.: 
                prev_score += row[0] 
                cumulative_score.append(prev_score) 
                iteration.append(row[2]) 
                model_idx.append(row[3]) 
            else: 
                prev_score = 0 
                pass
            pass
        pass 
    ## can be replaced with flatmap, toDF, and SparkSQL 
    df = pd.DataFrame({'score': cumulative_score, 'iter': iteration, 'model': model_idx}) 
    return df.groupby('iter').agg({'score': 'mean'}).score.tolist()

mean_scores = process_data(y)
plt.plot(mean_scores) 
plt.show()
