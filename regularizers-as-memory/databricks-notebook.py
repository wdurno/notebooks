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

def f(model_idx):
    model_idx = int(model_idx) 
    ## install requirements
    import os
    os.system('pip install torch tqdm gym pygame') ## does %pip work on auto-scaled nodes? 
    ## run experiment 
    from regmem import Model
    model = Model() 
    result_tuples = model.simulate(total_iters=10000, plot_prob_func=False, plot_rewards=False) 
    ## format output 
    out = []
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
        out.append((cumulative_reward, done, iter_idx, model_idx))
    return out 

x = sc.parallelize(list(range(100)), 50) 
y = x.flatMap(f) 
schema = ['score', 'done', 'iter', 'model'] 
z = y.toDF(schema=schema) 
w = z.groupBy('iter').mean('score')
df = w.toPandas()
scores = df.sort_values('iter')['avg(score)'].tolist()

# COMMAND ----------

## analysis 

import matplotlib.pyplot as plt 
import pandas as pd 

plt.plot(scores) 
plt.show()
