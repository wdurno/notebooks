# Databricks notebook source
# MAGIC %md
# MAGIC # Azure Databricks notebook
# MAGIC 
# MAGIC I'm using Databricks to avoid architecting and containerization work. 
# MAGIC Default environments seem to have most dependencies, but need init step `pip install gym pygame`. 
# MAGIC Used Spark config "spark.executor.pyspark.memory 12g".
# MAGIC 
# MAGIC I'm using a scaled compute environment because my experiments have _noisy_ results and non-trivial compute time per experiment.
# MAGIC By executing at scale, I can average-out noise and derive statistically significant findings.

# COMMAND ----------

def f(job_idx):
    ## install requirements
    import os
    os.system('pip install torch tqdm gym pygame')
    ## run experiment 
    from regmem import Model
    model = Model() 
    return model.simulate(total_iters=10000, plot_prob_func=False, plot_rewards=False) 

x = sc.parallelize(list(range(10)), 5) 
y = x.map(f).collect()
y
