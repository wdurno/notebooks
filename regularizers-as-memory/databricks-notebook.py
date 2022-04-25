# Databricks notebook source
# MAGIC %md
# MAGIC # Azure Databricks notebook
# MAGIC 
# MAGIC I'm using Databricks to avoid architecting and containerization work. 
# MAGIC Default environments seem to have most dependencies, but need init step `pip install gym pygame`. 
# MAGIC 
# MAGIC I'm using a scaled compute environment because my experiments have _noisy_ results and non-trivial compute time per experiment.
# MAGIC By executing at scale, I can average-out noise and derive statistically significant findings.
