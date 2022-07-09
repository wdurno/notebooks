from regmem_cnn import Model

m10k = Model() 
m10k_raw_results = m10k.simulate(total_iters=10000) 

m20k_krylov = m10k.copy() 
m20k_krylov.convert_observations_to_memory(krylov_rank=10) 
m20k_krylov_raw_results = m20k_krylov.simulate(total_iters=10000) 

m20k_eigen = m10k.copy() 
m20k_eigen.convert_observations_to_memory(n_eigenvectors=10) 
m20k_eigen_raw_results = m20k_eigen.simulate(total_iters=10000) 

