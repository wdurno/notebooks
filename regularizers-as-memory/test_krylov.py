import torch 
from lanczos import l_lanczos 
p = 10 
r = 4 
n = 10000 

A = torch.normal(0, torch.ones([p, r])) 
AAT = A.matmul(A.transpose(0, 1)) 
AAT_eigs = torch.linalg.eig(AAT) 
AAT_eigenvectors = AAT_eigs.eigenvectors.real[:, :r] 
AAT_eigenvalues = AAT_eigs.eigenvalues.real[:r] 
sqrt_AAT = AAT_eigenvectors.matmul(torch.diag(torch.sqrt(AAT_eigenvalues))).matmul(AAT_eigenvectors.transpose(0, 1))  

sample = sqrt_AAT.matmul(torch.normal(0, torch.ones([p, n]))) 

def get_grad_generator(): 
    def grad_generator():
        for i in range(n): 
            yield sample[:,i] 
    return grad_generator 

for rr in range(1, r+2): 
    approx_A = l_lanczos(get_grad_generator, rr, p) 
    approx_AAT = approx_A.matmul(approx_A.transpose(0, 1))/n  
    err = (AAT - approx_AAT).abs().sum() 
    print(f'err: {err}') 

