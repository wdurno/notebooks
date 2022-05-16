import torch 

STD = torch.tensor([[4.,1.,1.],[1,3,1],[1,1,2]]) 
vals, vecs = torch.eig(STD, eigenvectors=True) 
SQRT_STD = vecs.matmul(torch.diag(torch.sqrt(vals[:,0]))) 
DIM = 3

def sample():
    return SQRT_STD.matmul(torch.normal(0, torch.ones([DIM,1]))) 

def als(grads, eps=1e-3, max_iter=1000, rank=2):
    p = int(grads[0].shape[0])
    beta = torch.zeros(size=[p,rank])
    laps = 0
    col = 0
    for grad in grads:
        beta[:,col] = grad[:,0]
        col += 1
        if col == rank:
            col = 0
            laps += 1
            pass
        pass
    beta = beta / float(laps)
    ## iterate to convergence
    continue_iterating = True
    total_iterations = 0
    while continue_iterating:
        ## iteration init tasks
        total_iterations += 1
        prev_beta = beta
        ## calculate new beta
        betaTbeta_inv = beta.transpose(0,1).matmul(beta).inverse() ## should be rank X rank matrix, so small and fast
        sum_betaT_del_delT = 0
        for grad in grads:
            sum_betaT_del_delT += beta.transpose(0,1).matmul(grad).matmul(grad.transpose(0,1))
            pass
        sum_betaT_del_delT /= len(grads) ## not required in actual code 
        beta = betaTbeta_inv.matmul(sum_betaT_del_delT).transpose(0,1)*.5 + beta*.5 ## avoid thrashing 
        ## check for convergence
        if (beta - prev_beta).abs().sum() < eps:
            continue_iterating = False
        ## check for iteration limit
        if total_iterations > max_iter:
            continue_iterating = False
            pass
        pass
    return beta.matmul(beta.transpose(0,1))  

def control(grads):
    out = 0. 
    for grad in grads:
        out += grad.matmul(grad.transpose(0,1)) 
    out /= len(grads) 
    return out 

samples50 = [sample() for _ in range(50)] 
samples100 = [sample() for _ in range(100)] 
samples1000 = [sample() for _ in range(1000)] 

print( (STD - als(samples50)).abs().sum() ) 
print( (STD - als(samples100)).abs().sum() ) 
print( (STD - als(samples1000)).abs().sum() ) 
print( (STD - control(samples1000)).abs().sum() ) 

