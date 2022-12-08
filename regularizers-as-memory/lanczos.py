## Lanczos method sketch 
import numpy as np
from scipy.linalg import block_diag 
import torch 
import traceback 
from tqdm import tqdm 

def lanczos(AAT, r): 
    'Lanczos algorithm: produce AA^T = V T V^T' 
    vecs = [] 
    diags = [] 
    off_diags = [] 
    ## init 
    n = AAT.shape[0] 
    v = np.random.rand(n,1) 
    v = v / np.sqrt(np.matmul(np.transpose(v),v))
    next_v = np.matmul(AAT, v) 
    diag = np.matmul(np.transpose(next_v), v) 
    next_v = next_v - diag * v 
    vecs.append(v) 
    diags.append(diag) 
    for _ in range(r-1): 
        prev_v = v 
        off_diag = np.sqrt(np.matmul(np.transpose(next_v),next_v)) 
        v = next_v / off_diag  
        next_v = np.matmul(AAT, v) 
        diag = np.matmul(np.transpose(next_v), v) 
        next_v = next_v - diag * v - off_diag * prev_v 
        vecs.append(v) 
        diags.append(diag) 
        off_diags.append(off_diag) 
        pass 
    ## build it 
    V = np.concatenate(vecs, axis=1) 
    diags = np.array(diags).reshape([-1]) 
    off_diags = np.array(off_diags).reshape([-1]) 
    T = np.diag(diags) + np.diag(off_diags, -1) + np.diag(off_diags, 1) 
    VTVT = np.matmul(V, T)
    VTVT = np.matmul(VTVT, np.transpose(V)) 
    return VTVT

def l_lanczos(get_grad_generator, r, p, eps=0., device=None, mfi_alternate=None, disable_tqdm=True):
    '''
    limited-memory Lanczos algorithm
    inputs:
    - get_grad_generator: a function that returns a grad sampler. grads are N(0,Fisher Information) distributed.
    - r: Krylov space rank
    - p: dimension of (p X p) Fisher Information 
    - device: which device to execute on 
    - mfi_alternate: an alternative function to replace the below `multiply_fisher_infromation` 
    - disable_tqdm: if True, silence the progress bar 
    outputs:
    - A: a (p X r) matrix, providing low-rank Fisher Information approximation AA^T
    '''
    def multiply_fisher_information(x, disable_tqdm=disable_tqdm):
        grad_generator = get_grad_generator() 
        out = 0. 
        for g in grad_generator():
            #gTx = g.transpose(0,1).matmul(x) 
            #ggTx = g.matmul(gTx) 
            #out += ggTx 
            ## using one-liner to encourage garbage collection 
            g = g.reshape([-1, 1]) 
            out += g.matmul(g.transpose(0,1).matmul(x)) 
            if eps > 0.:
                out += eps * x 
            pass 
        return out  
    if mfi_alternate is not None: 
        multiply_fisher_information = mfi_alternate 
        pass 
    vecs = [] 
    diags = [] 
    off_diags = [] 
    ## init 
    v = torch.normal(0, torch.ones([p, 1])) 
    if device is not None: 
        v = v.to(device) 
    v = v / torch.sqrt(v.transpose(0,1).matmul(v)) 
    ## next_v = AAT.matmul(v)  
    next_v = multiply_fisher_information(v) 
    diag = next_v.transpose(0,1).matmul(v) 
    next_v = next_v - diag * v 
    vecs.append(v) 
    diags.append(diag) 
    pbar = tqdm(range(r-1), disable=disable_tqdm) 
    for _ in pbar: 
        prev_v = v 
        off_diag = torch.sqrt(next_v.transpose(0,1).matmul(next_v))
        v = next_v / off_diag  
        ## next_v = AAT.matmul(v)  
        next_v = multiply_fisher_information(v) 
        diag = next_v.transpose(0,1).matmul(v) 
        next_v = next_v - diag * v - off_diag * prev_v 
        vecs.append(v) 
        diags.append(diag) 
        off_diags.append(off_diag) 
        pass 
    ## build it 
    V = torch.cat(vecs, dim=1) 
    diags = torch.tensor(diags).reshape([-1])
    off_diags = torch.tensor(off_diags).reshape([-1]) 
    T = torch.diag(diags) + torch.diag(off_diags, -1) + torch.diag(off_diags, 1) 
    if device is not None: 
        T = T.to(device) 
    ## combine V & T into single matrix A 
    eigs = torch.linalg.eigh(T) 
    positive_eigenvalues = torch.relu(eigs.eigenvalues) ## for sqrt 
    sqrt_T = eigs.eigenvectors.matmul(torch.diag(torch.sqrt(positive_eigenvalues))).matmul(eigs.eigenvectors.transpose(0,1)) 
    A = V.matmul(sqrt_T) 
    return A 

def combine_krylov_spaces(A, B, device=None, krylov_eps=0.): 
    '''
    Uses a modified Lanczos algorithm to combine Krylov bases `A` and `B`. 
    inputs: 
    - A: a Krylov basis, perhaps produced by `l_lanczos`, so must be a rectangular tensor 
    - B: a Krylov basis, perhaps produced by `l_lanczos`, must be the same shape as `A` 
    - device: which device to execute on 
    outputs: 
    - C: a combined Krylov basis 
    '''
    def mfi_alternate(x): 
        x1 = A.matmul(A.transpose(0,1).matmul(x)) 
        x2 = B.matmul(B.transpose(0,1).matmul(x)) 
        if krylov_eps > 0.:
            return x1 + x2 + krylov_eps * x 
        return x1 + x2 
    p, r = tuple(A.shape) 
    C = l_lanczos(get_grad_generator=None, r=r, p=p, eps=krylov_eps, device=device, mfi_alternate=mfi_alternate)  
    return C * .5  

