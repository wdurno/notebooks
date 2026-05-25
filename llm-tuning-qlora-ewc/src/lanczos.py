import numpy as np
import torch 
from tqdm.auto import tqdm 

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

def l_lanczos(
    get_grad_generator,
    r,
    p,
    device=None,
    mfi_alternate=None,
    diag_alternate=None,
    disable_tqdm=True,
    calc_diag=False,
    min_off_diag=1e-12,
    normalize=True,
):
    '''
    limited-memory Lanczos algorithm
    inputs:
    - get_grad_generator: a function that returns a grad sampler. grads are N(0,Fisher Information) distributed.
    - r: Krylov space rank
    - p: dimension of (p X p) Fisher Information 
    - device: which device to execute on 
    - mfi_alternate: an alternative function to replace the below `multiply_fisher_information` 
    - diag_alternate: an alternative function returning diag(Fisher Information)
    - disable_tqdm: if True, silence the progress bar 
    - calc_diag: if True, calculate residual diagonal covariance vector 
    - min_off_diag: stop early if the Krylov residual norm falls below this value
    - normalize: if True, average Fisher-vector products and diagonal estimates by sample count
    outputs:
    - A: a (p X r) matrix, providing low-rank Fisher Information approximation AA^T
    '''
    if r < 1: 
        ## degenerate case 
        out = torch.zeros([p, 0])
        if device is not None:
            out = out.to(device)
        if calc_diag:
            diagonal = torch.zeros([p, 1], device=out.device)
            return out, diagonal
        return out

    def multiply_fisher_information(x, disable_tqdm=disable_tqdm):
        grad_generator = get_grad_generator()
        out = torch.zeros_like(x)
        n = 0
        # for g in grad_generator(): 
        for g in tqdm(grad_generator(), disable=disable_tqdm):
            #gTx = g.transpose(0,1).matmul(x) 
            #ggTx = g.matmul(gTx) 
            #out += ggTx 
            ## using one-liner to encourage garbage collection 
            g = g.to(x.device).reshape([-1, 1]) 
            out += g.matmul(g.transpose(0,1).matmul(x)) 
            n += 1
        if normalize and n > 0:
            out = out / n
        return out  

    if mfi_alternate is not None: 
        multiply_fisher_information = mfi_alternate 

    vecs = [] 
    diags = [] 
    off_diags = [] 
    ## init 
    if device is None:
        device = torch.device("cpu")
    v = torch.randn([p, 1], device=device)
    v = v / torch.sqrt(v.transpose(0,1).matmul(v)) 
    ## next_v = AAT.matmul(v)  
    next_v = multiply_fisher_information(v) 
    diag = next_v.transpose(0,1).matmul(v) 
    next_v = next_v - diag * v 
    vecs.append(v) ## wiki says to add this vector, even before FI multiplication  
    diags.append(diag) 
    pbar = tqdm(range(r-1), disable=disable_tqdm) 
    for _ in pbar: 
        prev_v = v 
        off_diag = torch.sqrt(next_v.transpose(0,1).matmul(next_v))
        if off_diag.item() <= min_off_diag:
            break
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
    diags = torch.cat([d.reshape([1]) for d in diags]).to(device)
    T = torch.diag(diags)
    if off_diags:
        off_diags = torch.cat([d.reshape([1]) for d in off_diags]).to(device)
        T = T + torch.diag(off_diags, -1) + torch.diag(off_diags, 1)
    ## combine V & T into single matrix A 
    eigs = torch.linalg.eigh(T) 
    positive_eigenvalues = torch.relu(eigs.eigenvalues) ## for sqrt 
    sqrt_T = eigs.eigenvectors.matmul(torch.diag(torch.sqrt(positive_eigenvalues))).matmul(eigs.eigenvectors.transpose(0,1)) 
    A = V.matmul(sqrt_T) 
    if not calc_diag: 
        return A 
    ## calc diagonal_residual = diag(Fisher Information - AA^T) 
    if diag_alternate is None:
        grad_generator = get_grad_generator()
        diagonal_residual = torch.zeros([p, 1], device=device)
        n = 0
        for g in grad_generator(): 
            g = g.to(device).reshape([-1,1]) 
            diagonal_residual += g*g ## sums to N*diag(Fisher Information) 
            n += 1
        if normalize and n > 0:
            diagonal_residual = diagonal_residual / n
    else:
        diagonal_residual = diag_alternate().to(device)
    diagonal_residual -= (A*A).sum(dim=1).reshape([-1,1]) 
    diagonal_residual[diagonal_residual < 0.] = 0. ## handle tiny numerical errors 
    return A, diagonal_residual.reshape([-1,1])  

def combine_krylov_spaces(A, B, device=None): 
    '''
    Uses a modified Lanczos algorithm to combine Krylov bases `A` and `B`. 
    inputs: 
    - A: a Krylov basis, perhaps produced by `l_lanczos`, so must be a rectangular tensor 
    - B: a Krylov basis, perhaps produced by `l_lanczos`, must be the same shape as `A` 
    - device: which device to execute on 
    outputs: 
    - C: a combined Krylov basis 
    '''
    if device is None:
        device = A.device

    def mfi_alternate(x): 
        x1 = A.matmul(A.transpose(0,1).matmul(x)) 
        x2 = B.matmul(B.transpose(0,1).matmul(x)) 
        return x1 + x2 
    p, r = tuple(A.shape) 
    C = l_lanczos(get_grad_generator=None, r=r, p=p, device=device, mfi_alternate=mfi_alternate)  
    return C   
