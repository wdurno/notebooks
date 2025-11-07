## Lanczos method sketch 
import numpy as np
from scipy.linalg import block_diag 
import torch 
import traceback 
#from tqdm import tqdm 
from tqdm.notebook import tqdm 

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

def l_lanczos(get_grad_generator, r, p, eps=0., device=None, mfi_alternate=None, disable_tqdm=True, calc_diag=False):
    '''
    limited-memory Lanczos algorithm
    inputs:
    - get_grad_generator: a function that returns a grad sampler. grads are N(0,Fisher Information) distributed.
    - r: Krylov space rank
    - p: dimension of (p X p) Fisher Information 
    - device: which device to execute on 
    - mfi_alternate: an alternative function to replace the below `multiply_fisher_infromation` 
    - disable_tqdm: if True, silence the progress bar 
    - calc_diag: if True, calculate residual diagonal covariance vector 
    outputs:
    - A: a (p X r) matrix, providing low-rank Fisher Information approximation AA^T
    '''
    if r < 1: 
        ## degenerate case 
        return torch.zeros([p, 1]) 
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
    vecs.append(v) ## wiki says to add this vector, even before FI multiplication  
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
    if not calc_diag: 
        return A 
    ## calc diagonal_residual = diag(Fisher Information - AA^T) 
    grad_generator = get_grad_generator() 
    diagonal_residual = 0. 
    for g in grad_generator(): 
        g = g.reshape([-1,1]) 
        diagonal_residual += g*g ## sums to N*diag(Fisher Information) 
        pass 
    diagonal_residual -= (A*A).sum(dim=1).reshape([-1,1]) 
    diagonal_residual[diagonal_residual < 0.] = 0. ## handle tiny numerical errors 
    return A, diagonal_residual.reshape([-1,1])  

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
    return C   

def get_get_amari_chentsov_product(get_grad_generator, delta, positive_part=True): 
    def get_amari_chentsov_product(x): 
        grad_generator = get_grad_generator() 
        out = 0. 
        for g in grad_generator(): 
            g = g.reshape([-1, 1]) 
            c = g.transpose(0,1).matmul(delta) ## scalar 
            if (positive_part and c > 0.) or (not positive_part and c < 0.): 
                out += c * g.matmul(g.transpose(0,1).matmul(x)) ## matrix 
        return out 
    return get_amari_chentsov_product 

# def combine_psd(A, B): ## TODO what if BB^T negative definite 
#     'Returns PSD CC^T s.t. C = argmin_C \| CC^T - AA^T - BB^T \|_F^2' 
#     ## TODO this is sub-optimal poc-grade code 
#     r = A.shape[1] 
#     Y = torch.concatenate([A,B], dim=1) 
#     G = Y.transpose(0,1).matmul(Y) 
#     eigs = torch.linalg.eigh(G) 
#     r_idx = _get_top_r_positive_indices(eigs.eigenvalue, r) 
#     S_r = torch.diag(eigs.eigenvalues[r_idx].pow(-.5)) 
#     V_r = eigs.eigenvectors[:,r_idx] 
#     return Y.matmul(V_r.matmul(S_r)) 

# def _get_top_r_positive_indices(x, r): 
#     pos = torch.nonzero(x > 0, as_tuple=True)[0] 
#     _, top_idx = torch.topk(x[pos], k=min(r, pos.numel())) 
#     indices = pos[top_idx] 
#     return indices 

## yes, this is AI-generated code. Gotta turn-and-burn hypotheses for science 
def combine_psd(A, B, signB=+1, r=None, tol=None, pad_zeros=False):
    """
    Returns C such that C C^T is the Frobenius-nearest PSD to A A^T + signB * B B^T,
    truncated to rank r (default: r = A.shape[1]). Works for signB = +1 (add PSD)
    or signB = -1 (subtract NSD).

    Shapes: A, B are (n, rA), (n, rB), thin. Output C is (n, r_out).
    """
    assert signB in (+1, -1), "signB must be +1 or -1"
    n = A.shape[0]
    if r is None:
        r = A.shape[1]

    # 1) Build tall-skinny basis for span([A, B]) and get small R
    Y = torch.cat([A, B], dim=1)                       # (n, m), m = rA + rB
    Q, R = torch.linalg.qr(Y, mode='reduced')          # Q: (n,k), R: (k,m), k <= m

    # 2) Form signed small core K = R * diag(I, signB*I) * R^T
    mA = A.shape[1]
    R_A = R[:, :mA]                                    # (k, rA)
    R_B = R[:, mA:]                                    # (k, rB)
    if signB == +1:
        K = R_A @ R_A.T + R_B @ R_B.T
    else:  # signB == -1  => subtract NSD block
        K = R_A @ R_A.T - R_B @ R_B.T

    # Numerical symmetrization (cheap & keeps eigh happy)
    K = 0.5 * (K + K.T)

    # 3) Eigen-decompose small core, keep only positive spectrum
    evals, evecs = torch.linalg.eigh(K)                # ascending order
    if tol is None:
        # scale-aware tolerance
        tol = 1e-10 * torch.trace(torch.abs(K)) / max(1, K.shape[0])

    pos = evals > tol
    if pos.any():
        evals_pos = evals[pos]
        evecs_pos = evecs[:, pos]
        # sort positives descending
        idx = torch.argsort(evals_pos, descending=True)
        evals_pos = evals_pos[idx]
        evecs_pos = evecs_pos[:, idx]
        r_out = min(r, evals_pos.numel())
        evals_sel = evals_pos[:r_out]
        evecs_sel = evecs_pos[:, :r_out]
        # 4) Lift back: C = Q U_+ Λ_+^{1/2}
        C = Q @ (evecs_sel * torch.sqrt(evals_sel).unsqueeze(0))
        if pad_zeros and r_out < r:
            # Pad with zero-columns to keep a fixed (n, r) shape if you prefer
            pad = torch.zeros((n, r - r_out), dtype=C.dtype, device=C.device)
            C = torch.cat([C, pad], dim=1)
        return C
    else:
        # No positive eigenvalues -> projection to PSD cone is zero
        return torch.zeros((n, r), dtype=Y.dtype, device=Y.device) if pad_zeros else torch.zeros((n, 0), dtype=Y.dtype, device=Y.device)
