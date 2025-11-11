## Lanczos method sketch 
import numpy as np
import torch 
import traceback 
from typing import Callable, Optional, Iterable 
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

## AI-generated code follows ...yes, I reviewed it 
def _orthonormalize_columns(M: torch.Tensor) -> torch.Tensor:
    """
    Returns an orthonormal basis spanning the columns of M via skinny QR.
    Shape: (p, r_in) -> (p, r_out), with r_out = rank(M).
    """
    # Handle dtype/device transparently
    Q, _ = torch.linalg.qr(M, mode='reduced')  # (p, r_out)
    return Q

def _rotate_tensor3(T: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Transport a symmetric order-3 tensor T under a change of basis:
      T' = T ×1 R^T ×2 R^T ×3 R^T
    Shapes:
      T: (k_old, k_old, k_old)
      R: (k_old, k_new)   where columns live in the new basis
    Returns:
      T': (k_new, k_new, k_new)
    """
    # einsum indices: pa,qb,rc,pqr -> abc
    return torch.einsum('pa,qb,rc,pqr->abc', R, R, R, T)

def _project_scores_batch(U: torch.Tensor, g_batch: torch.Tensor) -> torch.Tensor:
    """
    Project a batch of scores to the reduced basis.
    Inputs:
      U: (p, r)  orthonormal basis
      g_batch: one of
         - (p,) or (p,1) single score
         - (b, p) or (b, p, 1) batch of b scores
    Returns:
      Y: (b, r) batch in reduced coords (b=1 if single)
    """
    if g_batch.ndim == 1:              # (p,)
        g_batch = g_batch.unsqueeze(0) # -> (1, p)
    elif g_batch.ndim == 2 and g_batch.shape[1] == 1:  # (p,1)
        g_batch = g_batch.squeeze(1).unsqueeze(0)      # -> (1, p)
    elif g_batch.ndim == 3 and g_batch.shape[-1] == 1: # (b, p, 1)
        g_batch = g_batch.squeeze(-1)                  # -> (b, p)
    # Now g_batch is (b, p)
    # y = U^T g for each sample => (b, r)
    return g_batch @ U

def update_amari_chentsov_tensor(
    get_grad_generator: Callable[[], Iterable[torch.Tensor]],
    information_basis: torch.Tensor,
    prior_tensor: Optional[torch.Tensor] = None,
    prior_information_basis: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    '''
    Updates (sums into) a small-gram estimate of the Amari–Chentsov tensor
    of size [r x r x r] in the information subspace.

    Inputs:
    - get_grad_generator: function returning an iterator of score tensors g.
      Each yielded item may be shaped (p,), (p,1), (b,p), or (b,p,1).
    - information_basis: approx top-r Fisher directions, shape (p, r0). Need not
      be normalized; we orthonormalize to U (p, r).
    - prior_tensor: optional previous small-gram tensor in prior basis, shape (r_prev, r_prev, r_prev).
    - prior_information_basis: optional prior basis (p, r_prev). If provided with prior_tensor,
      we rotate prior_tensor into the new basis before updating.

    Output:
    - updated_tensor: (r, r, r) accumulated (unnormalized) third-moment in the new basis.

    Notes:
    - This function **sums** y⊗y⊗y; divide by total samples externally for an average,
      or apply an EMA if desired.
    - Everything is O(p r^2) per batch for the projection, and O(r^3) for the tensor update;
      no n×n objects are formed.
    '''
    # 1) Orthonormalize the current information basis
    U = _orthonormalize_columns(information_basis)           # (p, r)
    p, r = U.shape
    device = U.device
    dtype = U.dtype

    # 2) Initialize / transport prior tensor if provided
    if prior_tensor is not None:
        if prior_information_basis is None:
            raise ValueError("prior_information_basis must be provided when prior_tensor is not None.")
        U_prev = _orthonormalize_columns(prior_information_basis)  # (p, r_prev)
        # Change-of-coordinates R = U_prev^T U (maps new coords from old)
        R = U_prev.transpose(0,1) @ U                              # (r_prev, r)
        T_init = _rotate_tensor3(prior_tensor.to(device=device, dtype=dtype), R)  # (r, r, r)
    else:
        T_init = torch.zeros((r, r, r), device=device, dtype=dtype)

    T = T_init

    # 3) Stream scores and accumulate y⊗y⊗y in the reduced space
    grad_generator = get_grad_generator()
    for g_batch in grad_generator():
        # Ensure device/dtype consistency without copying more than needed
        g_batch = g_batch.to(device=device, dtype=dtype)
        Y = _project_scores_batch(U, g_batch)             # (b, r)
        # Accumulate sum_b y_b ⊗ y_b ⊗ y_b
        T = T + torch.einsum('bi, bj, bk -> ijk', Y, Y, Y)

    # (Optional) enforce exact symmetry numerically (usually already symmetric) 
    # T = (T + T.transpose(0,1) + T.transpose(0,2) + T.permute(1,2,0) + T.permute(2,0,1) + T.permute(1,0,2)) / 6.0 

    return T

def combine_psd_plus_sym_core(A, U, Mk, r=None, tol=None, pad_zeros=True):
    """
    Return C such that C C^T is the Frobenius-nearest rank-r PSD approximation to:
        AA^T + U Mk U^T
    where A is (n x rA) thin, U is (n x k) with k << n, Mk is (k x k) symmetric.
    No n x n matrices are formed.

    Padding zeros defaults to True because information transport does tend to _destroy_ some information, with smallest eigenvalues getting zeroed.
    By [Karakida, Akaho, Amari 2019], the eigenvalue distribution has a sharp drop-off, with essentially one very largest eigenvalue. 
    So, information loss due to transport should be quite small. 

    If fewer than r positive eigenvalues exist, returns that many columns (or pads zeros).
    """
    n = A.shape[0]
    if r is None:
        r = A.shape[1]

    # 1) Build tall-skinny basis and small R via QR on [A, U]
    Y = torch.cat([A, U], dim=1)                    # (n, rA + k)
    Q, R = torch.linalg.qr(Y, mode='reduced')       # Q: (n, t), R: (t, rA+k), t <= rA+k

    rA = A.shape[1]
    R_A = R[:, :rA]                                 # (t, rA)
    R_U = R[:, rA:]                                 # (t, k)

    # 2) Small signed core: K = R_A R_A^T + R_U Mk R_U^T 
    K = R_A @ R_A.T + R_U @ Mk @ R_U.T              # (t, t)
    K = 0.5 * (K + K.T)                             # symmetrize numerically

    # 3) Eigendecompose, keep positive spectrum, truncate to rank r
    evals, evecs = torch.linalg.eigh(K)             # ascending
    if tol is None:
        tol = 1e-10 * torch.trace(torch.abs(K)) / max(1, K.shape[0])

    pos = evals > tol
    if not torch.any(pos):
        # No positive eigenvalues => PSD projection is zero
        if pad_zeros:
            return torch.zeros((n, r), dtype=Y.dtype, device=Y.device)
        return torch.zeros((n, 0), dtype=Y.dtype, device=Y.device)

    evals_pos = evals[pos]
    evecs_pos = evecs[:, pos]
    idx = torch.argsort(evals_pos, descending=True)
    evals_pos = evals_pos[idx]
    evecs_pos = evecs_pos[:, idx]

    r_out = min(r, evals_pos.numel())
    evals_sel = evals_pos[:r_out]
    evecs_sel = evecs_pos[:, :r_out]

    # 4) Lift: C = Q U_+ Λ_+^{1/2}
    C = Q @ (evecs_sel * torch.sqrt(evals_sel).unsqueeze(0))

    if pad_zeros and r_out < r:
        pad = torch.zeros((n, r - r_out), dtype=C.dtype, device=C.device)
        C = torch.cat([C, pad], dim=1)
    return C

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
