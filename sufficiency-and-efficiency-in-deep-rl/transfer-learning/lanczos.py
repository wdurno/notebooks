## Lanczos method sketch 
import numpy as np
import torch 
import traceback 
from typing import Callable, Optional, Iterable, Tuple 
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

def _orthonormalize_columns(M: torch.Tensor) -> torch.Tensor:
    Q, _ = torch.linalg.qr(M, mode='reduced')
    return Q

def _rotate_tensor3(T: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    # T' = T ×1 R^T ×2 R^T ×3 R^T
    return torch.einsum('pa,qb,rc,pqr->abc', R, R, R, T)

def _project_scores_batch(U: torch.Tensor, g_batch: torch.Tensor) -> torch.Tensor:
    if g_batch.ndim == 1:              # (p,)
        g_batch = g_batch.unsqueeze(0) # -> (1,p)
    elif g_batch.ndim == 2 and g_batch.shape[1] == 1:  # (p,1)
        g_batch = g_batch.squeeze(1).unsqueeze(0)      # -> (1,p)
    elif g_batch.ndim == 3 and g_batch.shape[-1] == 1: # (b,p,1)
        g_batch = g_batch.squeeze(-1)                  # -> (b,p)
    return g_batch @ U                                  # (b,r)

def _symmetrize(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.T)

def _safe_cholesky_from_cov(
    G: torch.Tensor,
    ridge_scale: float = 1e-10,
    max_tries: int = 6,
    growth: float = 10.0,
    min_eig_floor: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (L, G_spd) with L @ L.T = G_spd, where G_spd is a numerically
    SPD-ified version of the PSD covariance-like matrix G.

    Strategy:
      1) Symmetrize G.
      2) Try Cholesky with ridge = ridge_scale * trace(G)/k; geometric backoff.
      3) If still failing, eig-clip to floor the spectrum, reconstruct, Cholesky.

    Args:
      G: (k,k) symmetric (up to roundoff).
      ridge_scale: base ridge multiplier relative to trace(G)/k.
      max_tries: backoff attempts before eig-clip.
      growth: multiplicative growth for ridge per attempt.
      min_eig_floor: absolute floor for smallest eigen after clipping.

    Returns:
      L: (k,k) lower Cholesky
      G_spd: (k,k) SPD matrix actually factorized
    """
    k = G.shape[0]
    Gs = 0.5 * (G + G.T)
    # scale ridge by average variance; fall back to 1 if trace ~ 0
    tr = torch.trace(Gs).clamp_min(1e-24)
    base = (tr / float(k)).item()
    ridge = ridge_scale * (base if base > 0.0 else 1.0)

    eye = torch.eye(k, device=G.device, dtype=G.dtype)
    # Try growing ridge
    for _ in range(max_tries):
        try:
            L = torch.linalg.cholesky(Gs + ridge * eye)
            return L, (Gs + ridge * eye)
        except RuntimeError:
            ridge *= growth

    # Fallback: eigen clip
    evals, evecs = torch.linalg.eigh(Gs)
    # floor at max(min_eig_floor, small fraction of mean diag)
    mean_diag = (torch.diag(Gs).mean()).clamp_min(1e-24).item()
    floor = max(min_eig_floor, 1e-12 * mean_diag)
    evals_clipped = torch.clamp(evals, min=floor)
    G_spd = (evecs * evals_clipped.unsqueeze(0)) @ evecs.T
    # Final Cholesky (should succeed)
    L = torch.linalg.cholesky(G_spd)
    return L, G_spd


def update_amari_chentsov_tensor(
    get_grad_generator: Callable[[], Iterable[torch.Tensor]],
    information_basis: torch.Tensor,
    prior_tensor: Optional[torch.Tensor] = None,
    prior_information_basis: Optional[torch.Tensor] = None,
    # --- New optional state for whitened-core CP model ---
    prior_L: Optional[torch.Tensor] = None,
    prior_W: Optional[torch.Tensor] = None,   # (r_prev, m) orthonormal in prior whitened core
    prior_alpha: Optional[torch.Tensor] = None, # (m,)
    m_components: Optional[int] = None,       # default: r
    power_steps: int = 1,                     # small (1–3) is usually enough per call
    power_shift: float = 0.0,                 # tiny stabilizer in core power-step (e.g., 1e-4)
    ridge_scale: float = 1e-10,               # for Cholesky / whitening
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Updates a small-gram estimate of the Amari-Chentsov tensor of size [r x r x r]
    in the (current) information subspace AND maintains a whitened-core CP model (L,W,alpha).

    Inputs:
    - get_grad_generator: function returning an iterator of score tensors g.
      Each yielded item may be shaped (p,), (p,1), (b,p), or (b,p,1).
    - information_basis: approx top-r Fisher directions, shape (p, r0). Need not
      be normalized; we orthonormalize to U (p, r).
    - prior_tensor: optional previous small-gram tensor in its prior basis, (r_prev, r_prev, r_prev).
    - prior_information_basis: optional prior basis (p, r_prev). If provided with prior_tensor,
      we rotate prior_tensor into the new basis before updating.

    New optional inputs (whitened-core model):
    - prior_L: previous Cholesky for the Fisher core in the prior basis, (r_prev, r_prev).
    - prior_W: previous orthonormal CP directions in the prior whitened core, (r_prev, m).
    - prior_alpha: previous weights, (m,).
    - m_components: number of CP components to track in the core (default: r).
    - power_steps: number of stochastic power iterations to apply (default: 1).
    - power_shift: a tiny shift added to each power step in core-space for stability.
    - ridge_scale: small ridge multiplier for core Cholesky.

    Outputs (tuple):
    - updated_tensor: (r, r, r)   the accumulated small-gram tensor in the NEW basis (sum, not avg)
    - L: (r, r)                   current core Cholesky factor s.t. Gk ≈ L L^T
    - W: (r, m)                   current orthonormal CP directions in whitened core
    - alpha: (m,)                 current CP weights

    Notes:
    - We continue to maintain the raw 3rd-moment small-gram (sum of y⊗y⊗y), so legacy code still works.
    - Additionally we (a) compute/transport the Fisher core and whitener L, and
      (b) update a low-variance CP model (W,alpha) in the whitened r-space.
    '''
    # 1) Orthonormalize current info basis U (p, r)
    U = _orthonormalize_columns(information_basis)
    p, r = U.shape
    device, dtype = U.device, U.dtype
    m = m_components if m_components is not None else r

    # 2) Accumulate current projected stats (sum of y⊗y⊗y and sum of y y^T, and count)
    T_sum = torch.zeros((r, r, r), device=device, dtype=dtype)
    Gk_sum = torch.zeros((r, r), device=device, dtype=dtype)
    n_total = 0

    # Single pass: accumulate sums
    grad_generator = get_grad_generator()
    for g_batch in grad_generator():
        g_batch = g_batch.to(device=device, dtype=dtype)
        Y = _project_scores_batch(U, g_batch)         # (b, r)
        b = Y.shape[0]
        n_total += b

        # 3rd moment sum
        T_sum += torch.einsum('bi, bj, bk -> ijk', Y, Y, Y)
        # 2nd moment sum
        Gk_sum += (Y.T @ Y)

    # If no data, keep prior (transported) if available, otherwise zeros
    if n_total == 0:
        # Transport prior small-gram if provided; else zeros
        if prior_tensor is not None and prior_information_basis is not None:
            U_prev = _orthonormalize_columns(prior_information_basis.to(device=device, dtype=dtype))
            R = U_prev.T @ U
            T_new = _rotate_tensor3(prior_tensor.to(device=device, dtype=dtype), R)  # (r,r,r)
        else:
            T_new = torch.zeros((r, r, r), device=device, dtype=dtype)
        # For core model, if prior_L/W/alpha provided AND prior basis provided, transport; else init
        if prior_L is not None and prior_W is not None and prior_alpha is not None and prior_information_basis is not None:
            U_prev = _orthonormalize_columns(prior_information_basis.to(device=device, dtype=dtype))
            R = U_prev.T @ U
            # Fisher core unknown (no new samples): keep L via similarity w/ small ridge
            Gk_old = prior_L @ prior_L.T
            Gk_new = _symmetrize(R.T @ Gk_old @ R)
            rho = ridge_scale * torch.trace(Gk_new).clamp_min(1e-12) / max(1, r)
            L = torch.linalg.cholesky(Gk_new + rho * torch.eye(r, device=device, dtype=dtype))
            # A = L^{-1} R^T L_old  (solve triangular)
            A = torch.linalg.solve(L, R.T @ prior_L.to(device=device, dtype=dtype))
            W_raw = A @ prior_W.to(device=device, dtype=dtype)
            W, _ = torch.linalg.qr(W_raw, mode='reduced')
            alpha = prior_alpha.to(device=device, dtype=dtype)
        else:
            # cold start core model
            L = torch.eye(r, device=device, dtype=dtype)
            W = torch.linalg.qr(torch.randn(r, m, device=device, dtype=dtype), mode='reduced')[0]
            alpha = torch.zeros((m,), device=device, dtype=dtype)
        return T_new, L, W, alpha

    # 3) Build Fisher core Gk and whitener L
    # Use average (second moment), then ridge + Cholesky
    Gk = Gk_sum / float(n_total)                    # (r, r)
    Gk = _symmetrize(Gk)
    #rho = ridge_scale * torch.trace(Gk).clamp_min(1e-12) / max(1, r)
    #L = torch.linalg.cholesky(Gk + rho * torch.eye(r, device=device, dtype=dtype))  # (r, r)
    L, _ = _safe_cholesky_from_cov(Gk, ridge_scale=ridge_scale) 

    # 4) Construct/transport small-gram in NEW basis:
    if prior_tensor is not None and prior_information_basis is not None:
        U_prev = _orthonormalize_columns(prior_information_basis.to(device=device, dtype=dtype))
        R = U_prev.T @ U
        T_init = _rotate_tensor3(prior_tensor.to(device=device, dtype=dtype), R)  # (r, r, r)
    else:
        T_init = torch.zeros((r, r, r), device=device, dtype=dtype)

    T_new = T_init + T_sum  # still a SUM (caller can normalize/EMA if desired)

    # 5) Update whitened-core CP model (W, alpha)
    # Transport prior (if given) into the NEW whitened core: A = L^{-1} R^T L_old
    if prior_W is not None and prior_alpha is not None and prior_L is not None and prior_information_basis is not None:
        U_prev = _orthonormalize_columns(prior_information_basis.to(device=device, dtype=dtype))
        R = U_prev.T @ U
        A = torch.linalg.solve(L, R.T @ prior_L.to(device=device, dtype=dtype))
        W_raw = A @ prior_W.to(device=device, dtype=dtype)   # (r, m_prev)
        W_tr, _ = torch.linalg.qr(W_raw, mode='reduced')
        # If m changes, adjust
        if m is None:
            W = W_tr
            m = W.shape[1]
        else:
            # pad or truncate to m
            if W_tr.shape[1] >= m:
                W = W_tr[:, :m]
            else:
                extra = torch.linalg.qr(torch.randn(r, m - W_tr.shape[1], device=device, dtype=dtype), mode='reduced')[0]
                W = torch.cat([W_tr, extra], dim=1)
        # alpha carries over (pad/truncate)
        alpha_prev = prior_alpha.to(device=device, dtype=dtype)
        if alpha_prev.numel() >= m:
            alpha = alpha_prev[:m].clone()
        else:
            alpha = torch.zeros((m,), device=device, dtype=dtype)
            alpha[:alpha_prev.numel()] = alpha_prev
    else:
        # cold start W, alpha in whitened core
        W = torch.linalg.qr(torch.randn(r, m, device=device, dtype=dtype), mode='reduced')[0]
        alpha = torch.zeros((m,), device=device, dtype=dtype)

    # 6) Perform a few stochastic power steps in whitened core (variance-efficient)
    # We need whitened projected scores z = L^{-1} U^T g. We'll stream again (no big memory).
    # Accumulate the necessary core quantities per step.
    for _ in range(max(1, power_steps)):
        # Accumulate Y2Z := E[ z ( (z W)^2 ) ]  -> shape (r, m)
        Y2Z = torch.zeros((r, m), device=device, dtype=dtype)
        C3  = torch.zeros((m,), device=device, dtype=dtype)  # for alpha update: E[(z^T w)^3]
        n2  = 0
        grad_generator = get_grad_generator()
        for g_batch in grad_generator():
            g_batch = g_batch.to(device=device, dtype=dtype)
            Y = _project_scores_batch(U, g_batch)        # (b, r)
            # z = L^{-1} y   (solve triangular: L z^T = Y^T)
            # Use solve for stability; compute z row-wise via right-solve on transposes
            Zt = torch.linalg.solve(L, Y.T)              # (r, b)
            Z = Zt.T                                     # (b, r)
            b = Z.shape[0]
            n2 += b
            GW = Z @ W                                   # (b, m)
            GW2 = GW * GW                                # (b, m)
            # E[z ( (z W)^2 )]  -> (r, m)
            Y2Z += Z.T @ GW2
            # cubic scalars for alpha: mean((z^T w_i)^3)
            C3  += (GW ** 3).sum(dim=0)

        if n2 > 0:
            Y2Z = Y2Z / float(n2)
            C3  = C3  / float(n2)
        # shifted power step + block-QR ("Gram–Schmidt")
        W_new = Y2Z + (power_shift * W)
        W, _ = torch.linalg.qr(W_new, mode='reduced')
        # smooth alpha a bit (EMA-ish) to avoid jitter; here we just overwrite with the fresh estimate:
        alpha = C3

    return T_new, L, W, alpha

def combine_psd_plus_sym_core(
    A: torch.Tensor,
    U: torch.Tensor,
    Mk: Optional[torch.Tensor] = None,
    r: Optional[int] = None,
    tol: Optional[float] = None,
    pad_zeros: bool = False,
    # --- New: supply whitened-core CP model to form Mk on the fly ---
    L: Optional[torch.Tensor] = None,           # (r, r)
    W: Optional[torch.Tensor] = None,           # (r, m)
    alpha: Optional[torch.Tensor] = None,       # (m,)
    delta: Optional[torch.Tensor] = None,       # (p,) or (p,1)
    ridge_scale: float = 1e-10,                 # numeric ridge for core ops
) -> torch.Tensor:
    """
    Return C such that C C^T is the Frobenius-nearest rank-r PSD approximation to:
        AA^T + U Mk U^T

    - If Mk is provided (k×k symmetric), we use it directly (legacy path).
    - Otherwise, if (L, W, alpha, delta) are provided, we form the increment in the
      **whitened core**:
          delta_U = U^T delta
          hat_delta = L^T delta_U
          tilde_M = W diag(alpha ⊙ (W^T hat_delta)) W^T
          Mk = L tilde_M L^T

      and proceed.

    The combine is done in the Fisher metric by whitening with G_k ≈ (U^T A)(U^T A)^T:

    Steps (core-space):
      1) G_k = (U^T A)(U^T A)^T ;  L_g L_g^T = G_k + ρ I
      2) S = L_g^{-1} Mk L_g^{-T}
      3) Eig of (I + S), clip negatives, keep top r positives
      4) C = U L_g W_sel diag(sqrt(μ_sel))

    If fewer than r positive modes exist, optionally pad zeros to keep width r.
    """
    n = A.shape[0]
    if r is None:
        r = A.shape[1]
    device, dtype = A.device, A.dtype

    # Orthonormalize U to be safe (tall-skinny)
    U = _orthonormalize_columns(U.to(device=device, dtype=dtype))
    k = U.shape[1]

    # Build / validate Mk
    if Mk is None:
        if (L is None) or (W is None) or (alpha is None) or (delta is None):
            raise ValueError("Either Mk must be provided, or (L, W, alpha, delta) must all be given.")
        # delta_U and hat_delta
        if delta.ndim == 2 and delta.shape[1] == 1:
            delta = delta.squeeze(1)
        delta = delta.to(device=device, dtype=dtype)
        delta_U = U.T @ delta                        # (k,)
        hat_delta = L.T @ delta_U                    # (k,)
        # tilde_M (whitened core): W diag(alpha ⊙ (W^T hat_delta)) W^T
        s = (W.T @ hat_delta)                        # (m,)
        diag_vals = alpha * s                         # (m,)
        tilde_M = (W * diag_vals.unsqueeze(0)) @ W.T # (k,k)
        Mk = L @ tilde_M @ L.T                       # map back to core
        Mk = _symmetrize(Mk)
    else:
        Mk = _symmetrize(Mk.to(device=device, dtype=dtype))

    # Build Fisher core G_k ≈ (U^T A)(U^T A)^T and whitener L_g
    R = U.T @ A                                      # (k, r)
    Gk = R @ R.T                                     # (k, k)
    Gk = _symmetrize(Gk)
    #rho = ridge_scale * torch.trace(Gk).clamp_min(1e-12) / max(1, k)
    #Lg = torch.linalg.cholesky(Gk + rho * torch.eye(k, device=device, dtype=dtype)) ## numerically unstable 
    Lg, _ = _safe_cholesky_from_cov(Gk, ridge_scale=1e-10)

    # Whiten the increment core: S = Lg^{-1} Mk Lg^{-T}
    Linv = torch.cholesky_inverse(Lg)
    S = Linv @ Mk @ Linv.T
    S = _symmetrize(S)

    # Eigendecompose (I + S), PSD-clip, keep top-r positives
    evals, vecs = torch.linalg.eigh(torch.eye(k, device=device, dtype=dtype) + S)
    if tol is None:
        #tol = 1e-12 * torch.trace(evals.abs()).clamp_min(1e-12).item()
        tol = 1e-12 * evals.abs().sum().clamp_min(1e-12).item()
    pos = evals > tol
    if not torch.any(pos):
        C = torch.zeros((n, 0), device=device, dtype=dtype)
        if pad_zeros:
            C = torch.zeros((n, r), device=device, dtype=dtype)
        return C

    evals_pos = evals[pos]
    vecs_pos = vecs[:, pos]
    idx = torch.argsort(evals_pos, descending=True)
    evals_sel = evals_pos[idx][:r]
    vecs_sel  = vecs_pos[:, idx][:, :evals_sel.numel()]

    # Lift: C = U Lg V_sel diag(sqrt(mu_sel))
    C = U @ (Lg @ (vecs_sel * torch.sqrt(evals_sel).unsqueeze(0)))

    # Optional: pad zeros to keep a fixed column count r
    if pad_zeros and C.shape[1] < r:
        pad = torch.zeros((n, r - C.shape[1]), device=device, dtype=dtype)
        C = torch.cat([C, pad], dim=1)
    return C

def make_matrix_core(
    U: torch.Tensor,
    Mk: Optional[torch.Tensor] = None,
    *,
    L: Optional[torch.Tensor] = None,        # (k, k) Cholesky of core Fisher; whitener is L^{-1}
    W: Optional[torch.Tensor] = None,        # (k, m) orthonormal in whitened core
    alpha: Optional[torch.Tensor] = None,    # (m,) CP weights
    delta: Optional[torch.Tensor] = None,    # (p,) or (p,1)
    return_whitened: bool = False,           # if True, also return tilde_M (whitened core)
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Construct the small-core symmetric increment Mk (k×k) to represent U Mk U^T.

    Two modes:
      1) Explicit core (legacy): pass Mk directly -> returns (symmetrized Mk, None).
      2) CP-in-whitened-core: pass (L, W, alpha, delta) -> builds
            delta_U  = U^T delta
            hat_delta = L^T delta_U                           # covariant transform
            tilde_M  = W diag(alpha ⊙ (W^T hat_delta)) W^T    # in whitened core
            Mk       = L tilde_M L^T                          # map back to core
         returns (Mk_sym, tilde_M_sym if return_whitened else None)

    Args:
      U:  (p, k) tall-skinny information basis for the core subspace (columns approx orthonormal).
      Mk: (k, k) optional explicit symmetric increment core (if provided, other CP args are ignored).
      L:  (k, k) Cholesky factor of core Fisher G_k ≈ L L^T (used only if Mk is None).
      W:  (k, m) orthonormal CP directions in whitened core (used only if Mk is None).
      alpha: (m,) CP weights (used only if Mk is None).
      delta: (p,) or (p,1) step vector in ambient space (used only if Mk is None).
      return_whitened: additionally return the whitened-core matrix tilde_M.

    Returns:
      Mk_sym:         (k, k) symmetric small-core increment.
      tilde_M_sym:    (k, k) symmetric whitened-core increment (or None if not requested / Mk given).

    Notes:
      - Assumes U’s columns span the working core; if U is not strictly orthonormal, you can
        orthonormalize outside this helper (kept lean by design).
      - Shapes and dtypes are aligned to U for safety.
    """
    device, dtype = U.device, U.dtype

    # Path 1: explicit Mk
    if Mk is not None:
        Mk = Mk.to(device=device, dtype=dtype)
        return _symmetrize(Mk), (None if not return_whitened else _symmetrize(Mk))

    # Path 2: build from (L, W, alpha, delta)
    if any(x is None for x in (L, W, alpha, delta)):
        raise ValueError("Either provide Mk explicitly, or all of (L, W, alpha, delta).")

    L = L.to(device=device, dtype=dtype)
    W = W.to(device=device, dtype=dtype)
    alpha = alpha.to(device=device, dtype=dtype)

    # delta handling and projection to core
    delta = delta.to(device=device, dtype=dtype)
    if delta.ndim == 2 and delta.shape[1] == 1:
        delta = delta.squeeze(1)                        # (p,)
    if delta.ndim != 1:
        raise ValueError("delta must be (p,) or (p,1).")

    # Core coordinates and covariant transform
    delta_U = U.T @ delta                               # (k,)
    hat_delta = L.T @ delta_U                           # (k,)

    # Whitened-core increment: tilde_M = W diag(alpha ⊙ (W^T hat_delta)) W^T
    s = (W.T @ hat_delta)                               # (m,)
    diag_vals = alpha * s                               # (m,)
    tilde_M = (W * diag_vals.unsqueeze(0)) @ W.T        # (k, k)

    # Map back to core: Mk = L tilde_M L^T
    Mk_core = L @ tilde_M @ L.T

    tilde_M = _symmetrize(tilde_M)
    Mk_core = _symmetrize(Mk_core)

    return Mk_core, (tilde_M if return_whitened else None)
