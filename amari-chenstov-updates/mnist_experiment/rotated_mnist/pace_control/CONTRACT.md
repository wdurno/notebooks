# Plan 10 Mathematical Contract

The pace action $a_t$ is chosen from $\mathcal F_t$ and changes the next
encountered distribution $P_{t+1}=\mathcal T_{a_t}(P_t)$. It never rescales a
learner update. The fixed composition $\bar\pi$ is used unchanged by the EWC
objective and by

$$
q_{t+1}=(1-\bar\pi)^2q_t+\frac{\bar\pi^2}{m}.
$$

Its exact transient is

$$
q_t=q_\infty+(1-\bar\pi)^{2t}(q_0-q_\infty),
\qquad q_\infty=\frac{\bar\pi}{m(2-\bar\pi)}.
$$

For
$A_t=S_t+q_tD_t^{\mathrm{old}}$ and
$B_t=D_t^{\mathrm{new}}/m$, the applied surrogate is

$$
R_t(\pi)=(1-\pi)^2A_t+\pi^2B_t.
$$

It is strictly convex when $A_t+B_t>0$, and its unique minimizer is
$A_t/(A_t+B_t)$. Solving this first-order condition for $S_t$ gives

$$
S_t^\dagger=
\frac{\bar\pi}{1-\bar\pi}\frac{D_t^{\mathrm{new}}}{m}
-q_tD_t^{\mathrm{old}}.
$$

With $d\theta_t=a_tv_t$ and zero predictable centering trend,
$a_t^\dagger=\sqrt{[S_t^\dagger]_+/(v_t^T\mathsf M_tv_t)}$. For a nonzero
$\mu_t$, feasible paces are the nonnegative roots of

$$
(v_t^T\mathsf M_tv_t)a^2
-2(v_t^T\mathsf M_t\mu_t)a
+\mu_t^T\mathsf M_t\mu_t-S_t^\dagger=0.
$$

## Candidate Limit Checklist

Before a small-noise display is treated as a theorem, establish:

1. a triangular array with predictable, uniformly bounded $a_{k,m}$;
2. local Lipschitz regularity for $b$ and $\mathcal I$ on the visited compact set;
3. martingale-difference innovations with the claimed conditional covariance;
4. a conditional Lindeberg condition and tightness of the interpolated paths;
5. a remainder satisfying $\sum_{k\le T/h_m}\|r_{k,m}\|\to_p0$;
6. LAN-scale population displacement uniformly over the controlled route; and
7. an explicit treatment of the fast $q_t$ transient before replacing it by
   $q_\infty=\bar\pi/[m(2-\bar\pi)]$.

The finite tracking recursion remains the applied model until these conditions
are proved. Comparisons use both matched route-and-observation and matched
online-budget views; route non-completion within the fixed cap is failure, so
stalling cannot improve the headline result.
