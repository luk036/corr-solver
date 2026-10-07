# Outline

## Outline

```{=latex}
\tableofcontents
```

# Motivation

## Why Spatial Correlation Matters 🏭

- Feature shrink $\Rightarrow$ process variation is a **first-order** determinant of performance and yield
- Corner-based timing verifies at extreme PVT corners $\Rightarrow$ increasingly pessimistic and expensive
- On a 10-inverter chain, worst-case closure cost **+54% area / +44% power** versus statistical design at $\mu+3\sigma$ [@champac2018timing]
- Statistical timing propagates parameter *distributions* to path delay

## Process Variation Decomposition

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=7mm, font=\scriptsize]
\node[nblue] (d2d) {inter-die\\$X_{D2D}$};
\node[ngreen, right=of d2d] (corr) {WID correlated\\$X_{WID,c}$};
\node[nred, right=of corr] (rand) {WID random\\$X_{WID,r}$};
\node[nyellow, below=9mm of corr] (sum)
  {$\sigma_X^2=\sigma_{D2D}^2+\sigma_{WID,c}^2+\sigma_{WID,r}^2$};
\draw[ar] (d2d) -- (corr);
\draw[ar] (corr) -- (rand);
\draw[ar] (corr) -- (sum);
\end{tikzpicture}
\end{center}
```

- $X=\mu_X+X_{D2D}+X_{WID,c}+X_{WID,r}$; the components are independent, so the variances add
- the deterministic layout-dependent part is modeled and removed
- the purely random part together with measurement error becomes the **nugget**

## Correlation Dominates Path Delay

$$
\sigma_{DP}^2=\sum_{i=1}^{N}\sigma_{D_i}^2+2\sum_{i=1}^{N}\sum_{j=i+1}^{N}\operatorname{Cov}(D_i,D_j)
$$

- only spatially correlated parameters contribute to the covariance terms
- correlation dominates the path-delay variance as logic depth grows
- the model **must be positive definite**, or the covariance matrix is invalid

# Modeling the Correlation Function

## Parametric Kernels

$$\rho(h)=e^{-\alpha h}\ \text{(exp)},\qquad e^{-\alpha h^2}\ \text{(Gaussian)},\qquad \text{Matérn}$$

- positive definite **by construction**, parameterized by a length scale [@matern1986spatial; @rasmussen2006gaussian]
- but they **presuppose the shape**, and their likelihood is **non-convex**
- for a new process the correlation may even be **non-monotone**

## The Non-Parametric Alternative

$$\rho(h)=\sum_i p_i\,\Psi_i(h)$$

- polynomial and B-spline bases commit to **no particular shape**
- they can represent non-monotone correlation
- the price: a constrained, higher-dimensional fit, and positive definiteness must be enforced separately

## The Nugget Effect

- purely random intra-die variation **and** measurement error give white noise of variance $\tau^2$
- this introduces a discontinuity at the origin:
$$\tilde{R}(h)=\begin{cases}1, & h=0\\[2pt] \dfrac{\sigma^2}{\sigma^2+\tau^2}\,R(h), & h>0\end{cases}
\qquad \tilde{R}(\kappa,\psi)=R(\psi)+\kappa I,\quad \kappa=\tfrac{\tau^2}{\sigma^2}$$
- ignoring the nugget severely distorts the recovered correlation function

## Contributions

1. basis-expansion formulation of the covariance; LSQ and MLE fit by a **cutting-plane** method
2. the MLE objective is a **difference of convex** functions $\Rightarrow$ not globally convex
3. solved with the **convex-concave procedure (CCP)**; its fixed points are stationary points
4. **robust initialization**: free-MLE covariance plus a nearest-PD repair -- the main result
5. validated on polynomial & clamped B-spline bases, isotropic & anisotropic fields, in Python / C++ / Rust

# Problem Formulation

## Biased Sample Covariance

$$Y=\frac{1}{M}\sum_{m=1}^{M} y_m y_m^{\top}$$

- $M\ge n$: $Y$ is positive definite; $M<n$: rank-deficient and **singular**
- "more samples is easier": $Y$ approaches the true covariance and the problem becomes better conditioned

## Basis Expansion

$$\rho(h)=\sum_i p_i\Psi_i(h),\qquad \Omega(p)=\sum_{k=1}^{m} p_k F_k,\quad (F_k)_{ij}=\Psi_k(\lVert s_i-s_j\rVert)$$

- $\Omega(p)$ is **affine** in the coefficients $p$ -- inviting a convex program
- but the MLE objective contains $\Omega(p)^{-1}$, and a matrix inverse is not convex

## Least Squares (LSQ)

$$\min_{p,\kappa}\ \lVert\Omega(p)+\kappa I-Y\rVert_F \quad\text{s.t.}\quad \Omega(p)\succeq 0,\ \ \kappa\ge 0$$

- a **convex** problem; the nugget $\kappa I$ gives strict positive definiteness
- it targets the *shape* of the sample covariance and places no upper bound on $\Omega$

## Multi-Chip MLE (MLE-M)

$$\log L=-\tfrac{MN}{2}\log\sigma^2-\tfrac{M}{2}\log\det\tilde{R}-\tfrac{1}{2\sigma^2}\sum_m z_m^{*\top}\tilde{R}^{-1}z_m^{*}$$

- concentrating out $\sigma^2$ gives $\ \log L_0(\kappa,\psi)=-\log\det\tilde{R}-N\log\!\big(\operatorname{tr}(Y\tilde{R}^{-1})\big)$
- pooling **all $M$ chips** into one likelihood yields a single parameter set [@hargreaves2008]
- steps: center $z^{*}$; form $Y$; minimize $g=\log\det\tilde{R}+N\log\operatorname{tr}(Y\tilde{R}^{-1})$; then $\hat\sigma^2=\tfrac{1}{N}\operatorname{tr}(Y\tilde{R}^{-1})$
- $\log\det$ is evaluated via LU, because a direct determinant underflows

# Non-Parametric Bases

## Polynomial Basis: Conditioning Blow-up

$$\varphi_k(d)=d^{k},\qquad k=0,\dots,m-1$$

- every basis function has **global support**, so the design columns correlate as $m$ grows
- $\operatorname{cond}(A)$ climbs from $1.2\times10^{1}$ at $m=2$ to $\mathbf{3.5\times10^{10}}$ at $m=10$
- nine orders of magnitude on the *same* problem

## Clamped B-Spline Basis

$$K(d)=\sum_{i=1}^{m} c_i B_i(d),\qquad B_i\ge 0,\qquad \sum_i B_i=1$$

- local support of roughly three knot spans
- the knots must span the **actual data domain** $[0,d_{\max}]$
- legacy uniform knots extrapolate over most of the domain $\Rightarrow$ numerically meaningless
- clamped knots repeat the end knots $k+1$ times

## Clamped Knots

```{=latex}
\begin{center}
\includegraphics[width=0.95\linewidth]{figures/fig_knots.pdf}
\end{center}
```

## Conditioning Wins

```{=latex}
\begin{center}
\footnotesize
\begin{tabular}{rrrr}
\hline
$m$ & polynomial & B-spline (clamped) & B-spline (legacy) \\
\hline
4  & $1.21\times10^{3}$  & $5.05$ & $40.26$ \\
6  & $1.87\times10^{5}$  & $6.48$ & $99.40$ \\
8  & $6.72\times10^{7}$  & $6.51$ & $203.57$ \\
10 & $3.45\times10^{10}$ & $6.75$ & $409.47$ \\
\hline
\end{tabular}
\end{center}
```

- the clamped B-spline stays near $5$--$7$ across $m=4,\dots,10$, **at no cost in fidelity**
- a non-increasing coefficient sequence gives a monotone spline; apply it to *exactly* the control coefficients

## Design-Matrix Conditioning

```{=latex}
\begin{center}
\includegraphics[width=0.92\linewidth]{figures/fig_cond.pdf}
\end{center}
```

# Convexity and the CCP

## The MLE Objective Is Not Convex

$$f(\Omega)=\underbrace{\operatorname{tr}(\Omega^{-1}Y)}_{\text{convex}}+\underbrace{\log\det\Omega}_{\text{concave}}$$

- a **difference of convex** (DC) functions, hence in general neither convex nor concave
- the ellipsoid method **cannot** be applied directly

## The Convex-Concave Procedure

- $\log\det\Omega$ is concave, so its tangent is a **global upper bound**:
$$\log\det\Omega\le\log\det\Omega_k+\operatorname{tr}\!\big(\Omega_k^{-1}(\Omega-\Omega_k)\big)$$
- dropping constants gives the convex surrogate
$$S_k(\Omega)=\operatorname{tr}(\Omega^{-1}Y)+\operatorname{tr}(M_k\Omega),\qquad M_k=\Omega_k^{-1}$$
- monotone decrease: $f(\Omega_{k+1})\le S_k(\Omega_{k+1})\le S_k(\Omega_k)=f(\Omega_k)$
- a fixed point is a **stationary point** of the true MLE

## CCP Convergence

```{=latex}
\begin{center}
\includegraphics[width=0.9\linewidth]{figures/fig_cccp_bs.pdf}
\end{center}
```

## The Convex Subproblem (LMI)

$$\min_{p}\ \operatorname{tr}\!\big(\Omega(p)^{-1}Y\big)+\operatorname{tr}\!\big(M_k\,\Omega(p)\big)
\quad\text{s.t.}\quad \Omega(p)=\sum_i p_i F_i\succeq 0$$

- a **linear matrix inequality** in $p$: only $m$ variables, independent of $n$
- solved by the **ellipsoid method** [@boyd2004convex], which needs only a separation oracle

## Ellipsoid Method + LMI Oracle

- at $p_c$ the separation oracle returns a **cut** $(g,\beta)$ with $g^{\top}(p-p_c)+\beta\le 0$
- the deep-cut update:
$$p_c^{+}=p_c-\frac{\rho}{\tau^2}\,\tilde g,\qquad P^{+}=\delta\Big(P-\frac{\sigma}{\tau^2}\,\tilde g\,\tilde g^{\top}\Big)$$
- $\mathrm{LDL}^{\top}$ factor of $\Omega(p_c)$: positive pivots $\Rightarrow$ feasible; failure yields a witness cut
- **lazy**: the factorization stops at the first failing pivot, $O(p^3)$
- the volume shrinks $\approx e^{-1/(2m)}$ per step $\Rightarrow O(m^2\log(1/\varepsilon))$ iterations

## CCP Algorithm

1. $M \gets$ `mle_corr_mtx`($Y$)  — the robust initialization of the next section
2. **repeat** until the objective stalls:
3. $\quad$ solve the convex subproblem with $M$ $\Rightarrow$ $x'$
4. $\quad$ if $x'$ is missing or stalled, stop
5. $\quad$ $x\gets x'$; $\ M\gets\Omega(x)^{-1}$

# Robust Initialization

## The Problem: PSD Is Not PD

- the old start used the **LSQ** solution $x_0$ and set $M_0=\Omega(x_0)^{-1}$
- but LSQ enforces only $\Omega(x_0)\succeq 0$ -- the *boundary* is allowed
- the CCP majorization needs $\Omega_0\succ 0$ for $\Omega_0^{-1}$ and $\log\det\Omega_0$
- concrete failure: polynomial basis $F_0=\mathbf{1}$, $x_0=[4,0,0,0]$ $\Rightarrow$ $\Omega(x_0)=4\cdot\mathbf{1}$, rank one $\Rightarrow$ **error**

```{=latex}
\begin{center}
\begin{tikzpicture}[node distance=6mm, font=\scriptsize]
\node[nred] (lsq) {LSQ start $x_0$\\$\Omega(x_0)\succeq0$ (PSD)};
\node[nred, right=10mm of lsq] (sing) {can be singular\\$\Omega^{-1}$ fails};
\node[ngreen, below=12mm of lsq] (free) {free-MLE $\Omega^\star=Y$};
\node[ngreen, right=10mm of free] (clip) {clip $\lambda\ge\varepsilon$\\return precision};
\draw[ar] (lsq) -- (sing);
\draw[ar] (free) -- (clip);
\end{tikzpicture}
\end{center}
```

## The Free Maximum-Likelihood Covariance

$$\min_{\Omega\succ0}\ \log\det\Omega+\operatorname{tr}(\Omega^{-1}Y)$$

- set the gradient to zero: $\Omega^{-1}-Y=0$ $\Rightarrow$ $\ \Omega^{\star}=Y$
- the free-MLE covariance is simply the **sample covariance**
- a natural initialization: the exact minimizer of the objective *without* the family constraint

## Nearest Positive-Definite Repair

- the free problem is unbounded when $Y$ is not positive definite
- repair to the nearest PD matrix in Frobenius norm by **spectral clipping** [@higham1988computing]:
$$\Omega_0=V\operatorname{diag}\!\big(\max(\lambda,\varepsilon)\big)V^{\top}$$
- positive definite by construction $\Rightarrow$ **always invertible**

## Return the Precision Directly

- the CCP never needs $\Omega_0$ itself -- only its inverse
- reciprocate the clipped eigenvalues:
$$\Omega_0^{-1}=V\operatorname{diag}\!\Big(\frac{1}{\max(\lambda,\varepsilon)}\Big)V^{\top}$$
- one symmetric eigendecomposition, **no external SDP solver**
- if $Y\succ 0$ the floor is inactive and the initializer is exactly $Y^{-1}$

## `mle_corr_mtx`

1. symmetrize: $Y\gets(Y+Y^{\top})/2$
2. eigendecompose $Y=V\operatorname{diag}(w)V^{\top}$ (ascending)
3. $\delta\gets\varepsilon\cdot\max(w_{\max},1)$
4. $w\gets\max(w,\delta)$
5. **return** $V\operatorname{diag}(1/w)V^{\top}$, i.e. $\Omega_0^{-1}$ directly

# Numerical Experiments

## Setup and Synthetic Data

- Halton low-discrepancy sites with a known generating kernel, isolating the *algorithm*
- $n=20$ sites, $N=3000$ samples; isotropic $K(d)=4\,e^{-0.12d^2}$
- **Cholesky generator** [@cressie1993statistics]: $V=\sigma^2K+\tau^2I=LL^{\top}$; $y_m=Lx_m$, $x_m\sim\mathcal{N}(0,I)$; $Y=\tfrac{1}{M}\sum_m y_my_m^{\top}$
- synthetic data establishes *algorithmic* behavior, not production readiness

## LSQ vs MLE vs CCP

```{=latex}
\begin{center}
\footnotesize
\begin{tabular}{lrrr}
\hline
problem & LSQ & MLE & CCP \\
\hline
iso $(1,1)$   & $0.128$ & $0.705$ & $0.293$ \\
aniso $(1,3)$ & $0.334$ & $0.451$ & $0.451$ \\
aniso $(3,1)$ & $0.476$ & $0.716$ & $0.683$ \\
\hline
\end{tabular}
\end{center}
```

- LSQ tracks the shape; the MLE oracle is pinned to its bound and near-singular; the CCP recovers the free family MLE
- the MLE gap does **not** close as $N\to\infty$ $\Rightarrow$ structural **model misspecification**

## Fitted Correlation Functions

```{=latex}
\begin{center}
\includegraphics[width=0.92\linewidth]{figures/fig_corr.pdf}
\end{center}
```

## MLE-M vs Alternatives

- the nugget-aware **MLEnug** keeps $\operatorname{err}(R(h))<2.3\%$ in every regime
- **MLEsim** (ignores the nugget) reaches $78\%$ error; MLEnug dominates the least-squares **RESCF** *and* runs faster

```{=latex}
\begin{center}
\footnotesize
\begin{tabular}{llrr}
\hline
scheme & method & $\operatorname{err}(\sigma^2)$ & $\operatorname{err}(R(h))$ \\
\hline
uniform      & RESCF (LSE) & $0.95$--$7.31\%$   & $0.75$--$4.26\%$ \\
uniform      & MLEsim      & $10.98$--$99.18\%$ & $5.44$--$55.72\%$ \\
uniform      & MLEnug      & $0.30$--$4.12\%$   & $0.27$--$2.27\%$ \\
Monte Carlo  & RESCF (LSE) & $0.91$--$2.32\%$   & $1.06$--$2.33\%$ \\
Monte Carlo  & MLEsim      & $7.58$--$86.13\%$  & $42.48$--$77.92\%$ \\
Monte Carlo  & MLEnug      & $0.32$--$1.67\%$   & $0.29$--$1.39\%$ \\
\hline
\end{tabular}
\end{center}
```

## Robust Initialization: Same Optimum

```{=latex}
\begin{center}
\footnotesize
\begin{tabular}{lrr}
\hline
 & LSQ warm start (old) & free-MLE init (new) \\
\hline
CCP objective $f$      & $29.775870$ & $29.775870$ \\
coefficient rel.\ diff & ---         & $1.9\times10^{-4}$ \\
linearization rounds   & $16$        & $18$ \\
singular $\Omega(x_0)$ & error       & works \\
\hline
\end{tabular}
\end{center}
```

- identical optimum; the new initializer adds **two** rounds
- it tolerates a singular start that previously raised an error
- the initialization changes the **optimization path**, not the answer

## Sample-Size Sweeps

- $M\ge n$: the error falls like $1/\sqrt{M}$ and flattens beyond $\approx200$ chips
- $M<n$: $Y$ is rank-deficient and the MLE oracle is infeasible
- the monotone B-spline CCP stays **well-posed** even at the rank-deficient $N=1$

## Accuracy vs Sample Size

```{=latex}
\begin{center}
\includegraphics[width=0.92\linewidth]{figures/fig_samples.pdf}
\end{center}
```

## Anisotropy and Kernel Misspecification

- anisotropic length scales recovered to $\approx1\%$ (Gaussian / Matérn), a few $\%$ (exponential)
- misspecification is **not** cured by more data: fitting a smooth kernel with a rough one inflates $\ell$ many-fold
- no parametric kernel is a safe default $\Rightarrow$ **validate** (fit several, compare held-out likelihood), don't assume

## Anisotropic Correlation Surfaces

```{=latex}
\begin{center}
\includegraphics[width=0.86\linewidth]{figures/fig_aniso_surfaces.png}
\end{center}
```

## Kernel Misspecification

```{=latex}
\begin{center}
\includegraphics[width=0.62\linewidth]{figures/fig_misspec.png}
\end{center}
```

## Cross-Language Verification

- the algorithms were reimplemented in **Python, C++, and Rust**
- conditioning agrees to nine significant digits; the reference value at $m=10$ was adjudicated at sixty digits
- two bugs found and fixed: a triangular inverse that assumed symmetry, and monotonicity index arithmetic that assumed a trailing objective variable
- all three implementations now agree on every reported quantity

# Discussion and Conclusion

## Limitations

- **synthetic only**: Cholesky fields with analytical kernels; no non-stationarity, non-Gaussian noise, or lithographic signatures
- objective alignment: LSQ tracks *shape*, the MLE gives a *calibrated likelihood*
- boundary protection guarantees positive definiteness only for the **first** round
- scalability: an $O(n^3)$ factorization per query limits dense full-chip use ($n>10^3$)
- the site-wise LMI certifies PD only at the sites, not on the continuum (needs Bochner)

## When Is Non-Parametric CCP Warranted?

- family **known** $\Rightarrow$ fit the parametric kernel (simpler, more accurate)
- shape **unknown** or **non-monotone** $\Rightarrow$ clamped monotone B-spline plus CCP
- a misspecified kernel is simply wrong, however good its residual looks

## Conclusion ✅

- non-parametric extraction = basis expansion + cutting-plane fit (LSQ / MLE)
- the MLE objective is a difference of convex functions, solved by the CCP
- **robust init**: free-MLE $\Omega^\star=Y$, repaired by spectral clipping, precision returned directly
- always positive definite, removes the singular failure mode, same optimum at $+2$ iterations
- validated on polynomial / B-spline bases, isotropic / anisotropic fields, in Python / C++ / Rust

## Thank You

```{=latex}
\begin{center}
{\Large Questions?}\\[2mm]
{\small code: \url{https://github.com/luk036/corr-solver}}
\end{center}
```

- paper: `paper/spatial.md`

## References {.allowframebreaks}
