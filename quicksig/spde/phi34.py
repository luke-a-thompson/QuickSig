from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import jax
import jax.numpy as jnp


@dataclass
class SimParams:
    N: int
    L: float
    dx: float  # = L / N
    dt: float
    steps: int
    dtype: jnp.dtype
    seed: int
    use_bandlimited_noise: bool


@dataclass
class Precomp:
    kx: jax.Array  # shape (N,)
    ky: jax.Array  # shape (N,)
    kz: jax.Array  # shape (N,)
    lam: jax.Array  # eigenvalues lambda(q) on rfft grid, shape (N, N, N//2+1)
    M: jax.Array  # 1 + dt * lam on rfft grid, shape (N, N, N//2+1)
    Ceps: float  # renormalisation scalar


def _kvec(N: int, L: float, dtype: jnp.dtype) -> jax.Array:
    q: jax.Array = 2.0 * jnp.pi / float(L) * jnp.fft.fftfreq(N, d=1.0 / float(N))
    return q.astype(dtype)


def _kvec_rfft(N: int, L: float, dtype: jnp.dtype) -> jax.Array:
    q: jax.Array = 2.0 * jnp.pi / float(L) * jnp.fft.rfftfreq(N, d=1.0 / float(N))
    return q.astype(dtype)


def _laplacian_symbol_rfft(kx: jax.Array, ky: jax.Array, kz_r: jax.Array, dx: float) -> jax.Array:
    sx: jax.Array = jnp.sin(0.5 * kx * float(dx)) ** 2
    sy: jax.Array = jnp.sin(0.5 * ky * float(dx)) ** 2
    sz: jax.Array = jnp.sin(0.5 * kz_r * float(dx)) ** 2
    # Broadcast to (Nx, Ny, Nz_r)
    sx3: jax.Array = sx[:, None, None]
    sy3: jax.Array = sy[None, :, None]
    sz3: jax.Array = sz[None, None, :]
    lam: jax.Array = (4.0 / (float(dx) ** 2)) * (sx3 + sy3 + sz3)
    return lam


def _laplacian_symbol_full(kx: jax.Array, ky: jax.Array, kz: jax.Array, dx: float) -> jax.Array:
    sx: jax.Array = jnp.sin(0.5 * kx * float(dx)) ** 2
    sy: jax.Array = jnp.sin(0.5 * ky * float(dx)) ** 2
    sz: jax.Array = jnp.sin(0.5 * kz * float(dx)) ** 2
    sx3: jax.Array = sx[:, None, None]
    sy3: jax.Array = sy[None, :, None]
    sz3: jax.Array = sz[None, None, :]
    lam: jax.Array = (4.0 / (float(dx) ** 2)) * (sx3 + sy3 + sz3)
    return lam


def calibrate_renorm_constant(
    params: SimParams,
    finite_correction: float = 0.0,
    num_tau: int = 32,
    include_sidebands: bool = False,
) -> float:
    """
    Compute the renormalisation scalar C^eps = 3 C0^eps - 9 C1^eps using Zhu–Zhu.

    C0^eps = (1/N^3) sum_{k!=0} 1/(2*lambda(k))
    C1^eps = C11^eps (+ optional sidebands C12) computed via an FFT-accelerated
             quadrature over tau of P_tau and V_tau kernels.

    finite_correction adds a fixed finite adjustment for a chosen convention.
    """
    c0, c1, c_mass = compute_C0_C1(params, num_tau=num_tau, include_sidebands=include_sidebands)
    return float(c_mass + float(finite_correction))


def compute_C0_C1(
    params: SimParams,
    num_tau: int = 32,
    include_sidebands: bool = False,
) -> tuple[float, float, float]:
    """
    Compute (C0, C1, C_mass) following Zhu–Zhu (2015) on the full FFT grid.

    C0 = (1/N^3) * sum_{k!=0} 1 / (2*lambda(k))
    C1 = C11 (+ optional C12 sidebands). We evaluate C11 via
         C11 = 2^-5 * ∫_0^∞ sum_{k1,k2} e^{-τ λ(k1+k2)} ((e^{-τ λ(k1)}-1)(e^{-τ λ(k2)}-1))/(λ(k1)λ(k2)) dτ
    Using circular convolution: for fixed τ, define B_τ(k) = (e^{-τ λ(k)} - 1)/λ(k) with B_τ(0)=0,
    and P_τ(k) = e^{-τ λ(k)}. Then the double sum equals sum_m P_τ(m) * (B_τ * B_τ)[m],
    where * is circular convolution over the index grid, computed by ifftn(fftn(B)^2).

    The τ-integral is approximated by a trapezoidal rule on [0, τ_max], with τ_max
    chosen as 8 / λ_min>0 to capture the exponential tail.
    """
    N: int = params.N
    L: float = params.L
    dx: float = params.dx
    dtype: jnp.dtype = params.dtype

    # Full complex FFT grid eigenvalues λ(k)
    kx: jax.Array = _kvec(N, L, dtype)
    ky: jax.Array = _kvec(N, L, dtype)
    kz: jax.Array = _kvec(N, L, dtype)
    lam: jax.Array = _laplacian_symbol_full(kx, ky, kz, dx)  # (N,N,N)

    # C0: exclude zero mode via safe division
    lam_safe: jax.Array = jnp.where(lam == 0.0, jnp.inf, lam)
    c0: float = float(jnp.mean(0.5 / lam_safe))

    # Quadrature setup for C11
    lam_min_pos: float = float(jnp.min(lam_safe))
    tau_max: float = 8.0 / lam_min_pos
    # Ensure at least 2 points for trapezoid
    n_tau: int = max(int(num_tau), 2)
    taus: jax.Array = jnp.linspace(0.0, tau_max, n_tau, dtype=dtype)
    dtau: jax.Array = taus[1] - taus[0]
    # Trapezoid weights
    w: jax.Array = jnp.ones((n_tau,), dtype=dtype)
    w = w.at[0].set(0.5)
    w = w.at[-1].set(0.5)

    def integrand_for_tau(tau: jax.Array) -> jax.Array:
        P: jax.Array = jnp.exp(-tau * lam)
        # Exclude zero mode by forcing B(0)=0
        B_raw: jax.Array = (P - 1.0) / lam_safe
        B: jax.Array = jnp.where(lam == 0.0, 0.0, B_raw)
        FB: jax.Array = jnp.fft.fftn(B, axes=(0, 1, 2))
        conv_BB: jax.Array = jnp.fft.ifftn(FB * FB, axes=(0, 1, 2)).real
        return jnp.sum(P * conv_BB)

    # Accumulate integral via Python loop (sufficient for precompute)
    acc: jax.Array = jnp.array(0.0, dtype=dtype)
    for i in range(int(n_tau)):
        acc = acc + w[i] * integrand_for_tau(taus[i])
    integral: jax.Array = acc * dtau

    c11: float = float((2.0**-5) * integral / float(N**6))

    c12: float = 0.0
    if include_sidebands:
        # Placeholder: sideband contributions can be added if aliasing is not mitigated.
        # With standard de-aliasing or projector choices this is typically set to 0.
        c12 = 0.0

    c1: float = c11 + c12
    c_mass: float = 3.0 * c0 - 9.0 * c1
    return c0, c1, c_mass


def precompute(params: SimParams, finite_correction: float = 0.0) -> Precomp:
    N: int = params.N
    L: float = params.L
    dx: float = params.dx
    dtype: jnp.dtype = params.dtype
    dt: float = params.dt

    kx: jax.Array = _kvec(N, L, dtype)
    ky: jax.Array = _kvec(N, L, dtype)
    kz: jax.Array = _kvec(N, L, dtype)
    kz_r: jax.Array = _kvec_rfft(N, L, dtype)

    lam_r: jax.Array = _laplacian_symbol_rfft(kx, ky, kz_r, dx)
    M: jax.Array = 1.0 + float(dt) * lam_r

    # Zhu–Zhu constants
    Ceps: float = calibrate_renorm_constant(params, finite_correction=finite_correction)
    return Precomp(kx=kx, ky=ky, kz=kz, lam=lam_r, M=M, Ceps=Ceps)


def _noise_real_space(key: jax.Array, params: SimParams) -> jax.Array:
    N: int = params.N
    dt: float = params.dt
    dx: float = params.dx
    scale: float = (float(dt) ** 0.5) / (float(dx) ** 1.5)
    return jax.random.normal(key, (N, N, N), dtype=params.dtype) * scale


def semi_implicit_step(
    phi: jax.Array,
    key: jax.Array,
    params: SimParams,
    pre: Precomp,
) -> tuple[jax.Array, jax.Array]:
    N: int = params.N
    dtype: jnp.dtype = params.dtype
    dt: float = params.dt

    # Nonlinear drift in real space
    drift: jax.Array = -phi * phi * phi + pre.Ceps * phi
    rhs: jax.Array = phi + float(dt) * drift

    # Additive noise increment matching space-time white noise
    key, sub = jax.random.split(key)
    dW: jax.Array = _noise_real_space(sub, params)
    rhs = rhs + dW

    # Solve (I - dt * Delta_eps) phi^{n+1} = rhs via rFFT diagonalization
    rhs_hat: jax.Array = jnp.fft.rfftn(rhs, axes=(0, 1, 2))
    phi_next_hat: jax.Array = rhs_hat / pre.M
    phi_next: jax.Array = jnp.fft.irfftn(phi_next_hat, s=(N, N, N), axes=(0, 1, 2))
    return phi_next.astype(dtype), key


def simulate(
    params: SimParams,
    pre: Precomp,
    phi0: jax.Array | None = None,
    snapshot_every: int = 0,
    burnin: int = 0,
) -> tuple[jax.Array, jax.Array | None]:
    """
    Run the semi-implicit Euler scheme for the dynamic Phi^4_3 model.

    Returns final field and optionally snapshots stacked along axis 0.
    If snapshot_every <= 0, no snapshots are returned (None).
    If burnin > 0, the first burnin frames are skipped when collecting snapshots.
    """
    N: int = params.N
    steps: int = params.steps
    dtype: jnp.dtype = params.dtype

    if phi0 is None:
        phi: jax.Array = jnp.zeros((N, N, N), dtype=dtype)
    else:
        phi = phi0.astype(dtype)

    def one_step(carry: tuple[jax.Array, jax.Array], _: None) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        state, key = carry
        state_next, key_next = semi_implicit_step(state, key, params, pre)
        return (state_next, key_next), state_next

    key: jax.Array = jax.random.PRNGKey(params.seed)
    (phi_final, _), traj = jax.lax.scan(one_step, (phi, key), xs=None, length=steps)

    snaps: jax.Array | None = None
    if snapshot_every and snapshot_every > 0:
        effective_burnin: int = int(burnin) if int(burnin) > 0 else 0
        traj_post: jax.Array = traj[effective_burnin:] if effective_burnin > 0 else traj
        num_frames: int = steps - effective_burnin
        if num_frames <= 0:
            snaps = traj_post[:0]
        else:
            idx: jax.Array = jnp.arange(num_frames)
            mask: jax.Array = (idx + 1) % int(snapshot_every) == 0
            snaps = traj_post[mask]

    return phi_final, snaps


def structure_factor(phi: jax.Array, L: float) -> jax.Array:
    """
    Compute the structure factor S(q) = |phi_hat(q)|^2 / Volume on the rFFT grid.
    """
    vol: float = float(L) ** 3
    hat_phi: jax.Array = jnp.fft.rfftn(phi, axes=(0, 1, 2))
    S: jax.Array = (hat_phi * jnp.conj(hat_phi)).real / vol
    return S


def two_point_correlation(phi: jax.Array) -> jax.Array:
    """
    Compute the equal-time two-point correlation C(x) = E[phi(0) phi(x)] estimator
    from a single snapshot via Wiener-Khinchin: inverse FFT of power spectrum,
    normalized by the number of sites.
    """
    N: int = phi.shape[0]
    power: jax.Array = jnp.abs(jnp.fft.fftn(phi, axes=(0, 1, 2))) ** 2
    corr: jax.Array = jnp.fft.ifftn(power, axes=(0, 1, 2)).real / float(N**3)
    return corr


def to_tcxyz(snaps: jax.Array) -> jax.Array:
    """
    Convert snapshots of shape (T, X, Y, Z) to (T, C, X, Y, Z) with C=1.
    """
    return snaps[:, None, ...]


__all__: list[str] = [
    "SimParams",
    "Precomp",
    "precompute",
    "calibrate_renorm_constant",
    "semi_implicit_step",
    "simulate",
    "structure_factor",
    "two_point_correlation",
    "to_tcxyz",
]


def main() -> None:
    """
    Run a default Phi^4_3 simulation and print basic info when executed as a script.
    """
    N: int = 48
    L: float = 0.1
    dx: float = L / float(N)
    dt: float = 0.01 * dx * dx
    # Keep physical horizons fixed; recompute steps if dt changes
    base_dt_coeff: float = 0.01  # reference coefficient for physical-time baselines
    total_time: float = 512 * base_dt_coeff * dx * dx
    burnin_time: float = 64 * base_dt_coeff * dx * dx
    sim_steps: int = int(np.ceil(total_time / dt))
    burnin_steps: int = int(np.ceil(burnin_time / dt))

    params: SimParams = SimParams(
        N=N,
        L=L,
        dx=dx,
        dt=dt,
        steps=sim_steps + burnin_steps,
        dtype=jnp.float32,
        seed=0,
        use_bandlimited_noise=False,
    )

    pre: Precomp = precompute(params)
    phi_final, snaps = simulate(params, pre, phi0=None, snapshot_every=1, burnin=burnin_steps)

    S_q: jax.Array = structure_factor(phi_final, L)
    C_x: jax.Array = two_point_correlation(phi_final)

    print("phi_final:", phi_final.shape)
    print("snaps:", None if snaps is None else snaps.shape)
    print("S_q:", S_q.shape, "C_x:", C_x.shape)

    # Save spacetime rollout (snaps) as NPY
    if snaps is not None:
        snaps_np: object
        if params.dtype == jnp.float64:
            snaps_np = np.asarray(jax.device_get(snaps), dtype=np.float64)
        else:
            snaps_np = np.asarray(jax.device_get(snaps), dtype=np.float32)
        out_path: str = "phi34_snaps.npy"
        np.save(out_path, snaps_np)
        print(f"Saved spacetime snaps (NPY) to {out_path}")

        # Also save TCXYZ (add channel dim C=1) for visualization tools
        snaps_tcxyz_np: object = np.expand_dims(snaps_np, axis=1)
        out_path_tcxyz: str = "phi34_snaps_tcxyz.npy"
        np.save(out_path_tcxyz, snaps_tcxyz_np)
        print(f"Saved TCXYZ snaps (NPY) to {out_path_tcxyz}")


if __name__ == "__main__":
    main()
