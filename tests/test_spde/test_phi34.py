import pytest
import jax
import jax.numpy as jnp

from quicksig.spde.phi34 import SimParams, precompute, simulate


def _block_mean_and_se(series: jax.Array, n_blocks: int = 16) -> tuple[float, float]:
    """
    Split a 1D series into up to n_blocks consecutive chunks, take chunk means,
    return (overall_mean, standard_error_of_mean). Handles short series gracefully.
    """
    N: int = int(series.shape[0])
    if N <= 1:
        mu: float = float(series.mean())
        se: float = 0.0
        return mu, se
    blocks: int = n_blocks
    if N < blocks:
        blocks = max(1, N // 2) or 1
    block_size: int = N // blocks
    if block_size == 0:
        mu = float(series.mean())
        se = float(jnp.std(series, ddof=1) / jnp.sqrt(float(max(1, N))))
        return mu, se
    trimmed: jax.Array = series[: block_size * blocks]
    reshaped: jax.Array = trimmed.reshape((blocks, block_size))
    block_means: jax.Array = jnp.mean(reshaped, axis=1)
    mu = float(jnp.mean(block_means))
    se = float(jnp.std(block_means, ddof=1) / jnp.sqrt(float(blocks)))
    return mu, se


def _radial_power_spectrum_equal_time(phi: jax.Array) -> dict[str, jax.Array]:
    """
    Time-averaged equal-time power spectrum, binned by integer-radius shells in FFT index space.

    Args:
        phi: Array with shape (T, Nx, Ny, Nz)

    Returns:
        dict with keys: shell_ids, P_shell_mean, P_shell_std_over_mean, n_modes
    """
    T: int = int(phi.shape[0])
    Nx: int = int(phi.shape[1])
    Ny: int = int(phi.shape[2])
    Nz: int = int(phi.shape[3])

    # Remove spatial mean per time to avoid k=0 dominating
    mu_t: jax.Array = jnp.mean(phi, axis=(1, 2, 3), keepdims=True)
    centered: jax.Array = phi - mu_t

    power_sum: jax.Array = jnp.zeros((Nx, Ny, Nz), dtype=centered.dtype)
    for t in range(T):
        ft: jax.Array = jnp.fft.fftn(centered[t], axes=(0, 1, 2))
        power_sum = power_sum + (ft.real * ft.real + ft.imag * ft.imag)
    Pk: jax.Array = power_sum / float(T)

    kx_idx: jax.Array = (jnp.fft.fftfreq(Nx) * float(Nx)).reshape((Nx, 1, 1))
    ky_idx: jax.Array = (jnp.fft.fftfreq(Ny) * float(Ny)).reshape((1, Ny, 1))
    kz_idx: jax.Array = (jnp.fft.fftfreq(Nz) * float(Nz)).reshape((1, 1, Nz))
    r_idx: jax.Array = jnp.sqrt(kx_idx * kx_idx + ky_idx * ky_idx + kz_idx * kz_idx)
    shells: jax.Array = jnp.rint(r_idx).astype(jnp.int32)
    max_shell: int = int(jnp.max(shells).item())

    shell_ids: list[int] = []
    P_shell_mean: list[float] = []
    P_shell_relstd: list[float] = []
    n_modes: list[int] = []

    for s in range(max_shell + 1):
        mask: jax.Array = shells == s
        if not bool(jnp.any(mask).item()):
            continue
        vals: jax.Array = Pk[mask]
        mean_val: float = float(jnp.mean(vals))
        # Add tiny epsilon in denominator for stability
        relstd_val: float = float(jnp.std(vals, ddof=1) / (jnp.mean(vals) + 1e-30))
        shell_ids.append(s)
        P_shell_mean.append(mean_val)
        P_shell_relstd.append(relstd_val)
        n_modes.append(int(mask.sum().item()))

    return {
        "shell_ids": jnp.array(shell_ids, dtype=jnp.int32),
        "P_shell_mean": jnp.array(P_shell_mean),
        "P_shell_std_over_mean": jnp.array(P_shell_relstd),
        "n_modes": jnp.array(n_modes, dtype=jnp.int32),
    }


@pytest.fixture()
def phi_snaps() -> jax.Array:
    """
    Generate a Phi^4_3 rollout and return snapshots with shape (T, N, N, N)
    using the full renormalization calibration.
    """
    # Small grid and short trajectory for fast tests
    N: int = 24
    L: float = 0.1
    dx: float = L / float(N)
    dt: float = 0.01 * dx * dx
    T: int = 512
    burnin_steps: int = 128

    params: SimParams = SimParams(
        N=N,
        L=L,
        dx=dx,
        dt=dt,
        steps=burnin_steps + T,
        dtype=jnp.float32,
        seed=123,
        use_bandlimited_noise=False,
    )

    pre = precompute(params)
    _, snaps = simulate(params, pre, phi0=None, snapshot_every=1, burnin=burnin_steps)
    assert snaps is not None
    return snaps


def test_z2_symmetry(phi_snaps: jax.Array) -> None:
    """
    Z2 symmetry: odd moments vanish — mean ≈ 0 and third central moment ≈ 0.
    We use blocked SEs over time to account crudely for autocorrelation.
    """
    T: int = int(phi_snaps.shape[0])

    # Global mean and central third moment
    mu_global: float = float(jnp.mean(phi_snaps))
    m3_central_global: float = float(jnp.mean((phi_snaps - mu_global) ** 3))

    # Per-time means for blocked SE
    mu_t: jax.Array = jnp.mean(phi_snaps, axis=(1, 2, 3))
    mu_bar, mu_se = _block_mean_and_se(mu_t, n_blocks=min(16, max(2, T // 2)))

    # Per-time third moment with per-slice centering
    mu_t_center: jax.Array = mu_t.reshape((T, 1, 1, 1))
    m3_t: jax.Array = jnp.mean((phi_snaps - mu_t_center) ** 3, axis=(1, 2, 3))
    m3_bar, m3_se = _block_mean_and_se(m3_t, n_blocks=min(16, max(2, T // 2)))

    # Tolerances based on blocked standard errors
    assert abs(mu_bar) <= 5.0 * (mu_se + 1e-8), ""
    assert abs(m3_bar) <= 5.0 * (m3_se + 1e-8)

    # Also check global stats are small in absolute terms
    assert abs(mu_global) < 5e-2
    assert abs(m3_central_global) < 5e-2


def test_isotropy_equal_time_power(phi_snaps: jax.Array) -> None:
    """
    Translation/rotation invariance at equal time: radially binned power depends only on |k|.
    Diagnostic: within-shell relative std is small for shells with enough modes.
    """
    spec = _radial_power_spectrum_equal_time(phi_snaps)
    relstd: jax.Array = spec["P_shell_std_over_mean"]
    counts: jax.Array = spec["n_modes"]

    # Filter to shells with sufficient number of modes to be meaningful
    mask: jax.Array = counts >= 24
    if not bool(jnp.any(mask).item()):
        # Fallback for very small grids: require at least 8 modes
        mask = counts >= 8

    sel: jax.Array = relstd[mask]
    median_relstd: float = float(jnp.median(sel)) if bool(jnp.any(mask).item()) else float(jnp.median(relstd))

    # Allow moderate sampling noise; isotropy should keep this reasonably small
    assert median_relstd < 0.35


def _antisymmetric_statistic(phi: jax.Array, tau: int) -> jax.Array:
    """Return per-time spatial averages of φ_t φ_{t+τ}^2 − φ_{t+τ} φ_t^2 as a 1D array."""
    T: int = int(phi.shape[0])
    values: list[jax.Array] = []
    for t in range(0, T - tau):
        a: jax.Array = phi[t]
        b: jax.Array = phi[t + tau]
        val: jax.Array = jnp.mean(a * (b * b) - b * (a * a))
        values.append(val)
    return jnp.array(values)


def test_time_reversal_antisymmetry(phi_snaps: jax.Array) -> None:
    """
    Stationarity/time-reversal invariance: antisymmetric two-time statistic should vanish.
    Check a few small integer lags τ.
    """
    T: int = int(phi_snaps.shape[0])
    taus: list[int] = [t for t in [1, 2, 4] if t < T]
    for tau in taus:
        s_t: jax.Array = _antisymmetric_statistic(phi_snaps, tau)
        s_bar, s_se = _block_mean_and_se(s_t, n_blocks=min(16, max(2, int(s_t.shape[0]) // 2)))
        # Accept if within 4 SE (plus tiny epsilon), which is generous for short runs
        assert abs(s_bar) <= 4.0 * (s_se + 1e-8)
