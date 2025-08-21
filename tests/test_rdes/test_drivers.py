import jax
import jax.numpy as jnp
from quicksig.rdes.drivers import bm_driver, correlate_bm_driver_against_reference, riemann_liouville_driver
from quicksig.rdes.rde_types import Path
import pytest


@pytest.fixture(scope="module")
def bm_samples() -> Path:
    """Generate and cache multiple BM paths for reuse across tests."""
    seed = 42
    timesteps = 1000
    dim = 1
    num_paths = 500

    key = jax.random.key(seed)
    keys = jax.random.split(key, num_paths)

    vmap_bm = jax.vmap(lambda k: bm_driver(k, timesteps=timesteps, dim=dim))
    paths = vmap_bm(keys)

    return paths


def test_bm_variance_scales_linearly(bm_samples: Path) -> None:
    r"""
    Brownian motion \(B_t\) satisfies \(\Var(B_t) = t\). On the uniform grid
    \(t_k = k/N\), the sample variance across \(M\) paths is linear in time.
    This test regresses empirical variances \(v_k\) against \(t_k\) and expects
    slope \(\hat s \approx 1\):
    \[
      \Var(B_{t_k}) = t_k,\quad \hat s \approx 1.
    \]
    """
    path = bm_samples.path
    timesteps = path.shape[1] - 1
    times = jnp.linspace(0.0, 1.0, timesteps + 1)

    empirical_variances = jnp.var(path, axis=0, ddof=1)
    t = times[1:]
    v = empirical_variances[1:].flatten()

    slope, _ = jnp.polyfit(t, v, 1)
    assert jnp.isclose(slope, 1.0, atol=0.1)


def test_bm_increment_mean_and_variance(bm_samples: Path) -> None:
    r"""
    For a standard Brownian motion \(B_t\) on a uniform grid with \(\Delta t = 1/N\),
    the increments \(\Delta B_k = B_{t_{k+1}} - B_{t_k}\) are i.i.d. \(\mathcal N(0, \Delta t)\).
    Pooling increments across paths, we test
    \[
      \E[\Delta B_k] = 0,\qquad \Var(\Delta B_k) = \Delta t = 1/N.
    \]
    """
    path = bm_samples.path
    timesteps = path.shape[1] - 1

    increments = jnp.diff(path, axis=1).flatten()
    mean_increments = jnp.mean(increments)
    assert jnp.isclose(mean_increments, 0.0, atol=0.1)

    expected_increment_var = 1.0 / timesteps
    empirical_increment_var = jnp.var(increments, ddof=1)
    assert jnp.isclose(empirical_increment_var, expected_increment_var, atol=0.1 * expected_increment_var)


def test_bm_zero_mean_at_all_times(bm_samples: Path) -> None:
    r"""
    For each fixed time \(t_k\), Brownian motion has zero mean: \(\E[B_{t_k}] = 0\).
    We average across simulated paths and check \(\bar B_{t_k} \approx 0\) for all \(k\).
    """
    path = bm_samples.path
    means_at_times = jnp.mean(path, axis=0)
    assert jnp.allclose(means_at_times, 0.0, atol=0.1)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("rho", [-0.9, 0.0, 0.9])
@pytest.mark.parametrize("timesteps", [500, 1000])
@pytest.mark.parametrize("dim", [1, 5])
def test_correlated_bm_driver_correlation(seed: int, rho: float, timesteps: int, dim: int):
    """
    Test that the correlated_bm_driver produces a path with the correct
    correlation.
    """
    key = jax.random.key(seed)
    key1, key2 = jax.random.split(key)

    # Generate two independent Brownian paths
    path1 = bm_driver(key1, timesteps=timesteps, dim=dim)
    path2 = bm_driver(key2, timesteps=timesteps, dim=dim)

    # Generate the correlated path
    correlated_path = correlate_bm_driver_against_reference(path1, path2, rho)

    # The correlation is defined for the increments
    increments1 = jnp.diff(path1.path, axis=0)
    correlated_increments = jnp.diff(correlated_path.path, axis=0)

    # Compute empirical correlation
    # We flatten in case dim > 1
    empirical_corr = jnp.corrcoef(increments1.flatten(), correlated_increments.flatten())[0, 1]

    # Check if the empirical correlation is close to the target rho
    assert jnp.isclose(empirical_corr, rho, atol=1e-1)

    # Also test correlation with the second path
    increments2 = jnp.diff(path2.path, axis=0)
    empirical_corr_vs_path2 = jnp.corrcoef(increments2.flatten(), correlated_increments.flatten())[0, 1]
    expected_corr_vs_path2 = jnp.sqrt(1 - rho**2)
    assert jnp.isclose(empirical_corr_vs_path2, expected_corr_vs_path2, atol=1e-1)


@pytest.fixture(scope="module")
def rl_samples() -> Path:
    """Generate and cache multiple RL fBM paths for reuse across tests."""
    seed = 42
    timesteps = 1000
    hurst = 0.3
    num_paths = 2000

    key = jax.random.key(seed)
    keys = jax.random.split(key, num_paths)

    # Generate BM paths first, then RL paths
    bm_keys = jax.random.split(key, num_paths)
    vmap_bm = jax.vmap(lambda k: bm_driver(k, timesteps=timesteps, dim=1))
    bm_paths = vmap_bm(bm_keys)

    vmap_rl = jax.vmap(lambda k, bm: riemann_liouville_driver(k, timesteps=timesteps, hurst=hurst, bm_path=bm))
    rl_paths = vmap_rl(keys, bm_paths)

    return rl_paths


def test_rl_zero_mean_at_all_times(rl_samples: Path) -> None:
    r"""
    For each fixed time \(t_k\), the RL process has zero mean: \(\E[V_{t_k}] = 0\).
    We average across simulated paths and check \(\bar V_{t_k} \approx 0\) for all \(k\).
    """
    paths = rl_samples.path  # Shape: (num_paths, timesteps+1, dim)
    means_at_times = jnp.mean(paths, axis=0)
    assert jnp.allclose(means_at_times, 0.0, atol=0.1)


def test_rl_marginal_variance_scaling(rl_samples: Path) -> None:
    r"""
    Marginal variance of RL process \(V_t\): for Hurst \(H\),
    \[
      \Var(V_{t_k}) \propto t_k^{2H}.
    \]
    We compute the ratio \(R_k = \widehat{\Var}(V_{\cdot,k}) / t_k^{2H}\) across selected times
    and expect \(R_k \approx 1\).
    """
    paths = rl_samples.path  # Shape: (num_paths, timesteps+1, dim)
    num_paths = paths.shape[0]
    timesteps = paths.shape[1] - 1
    hurst = 0.3

    # Test at several time points (skip t=0)
    test_indices = [100, 250, 500, 750, 1000]
    times = jnp.array(test_indices) / timesteps  # Normalize to [0,1]

    for k, t in zip(test_indices, times):
        if k <= timesteps:
            # Compute sample variance at time k
            var_k = jnp.var(paths[:, k, :], axis=0, ddof=1)
            theoretical_var = t ** (2 * hurst)

            # Ratio should be approximately 1
            ratio = jnp.squeeze(var_k) / theoretical_var  # make scalar
            tolerance = 3.0 / (num_paths**0.5)

            assert jnp.isclose(ratio, 1.0, atol=tolerance), f"Variance ratio at t={float(t):.3f} is {float(ratio):.3f}, expected ~1.0 ± {tolerance:.3f}"


def test_rl_gaussianity(rl_samples: Path) -> None:
    r"""
    Gaussianity: for fixed \(t_k\), \(V_{t_k}\) is Gaussian with variance scaling \(t_k^{2H}\).
    Standardising \(Z_k = V_{t_k} / t_k^{H}\) yields \(Z_k \sim \mathcal N(0,1)\).
    We check mean \(\approx 0\), variance \(\approx 1\), and normality via skew/kurtosis.
    """
    paths = rl_samples.path
    num_paths = paths.shape[0]
    timesteps = paths.shape[1] - 1
    hurst = 0.3

    # Test at a few time points
    test_indices = [250, 500, 750]
    times = jnp.array(test_indices) / timesteps

    for k, t in zip(test_indices, times):
        if k <= timesteps:
            # Extract values at time k
            V_k = paths[:, k, :].flatten()

            # Standardize: Z_k = V_k / t_k^H
            Z_k = V_k / (t**hurst)

            # Test mean ≈ 0 and variance ≈ 1
            mean_Z = jnp.mean(Z_k)
            var_Z = jnp.var(Z_k, ddof=1)

            assert jnp.isclose(mean_Z, 0.0, atol=0.1), f"Standardized mean at t={float(t):.3f} is {float(mean_Z):.3f}, expected ~0.0"
            assert jnp.isclose(var_Z, 1.0, atol=0.2), f"Standardized variance at t={float(t):.3f} is {float(var_Z):.3f}, expected ~1.0"

            # Jarque-Bera test (simplified - just check skewness and kurtosis)
            skew = jnp.mean(((Z_k - mean_Z) / jnp.sqrt(var_Z)) ** 3)
            kurt = jnp.mean(((Z_k - mean_Z) / jnp.sqrt(var_Z)) ** 4)

            # For normality: skew ≈ 0, kurt ≈ 3
            assert jnp.abs(skew) < 0.5, f"Skewness at t={float(t):.3f} is {float(skew):.3f}, expected ~0.0"
            assert jnp.abs(kurt - 3.0) < 1.0, f"Kurtosis at t={float(t):.3f} is {float(kurt):.3f}, expected ~3.0"


def test_rl_correlation_with_bm(rl_samples: Path) -> None:
    r"""
    Correlation with BM: if RL is built from Brownian motion \(W^{(1)}\), and an asset Brownian
    \(W^{(2)}\) is constructed with target correlation \(\rho\) to \(W^{(1)}\), then increments
    of the resulting RL path inherit a non-zero correlation with the driving Brownian increments.
    We validate by correlating pooled increments after passing correlated BMs through the RL driver.
    """
    # Generate correlated BM paths
    seed = 42
    timesteps = 1000
    num_paths = 500
    rho = 0.7  # Target correlation

    key = jax.random.key(seed)
    key1, key2 = jax.random.split(key)
    rl_keys = jax.random.split(key, num_paths)

    # Generate two sets of BM paths
    bm_keys1 = jax.random.split(key1, num_paths)
    bm_keys2 = jax.random.split(key2, num_paths)  # Different seed

    vmap_bm = jax.vmap(lambda k: bm_driver(k, timesteps=timesteps, dim=1))
    bm_paths1 = vmap_bm(bm_keys1)
    bm_paths2 = vmap_bm(bm_keys2)

    # Correlate the second set against the first
    vmap_corr = jax.vmap(lambda p1, p2: correlate_bm_driver_against_reference(p1, p2, rho))
    correlated_bm = vmap_corr(bm_paths1, bm_paths2)

    # Generate RL paths using the correlated BM
    hurst = 0.3
    vmap_rl = jax.vmap(lambda k, bm: riemann_liouville_driver(k, timesteps=timesteps, hurst=hurst, bm_path=bm))
    rl_paths = vmap_rl(rl_keys, correlated_bm)

    # Compute increments of both BM and RL
    bm_increments = jnp.diff(bm_paths1.path, axis=1)  # Shape: (num_paths, timesteps, dim)
    rl_increments = jnp.diff(rl_paths.path, axis=1)

    # Flatten for correlation computation
    bm_flat = bm_increments.flatten()
    rl_flat = rl_increments.flatten()

    # Compute correlation
    empirical_corr = jnp.corrcoef(bm_flat, rl_flat)[0, 1]

    # The RL process should have some correlation with the underlying BM
    # (though not necessarily equal to rho due to the RL transformation)
    assert jnp.abs(empirical_corr) > 0.1, f"Correlation between BM and RL increments is {float(empirical_corr):.3f}, expected > 0.1"


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("timesteps", [1000, 3000])
@pytest.mark.parametrize("hurst", [0.25, 0.5, 0.75])
def test_riemann_liouville_variance_scaling(seed: int, timesteps: int, hurst: float):
    """
    Test the variance scaling of the Riemann-Liouville fBM implementation.
    The variance of fBM scales as t^(2H), so a log-log plot of variance vs. time
    should have a slope of 2H.
    """
    key = jax.random.key(seed)
    N = timesteps
    T = 1.0
    M = 500  # Number of paths for the test

    # Generate M paths for the variance scaling test
    keys = jax.random.split(key, M)
    vmap_driver = jax.vmap(
        lambda k: riemann_liouville_driver(
            k,
            timesteps=N,
            hurst=hurst,
            bm_path=bm_driver(k, timesteps=N, dim=1),
        )
    )
    paths = vmap_driver(keys)

    # The time grid must have N+1 points to match the path length
    times = jnp.linspace(0, T, N + 1)

    # Compute empirical variance at each time point
    variances = jnp.var(paths.path, axis=0, ddof=1)  # unbiased estimate

    # Exclude t=0 (variance zero)
    t = times[1:]
    v = variances[1:]

    # Fit log-log regression
    logt = jnp.log(t)
    logv = jnp.log(v)
    slope, _ = jnp.polyfit(logt, logv, 1)

    # The theoretical slope is 2*H
    expected_slope = 2 * hurst

    # Check if the estimated slope is close to the theoretical value
    assert jnp.isclose(slope, expected_slope, atol=1e-1)
