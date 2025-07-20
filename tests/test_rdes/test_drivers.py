import jax
import jax.numpy as jnp
from quicksig.rdes.drivers import bm_driver, correlated_bm_driver, riemann_liouville_driver
import pytest


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("rho", [-0.9, -0.5, 0.0, 0.5, 0.9])
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

    # Create the correlation matrix
    corr_matrix = jnp.array([[1.0, rho], [rho, 1.0]])

    # Generate the correlated path
    correlated_path = correlated_bm_driver(path1, path2, corr_matrix)

    # The correlation is defined for the increments
    increments1 = jnp.diff(path1, axis=0)
    correlated_increments = jnp.diff(correlated_path, axis=0)

    # Compute empirical correlation
    # We flatten in case dim > 1
    empirical_corr = jnp.corrcoef(increments1.flatten(), correlated_increments.flatten())[0, 1]

    # Check if the empirical correlation is close to the target rho
    assert jnp.isclose(empirical_corr, rho, atol=1e-1)

    # Also test correlation with the second path
    increments2 = jnp.diff(path2, axis=0)
    empirical_corr_vs_path2 = jnp.corrcoef(increments2.flatten(), correlated_increments.flatten())[0, 1]
    expected_corr_vs_path2 = jnp.sqrt(1 - rho**2)
    assert jnp.isclose(empirical_corr_vs_path2, expected_corr_vs_path2, atol=1e-1)


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
    variances = jnp.var(paths, axis=0, ddof=1)  # unbiased estimate

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
