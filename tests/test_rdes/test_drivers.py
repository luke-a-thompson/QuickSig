import jax
import jax.numpy as jnp
from quicksig.rdes.drivers import bm_driver, correlate_bm_driver_against_reference, riemann_liouville_driver
import pytest

def test_bm_driver():
    """
    Test the standard Brownian motion driver using three key statistical properties:
    1. Variance scaling: Var(B_t) = t (variance grows linearly with time)
    2. Increment independence: Increments are independent and normally distributed
    3. Zero mean: The process has zero mean at all times
    """
    # Test parameters
    seed = 42
    timesteps = 1000
    dim = 1
    num_paths = 500  # Number of paths for statistical testing
    T = 1.0  # Total time
    
    key = jax.random.key(seed)
    keys = jax.random.split(key, num_paths)
    
    # Generate multiple Brownian motion paths
    vmap_bm = jax.vmap(lambda k: bm_driver(k, timesteps=timesteps, dim=dim))
    paths = vmap_bm(keys)  # Shape: (num_paths, timesteps+1, dim)
    
    # Test 1: Variance scaling - Var(B_t) = t
    times = jnp.linspace(0, T, timesteps + 1)
    empirical_variances = jnp.var(paths.path, axis=0, ddof=1)  # Shape: (timesteps+1, dim)
    
    # Exclude t=0 (variance is zero)
    t = times[1:]
    v = empirical_variances[1:].flatten()
    
    # Check that variance scales linearly with time
    # Fit linear regression: variance = slope * time
    slope, _ = jnp.polyfit(t, v, 1)
    expected_slope = 1.0  # For standard Brownian motion
    
    # Allow some tolerance for statistical variation
    assert jnp.isclose(slope, expected_slope, atol=0.1), f"Variance scaling test failed. Expected slope ~1.0, got {slope:.3f}"
    
    # Test 2: Increment independence and normality
    # Compute increments: ΔB_i = B_{i+1} - B_i
    increments = jnp.diff(paths.path, axis=1)  # Shape: (num_paths, timesteps, dim)
    
    # Flatten increments for analysis
    flat_increments = increments.flatten()
    
    # Check that increments have zero mean
    mean_increments = jnp.mean(flat_increments)
    assert jnp.isclose(mean_increments, 0.0, atol=0.1), f"Increment mean test failed. Expected ~0.0, got {mean_increments:.3f}"
    
    # Check that increment variance is approximately dt = 1/timesteps
    expected_increment_var = 1.0 / timesteps
    empirical_increment_var = jnp.var(flat_increments, ddof=1)
    assert jnp.isclose(empirical_increment_var, expected_increment_var, atol=0.1 * expected_increment_var), \
        f"Increment variance test failed. Expected ~{expected_increment_var:.6f}, got {empirical_increment_var:.6f}"
    
    # Test 3: Zero mean at all times
    # Check that the process has zero mean at each time point
    means_at_times = jnp.mean(paths.path, axis=0)  # Shape: (timesteps+1, dim)
    
    # Allow some tolerance for statistical variation
    assert jnp.allclose(means_at_times, 0.0, atol=0.1), \
        f"Zero mean test failed. Process should have zero mean at all times, but got max deviation {jnp.max(jnp.abs(means_at_times)):.3f}"


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

    # Create the correlation matrix
    corr_matrix = jnp.array([[1.0, rho], [rho, 1.0]])

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
