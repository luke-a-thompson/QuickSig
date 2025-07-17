import jax
import jax.numpy as jnp
from quicksig.rdes.drivers import riemann_liouville_driver, bm_driver
import pytest


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("timesteps", [1000, 3000])
@pytest.mark.parametrize("hurst", [0.25, 0.5, 0.75])
def test_variance_scaling(seed: int, timesteps: int, hurst: float):
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
