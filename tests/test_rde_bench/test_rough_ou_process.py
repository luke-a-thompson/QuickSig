import jax
import jax.numpy as jnp
import pytest
from quicksig.rde_bench.rough_ou_process import rough_ou_process
from quicksig.rde.rde_types import Path


@pytest.fixture(scope="module")
def rough_ou_samples() -> tuple[Path, dict]:
    """Generate and cache multiple rough OU process paths for reuse across tests."""
    seed = 42
    timesteps = 1000
    dim = 1
    num_paths = 1000
    
    # Rough OU parameters
    theta = 0.5
    mu = 0.0
    sigma = 0.3
    hurst = 0.7
    x0 = 1.0
    
    key = jax.random.key(seed)
    keys = jax.random.split(key, num_paths)
    
    vmap_rough_ou = jax.vmap(
        lambda k: rough_ou_process(k, timesteps=timesteps, dim=dim, theta=theta, mu=mu, sigma=sigma, hurst=hurst, x0=x0)
    )
    paths = vmap_rough_ou(keys)
    
    params = {
        "theta": theta,
        "mu": mu,
        "sigma": sigma,
        "hurst": hurst,
        "x0": x0,
        "timesteps": timesteps,
    }
    
    return paths, params


def test_rough_ou_analytical_expectation(rough_ou_samples: tuple[Path, dict]) -> None:
    r"""
    Test that the rough OU process matches the analytical expectation.
    
    For the rough Ornstein-Uhlenbeck process dX_t = θ(μ - X_t)dt + σdW_t^H,
    the expected value still follows:
    \[
        \E[X_t] = μ + (X_0 - μ)e^{-θt}
    \]
    because the fractional Brownian motion has zero mean.
    We test this at multiple time points along the path.
    """
    paths, params = rough_ou_samples
    theta = params["theta"]
    mu = params["mu"]
    x0 = params["x0"]
    timesteps = params["timesteps"]
    
    # Test at several time points
    test_indices = [100, 250, 500, 750, 1000]
    times = jnp.array(test_indices) / timesteps  # Normalize to [0, 1]
    
    for idx, t in zip(test_indices, times):
        # Empirical mean at time t
        empirical_mean = jnp.mean(paths.path[:, idx, 0])
        
        # Theoretical expectation: E[X_t] = μ + (x0 - μ)e^(-θt)
        theoretical_mean = mu + (x0 - mu) * jnp.exp(-theta * t)
        
        # Standard error for the mean
        std_dev = jnp.std(paths.path[:, idx, 0], ddof=1)
        se = std_dev / jnp.sqrt(paths.path.shape[0])
        
        # Check within 3 standard errors
        assert jnp.abs(empirical_mean - theoretical_mean) <= 3.0 * se, (
            f"At t={float(t):.3f}: empirical mean {float(empirical_mean):.4f} "
            f"vs theoretical {float(theoretical_mean):.4f} (SE={float(se):.4f})"
        )


def test_rough_ou_path_structure() -> None:
    r"""
    Test that the rough OU process returns a Path object with correct structure.
    
    Verifies:
    - Returns a Path object
    - Path has correct shape (timesteps + 1, dim)
    - Initial condition is respected
    - Interval is correct
    """
    key = jax.random.key(123)
    timesteps = 100
    dim = 3
    theta = 1.0
    mu = 2.0
    sigma = 0.5
    hurst = 0.6
    x0 = 1.5
    
    path = rough_ou_process(key, timesteps, dim, theta, mu, sigma, hurst, x0)
    
    # Check it's a Path object
    assert isinstance(path, Path)
    
    # Check shape: (timesteps + 1, dim)
    assert path.path.shape == (timesteps + 1, dim)
    
    # Check initial condition
    assert jnp.allclose(path.path[0, :], x0, atol=1e-6)
    
    # Check interval
    assert path.interval == (0, timesteps + 1)
    
    # Check properties
    assert path.num_timesteps == timesteps + 1
    assert path.ambient_dimension == dim


@pytest.mark.parametrize("hurst", [0.3, 0.5, 0.7])
def test_rough_ou_hurst_parameter(hurst: float) -> None:
    r"""
    Test that the rough OU process works with different Hurst parameters.
    
    The Hurst parameter H controls the memory characteristics:
    - H < 0.5: short memory (anti-persistent)
    - H = 0.5: memoryless (standard Brownian motion)
    - H > 0.5: long memory (persistent)
    
    We verify that paths are generated correctly for different H values
    and that mean reversion still occurs.
    """
    seed = 789
    timesteps = 1000
    dim = 1
    num_paths = 500
    theta = 1.0
    mu = 0.0
    sigma = 0.4
    x0 = 2.0
    
    key = jax.random.key(seed)
    keys = jax.random.split(key, num_paths)
    
    vmap_rough_ou = jax.vmap(
        lambda k: rough_ou_process(k, timesteps=timesteps, dim=dim, theta=theta, mu=mu, sigma=sigma, hurst=hurst, x0=x0)
    )
    paths = vmap_rough_ou(keys)
    
    # Check shape
    assert paths.path.shape == (num_paths, timesteps + 1, dim)
    
    # Check mean reversion towards mu
    final_values = paths.path[:, -1, 0]
    empirical_mean = jnp.mean(final_values)
    
    # At t=1.0, with theta=1.0, the process should have moved significantly towards mu
    t = 1.0
    theoretical_mean = mu + (x0 - mu) * jnp.exp(-theta * t)
    
    # Standard error
    std_dev = jnp.std(final_values, ddof=1)
    se = std_dev / jnp.sqrt(num_paths)
    
    # Check within 3 standard errors
    assert jnp.abs(empirical_mean - theoretical_mean) <= 3.0 * se, (
        f"H={hurst}: empirical mean {float(empirical_mean):.4f} "
        f"vs theoretical {float(theoretical_mean):.4f} (SE={float(se):.4f})"
    )


def test_rough_ou_parameter_validation() -> None:
    r"""
    Test that the rough OU process properly validates input parameters.
    """
    key = jax.random.key(123)
    timesteps = 100
    dim = 1
    
    # Valid parameters for baseline
    theta = 1.0
    mu = 0.0
    sigma = 0.5
    hurst = 0.7
    x0 = 0.0
    
    # Test invalid theta
    with pytest.raises(ValueError, match="theta must be positive"):
        rough_ou_process(key, timesteps, dim, theta=-0.5, mu=mu, sigma=sigma, hurst=hurst, x0=x0)
    
    with pytest.raises(ValueError, match="theta must be positive"):
        rough_ou_process(key, timesteps, dim, theta=0.0, mu=mu, sigma=sigma, hurst=hurst, x0=x0)
    
    # Test invalid sigma
    with pytest.raises(ValueError, match="sigma must be non-negative"):
        rough_ou_process(key, timesteps, dim, theta=theta, mu=mu, sigma=-0.1, hurst=hurst, x0=x0)
    
    # Test invalid hurst (must be in (0, 1))
    with pytest.raises(ValueError, match="hurst must be in \\(0, 1\\)"):
        rough_ou_process(key, timesteps, dim, theta=theta, mu=mu, sigma=sigma, hurst=0.0, x0=x0)
    
    with pytest.raises(ValueError, match="hurst must be in \\(0, 1\\)"):
        rough_ou_process(key, timesteps, dim, theta=theta, mu=mu, sigma=sigma, hurst=1.0, x0=x0)
    
    with pytest.raises(ValueError, match="hurst must be in \\(0, 1\\)"):
        rough_ou_process(key, timesteps, dim, theta=theta, mu=mu, sigma=sigma, hurst=1.5, x0=x0)


@pytest.mark.parametrize("theta", [0.5, 2.0])
@pytest.mark.parametrize("mu", [-1.0, 0.0, 2.0])
def test_rough_ou_mean_reversion(theta: float, mu: float) -> None:
    r"""
    Test that the rough OU process exhibits mean reversion behavior.
    
    Even with fractional Brownian motion, the drift term θ(μ - X_t)
    should cause the process to revert towards the mean μ over time.
    
    We verify that the empirical mean at later times converges
    towards the theoretical expectation.
    """
    seed = 456
    timesteps = 2000
    dim = 1
    num_paths = 1000
    sigma = 0.4
    hurst = 0.6
    x0 = 5.0  # Start far from mean to test reversion
    
    key = jax.random.key(seed)
    keys = jax.random.split(key, num_paths)
    
    vmap_rough_ou = jax.vmap(
        lambda k: rough_ou_process(k, timesteps=timesteps, dim=dim, theta=theta, mu=mu, sigma=sigma, hurst=hurst, x0=x0)
    )
    paths = vmap_rough_ou(keys)
    
    # At late time (t=1.0), check mean
    final_values = paths.path[:, -1, 0]
    empirical_mean = jnp.mean(final_values)
    
    # Theoretical expectation at t=1.0
    t = 1.0
    theoretical_mean = mu + (x0 - mu) * jnp.exp(-theta * t)
    
    # Standard error
    std_dev = jnp.std(final_values, ddof=1)
    se = std_dev / jnp.sqrt(num_paths)
    
    # Check within 3 standard errors
    assert jnp.abs(empirical_mean - theoretical_mean) <= 3.0 * se, (
        f"θ={theta}, μ={mu}: empirical mean {float(empirical_mean):.4f} "
        f"vs theoretical {float(theoretical_mean):.4f} (SE={float(se):.4f})"
    )
    
    # Also verify that the mean has moved towards mu from x0
    if x0 > mu:
        assert empirical_mean < x0, "Process should revert downward towards mean"
        assert empirical_mean > mu - 3.0 * se, "Process should not overshoot mean by too much"
    elif x0 < mu:
        assert empirical_mean > x0, "Process should revert upward towards mean"
        assert empirical_mean < mu + 3.0 * se, "Process should not overshoot mean by too much"


def test_rough_ou_default_initial_condition() -> None:
    r"""
    Test that when x0 is not provided, it defaults to mu.
    """
    key = jax.random.key(999)
    timesteps = 100
    dim = 2
    theta = 1.0
    mu = 3.5
    sigma = 0.2
    hurst = 0.5
    
    # Call without x0
    path = rough_ou_process(key, timesteps, dim, theta, mu, sigma, hurst)
    
    # Check that initial value is mu
    assert jnp.allclose(path.path[0, :], mu, atol=1e-6)

