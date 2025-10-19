import jax
import jax.numpy as jnp
import pytest
from quicksig.rde_bench.ou_process import ou_process
from quicksig.rde.rde_types import Path


@pytest.fixture(scope="module")
def ou_samples() -> tuple[Path, dict]:
    """Generate and cache multiple OU process paths for reuse across tests."""
    seed = 42
    timesteps = 1000
    dim = 1
    num_paths = 1000
    
    # OU parameters
    theta = 0.5
    mu = 0.0
    sigma = 0.3
    x0 = 1.0
    
    key = jax.random.key(seed)
    keys = jax.random.split(key, num_paths)
    
    vmap_ou = jax.vmap(
        lambda k: ou_process(k, timesteps=timesteps, dim=dim, theta=theta, mu=mu, sigma=sigma, x0=x0)
    )
    paths = vmap_ou(keys)
    
    params = {
        "theta": theta,
        "mu": mu,
        "sigma": sigma,
        "x0": x0,
        "timesteps": timesteps,
    }
    
    return paths, params


def test_ou_analytical_expectation(ou_samples: tuple[Path, dict]) -> None:
    r"""
    Test that the OU process matches the analytical expectation.
    
    For the Ornstein-Uhlenbeck process dX_t = θ(μ - X_t)dt + σdW_t,
    the analytical solution for the expected value is:
    \[
        \E[X_t] = μ + (X_0 - μ)e^{-θt}
    \]
    We test this at multiple time points along the path.
    """
    paths, params = ou_samples
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


def test_ou_path_structure() -> None:
    r"""
    Test that the OU process returns a Path object with correct structure.
    
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
    x0 = 1.5
    
    path = ou_process(key, timesteps, dim, theta, mu, sigma, x0)
    
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


@pytest.mark.parametrize("theta", [0.5, 2.0])
@pytest.mark.parametrize("mu", [-1.0, 0.0, 2.0])
def test_ou_stationary_variance(theta: float, mu: float) -> None:
    r"""
    Test that the OU process converges to its stationary distribution.
    
    For the OU process dX_t = θ(μ - X_t)dt + σdW_t starting from X_0,
    as t → ∞, the stationary variance is:
    \[
        \Var(X_∞) = \frac{σ^2}{2θ}
    \]
    We test this at a late time point with many paths.
    """
    seed = 456
    timesteps = 2000
    dim = 1
    num_paths = 2000
    sigma = 0.4
    x0 = 0.0
    
    key = jax.random.key(seed)
    keys = jax.random.split(key, num_paths)
    
    vmap_ou = jax.vmap(
        lambda k: ou_process(k, timesteps=timesteps, dim=dim, theta=theta, mu=mu, sigma=sigma, x0=x0)
    )
    paths = vmap_ou(keys)
    
    # At late time (t=1.0), the process should be close to stationary
    final_values = paths.path[:, -1, 0]
    
    # Empirical variance
    empirical_var = jnp.var(final_values, ddof=1)
    
    # Theoretical stationary variance: σ²/(2θ)
    theoretical_var = (sigma**2) / (2.0 * theta)
    
    # Also check that mean is close to μ
    empirical_mean = jnp.mean(final_values)
    
    # For t=1.0, the process should be reasonably close to stationary
    # The relaxation timescale is 1/θ, so we use a tolerance that depends on θ
    relaxation_factor = jnp.exp(-2.0 * theta * 1.0)  # e^(-2θt) is variance decay
    
    # If well-equilibrated (relaxation_factor small), variance should match closely
    if relaxation_factor < 0.1:
        assert jnp.isclose(empirical_var, theoretical_var, rtol=0.15), (
            f"Empirical variance {float(empirical_var):.4f} "
            f"vs theoretical {float(theoretical_var):.4f}"
        )
    
    # Mean should always be close to theoretical expectation
    t = 1.0
    theoretical_mean = mu + (x0 - mu) * jnp.exp(-theta * t)
    se = jnp.sqrt(empirical_var / num_paths)
    assert jnp.abs(empirical_mean - theoretical_mean) <= 3.0 * se, (
        f"Empirical mean {float(empirical_mean):.4f} "
        f"vs theoretical {float(theoretical_mean):.4f}"
    )

