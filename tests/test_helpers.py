import jax
import jax.numpy as jnp
from jax import lax
import pytest

_test_key = jax.random.PRNGKey(42)


@pytest.fixture
def scalar_path_fixture(request: pytest.FixtureRequest) -> jax.Array:
    """Generates a path with a given number of channels and timesteps."""
    n_features, num_timesteps = request.param
    key, subkey = jax.random.split(_test_key)
    return generate_scalar_path(subkey, n_features, num_timesteps)


@pytest.fixture
def linear_path_fixture(request: pytest.FixtureRequest) -> jax.Array:
    """Generates a linear path with a given number of channels and timesteps."""
    n_features, num_timesteps = request.param
    return generate_linear_path(n_features, num_timesteps)


@pytest.mark.parametrize("n_features", [1, 2, 3])
@pytest.mark.parametrize("num_timesteps", [10, 100])
def test_gbm_shape_and_positive(num_timesteps: int, n_features: int) -> None:
    """GBM path has expected shape and stays strictly positive."""
    global _test_key
    _test_key, subkey = jax.random.split(_test_key)

    path = generate_scalar_path(subkey, n_features, num_timesteps)
    assert path.shape == (num_timesteps, n_features)
    assert jnp.all(path > 0.0), "GBM should remain positive"


def generate_scalar_path(key: jax.Array, n_features: int, num_timesteps: int, mu: float = 0.5, sigma: float = 0.3) -> jax.Array:
    """
    Generate a multi-dimensional path following geometric Brownian motion (GBM).
    Uses JAX for random number generation and path computation.

    Args:
        key: JAX PRNGKey for random number generation.
        num_timesteps: Number of timesteps in the path.
        mu: Drift coefficient (default: 0.5).
        sigma: Volatility coefficient (default: 0.3).
        n_features: Number of dimensions/features in the path (default: 1).

    Returns:
        Tuple containing:
        - timestamps: JAX Array of timestamps, shape (num_timesteps,)
        - values: JAX Array of path values following GBM, shape (num_timesteps, n_features)
    """
    dtype = jnp.float32
    dt = jnp.array(1.0 / (num_timesteps - 1), dtype=dtype)
    std_normal_increments = jax.random.normal(key, (num_timesteps - 1, n_features), dtype=dtype)

    # Scale increments to be N(0, sqrt(dt)) for the GBM formula
    dW_increments = std_normal_increments * jnp.sqrt(dt)  # Shape: (num_timesteps - 1, n_features)

    initial_path_value = jnp.ones(n_features, dtype=dtype)  # S_0, shape (n_features,)

    def gbm_euler_step(s_prev: jax.Array, dW_i: jax.Array) -> tuple[jax.Array, jax.Array]:
        # s_prev: path value at t-1, shape (n_features,)
        # dW_i: Brownian increment N(0,sqrt(dt)) for the step, shape (n_features,)
        s_next = s_prev * (1 + mu * dt + sigma * dW_i)
        return s_next, s_next  # (new_carry_state, value_to_scan_out)

    # Perform the scan over the time steps using dW_increments
    # initial_carry is S_0
    _, path_values_from_t1 = lax.scan(gbm_euler_step, initial_path_value, dW_increments)
    # path_values_from_t1 contains S_1, S_2, ..., S_{T-1} if num_timesteps-1 increments
    # Shape: (num_timesteps - 1, n_features)

    # Prepend S_0 to the path
    s0_reshaped = initial_path_value.reshape(1, n_features)  # Shape: (1, n_features)
    values = jnp.concatenate([s0_reshaped, path_values_from_t1], axis=0)
    # values shape: (num_timesteps, n_features)

    return values


def generate_linear_path(n_features: int, start: float = 0.0, stop: float = 1.0, num_timesteps: int = 100) -> jax.Array:
    """Deterministic straight-line path for ground-truth tests."""
    t = jnp.linspace(start, stop, num_timesteps).reshape(-1, 1)  # (steps, 1)
    vals = jnp.repeat(t, n_features, axis=1)  # (steps, n_features)  # (steps, n_features)
    return vals
