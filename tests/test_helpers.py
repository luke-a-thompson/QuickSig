import jax
import jax.numpy as jnp
from jax import lax


def generate_scalar_path(key: jax.Array, num_timesteps: int = 100, mu: float = 0.5, sigma: float = 0.3, n_features: int = 1) -> jax.Array:
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


def linear_path(start: float, stop: float, num_steps: int, channels: int) -> jax.Array:
    """Deterministic straight-line path for ground-truth tests."""
    t = jnp.linspace(start, stop, num_steps).reshape(-1, 1)  # (steps, 1)
    vals = jnp.repeat(t, channels, axis=1)  # (steps, channels)
    return vals
