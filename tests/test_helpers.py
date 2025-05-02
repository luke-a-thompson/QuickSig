import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp
from jax.typing import ArrayLike


def generate_scalar_path(num_timesteps: int = 100, mu: float = 0.5, sigma: float = 0.3, n_features: int = 1) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Generate a multi-dimensional path following geometric Brownian motion (GBM).

    Args:
        num_steps: Number of timesteps in the path
        mu: Drift coefficient (default: 0.5)
        sigma: Volatility coefficient (default: 0.3)
        n_features: Number of dimensions/features in the path (default: 1)

    Returns:
        Tuple containing:
        - timestamps: Array of timestamps
        - values: Array of path values following GBM, shape (num_steps, n_features)
    """
    # Generate timestamps
    timestamps = np.linspace(0, 1, num_timesteps)
    dt = 1.0 / (num_timesteps - 1)

    # Generate Brownian increments for each feature
    dW = np.random.normal(0, np.sqrt(dt), (num_timesteps - 1, n_features))

    # Initialize path with starting value of 1 for each feature
    values = np.zeros((num_timesteps, n_features))
    values[0] = 1.0

    # Generate GBM path using Euler-Maruyama scheme for each feature
    for i in range(1, num_timesteps):
        values[i] = values[i - 1] * (1 + mu * dt + sigma * dW[i - 1])

    return timestamps, values


def signature_dim(channels: int, depth: int) -> int:
    # Σ_{k=1..depth} channels^k  — works for channels=1,2,3,...
    return sum(channels**k for k in range(1, depth + 1))


def _linear_path(start: float, stop: float, num_steps: int, channels: int) -> ArrayLike:
    """Deterministic straight-line path for ground-truth tests."""
    t = jnp.linspace(start, stop, num_steps).reshape(-1, 1)  # (steps, 1)
    vals = jnp.repeat(t, channels, axis=1)  # (steps, channels)
    return vals[None, :, :]  # add batch dim
