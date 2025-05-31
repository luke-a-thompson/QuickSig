import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike
import math


def generate_scalar_path(key: jax.Array, num_timesteps: int = 100, mu: float = 0.5, sigma: float = 0.3, n_features: int = 1) -> tuple[jax.Array, jax.Array]:
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
    timestamps = jnp.linspace(0, 1, num_timesteps, dtype=dtype)
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

    return timestamps, values


def signature_dim(channels: int, depth: int) -> int:
    # Σ_{k=1..depth} channels^k  — works for channels=1,2,3,...
    return sum(channels**k for k in range(1, depth + 1))


def _linear_path(start: float, stop: float, num_steps: int, channels: int) -> ArrayLike:
    """Deterministic straight-line path for ground-truth tests."""
    t = jnp.linspace(start, stop, num_steps).reshape(-1, 1)  # (steps, 1)
    vals = jnp.repeat(t, channels, axis=1)  # (steps, channels)
    return vals[None, :, :]  # add batch dim


def get_prime_factorization(n: int) -> dict[int, int]:
    factors: dict[int, int] = {}
    d = 2
    temp_n = n
    while d * d <= temp_n:
        while temp_n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            temp_n //= d
        d += 1
    if temp_n > 1:
        factors[temp_n] = factors.get(temp_n, 0) + 1
    return factors


def mobius_mu(n: int) -> int:
    if n == 1:
        return 1
    prime_factors = get_prime_factorization(n)
    for p in prime_factors:
        if prime_factors[p] > 1:
            return 0  # Has a squared prime factor
    return (-1) ** len(prime_factors)  # Product of k distinct primes


def get_divisors(n: int) -> list[int]:
    divs = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(list(divs))


def num_lyndon_words_of_length_k(num_symbols: int, length: int) -> int:
    if length == 0:
        return 0
    if num_symbols == 1:  # Alphabet has only one symbol e.g. "a"
        return 1 if length == 1 else 0  # Only "a" is a Lyndon word, "aa", "aaa" are not.

    divs = get_divisors(length)
    total_sum = 0
    for d in divs:
        total_sum += mobius_mu(length // d) * (num_symbols**d)
    return total_sum // length


def lyndon_words_dim(channels: int, depth: int) -> int:
    return sum(num_lyndon_words_of_length_k(channels, k) for k in range(1, depth + 1))
