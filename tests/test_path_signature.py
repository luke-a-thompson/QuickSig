import jax
import jax.numpy as jnp
import numpy as np
from quicksig import get_signature, get_log_signature
from quicksig.log_signature import LogSignatureType
import pytest
from tests.test_helpers import generate_scalar_path, signature_dim, _linear_path, lyndon_words_dim

# Rename TEST_KEY to _test_key
_test_key = jax.random.PRNGKey(42)


@pytest.mark.parametrize("n_features", [1, 2, 3])
@pytest.mark.parametrize("num_steps", [10, 100])
def test_gbm_shape_and_positive(num_steps: int, n_features: int) -> None:
    """GBM path has expected shape and stays strictly positive."""
    global _test_key
    _test_key, subkey = jax.random.split(_test_key)

    ts, vals = generate_scalar_path(subkey, num_timesteps=num_steps, n_features=n_features)
    assert ts.shape == (num_steps,)
    assert vals.shape == (num_steps, n_features)
    assert jnp.all(vals > 0.0), "GBM should remain positive"


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_signature_shape(channels: int, depth: int) -> None:
    """Signature tensor dimension matches algebraic formula."""
    global _test_key
    _test_key, subkey = jax.random.split(_test_key)

    _, vals = generate_scalar_path(subkey, n_features=channels)
    path = vals[None, :, :]
    sig = get_signature(path, depth=depth)
    expected_dim = signature_dim(channels, depth)
    assert sig.shape == (1, expected_dim)


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("log_signature_type", ["expanded", "lyndon"])
def test_log_signature_shape(channels: int, depth: int, log_signature_type: str) -> None:
    """Log signature tensor dimension matches algebraic formula."""
    global _test_key
    _test_key, subkey = jax.random.split(_test_key)

    _, vals = generate_scalar_path(subkey, n_features=channels)
    path = vals[None, :, :]  # vals is already JAX array
    log_sig = get_log_signature(path, depth=depth, log_signature_type=LogSignatureType(log_signature_type))

    # For expanded type, dimension is same as signature
    # For lyndon type, dimension is number of Lyndon words
    if log_signature_type == "expanded":
        expected_dim = signature_dim(channels, depth)
    else:  # lyndon
        # Number of Lyndon words at each level
        # expected_dim = sum(channels * (channels**k - 1) // (channels - 1) if channels > 1 else 1 for k in range(1, depth + 1))
        expected_dim = lyndon_words_dim(channels, depth)

    assert log_sig.shape == (1, expected_dim), f"Expected shape (1, {expected_dim}), got {log_sig.shape}"


def test_linear_path_exactness() -> None:
    """
    For a straight-line 1-D path x(t)=αt with total increment Δx,
    the level-k iterated integral equals (Δx)^k / k!.
    """
    delta_x = -0.08840573
    num_steps = 20
    depth = 3
    path = _linear_path(0.0, delta_x, num_steps=num_steps, channels=1)
    sig = get_signature(path, depth=depth)[0]

    # Ground‑truth closed form
    expected = jnp.array([delta_x, delta_x**2 / 2.0, delta_x**3 / 6.0], dtype=jnp.float32)

    # Convert JAX arrays to NumPy arrays for comparison with np.testing
    np.testing.assert_allclose(np.asarray(sig), np.asarray(expected), atol=1e-6, rtol=1e-6)


def test_zero_path_vanishes() -> None:
    """
    A constant path has zero increment, hence zero signature beyond level 0.
    """
    const_path = jnp.zeros((1, 50, 2), dtype=jnp.float32)
    sig = get_signature(const_path, depth=4)
    assert jnp.allclose(sig, 0.0)
