import jax
import jax.numpy as jnp
import numpy as np
from quicksig import get_signature, get_log_signature
from quicksig.utils import compute_signature_dim, compute_log_signature_dim
import pytest
from tests.test_helpers import generate_scalar_path, _linear_path
import signax

# Rename TEST_KEY to _test_key
_test_key = jax.random.PRNGKey(42)


@pytest.mark.parametrize("n_features", [1, 2, 3])
@pytest.mark.parametrize("num_steps", [10, 100])
def test_gbm_shape_and_positive(num_steps: int, n_features: int) -> None:
    """GBM path has expected shape and stays strictly positive."""
    global _test_key
    _test_key, subkey = jax.random.split(_test_key)

    path = generate_scalar_path(subkey, num_timesteps=num_steps, n_features=n_features)
    assert path.shape == (num_steps, n_features)
    assert jnp.all(path > 0.0), "GBM should remain positive"


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_signature_shape(channels: int, depth: int) -> None:
    """Signature tensor dimension matches algebraic formula."""
    global _test_key
    _test_key, subkey = jax.random.split(_test_key)

    path = generate_scalar_path(subkey, n_features=channels)
    sig = get_signature(path, depth=depth)
    expected_dim = compute_signature_dim(depth, channels)
    assert sig.shape == (expected_dim,), f"Expected shape {expected_dim}, got {sig.shape}"


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("log_signature_type", ["expanded", "lyndon"])
def test_log_signature_shape(depth: int, channels: int, log_signature_type: str) -> None:
    """Log signature tensor dimension matches algebraic formula."""
    global _test_key
    _test_key, subkey = jax.random.split(_test_key)

    path = generate_scalar_path(subkey, n_features=channels)
    log_sig = get_log_signature(path, depth=depth, log_signature_type=log_signature_type)

    # For expanded type, dimension is same as signature
    # For lyndon type, dimension is number of Lyndon words
    if log_signature_type == "expanded":
        expected_dim = compute_signature_dim(depth, channels)
    else:  # lyndon
        # Number of Lyndon words at each level
        # expected_dim = sum(channels * (channels**k - 1) // (channels - 1) if channels > 1 else 1 for k in range(1, depth + 1))
        expected_dim = compute_log_signature_dim(depth, channels)

    assert log_sig.shape == (expected_dim,), f"Expected shape {expected_dim}, got {log_sig.shape}"


def test_linear_path_exactness() -> None:
    r"""
    For a straight-line 1-D path $$x(t) = \alpha t$$ with total increment $$\Delta x$$,
    the level-k iterated integral equals $$(\Delta x)^k / k!$$.
    """
    delta_x = -0.08840573
    num_steps = 20
    depth = 3
    path = _linear_path(0.0, delta_x, num_steps=num_steps, channels=1)
    sig = get_signature(path, depth=depth)

    # Ground‑truth closed form
    expected = jnp.array([delta_x, delta_x**2 / 2.0, delta_x**3 / 6.0])

    # Convert JAX arrays to NumPy arrays for comparison with np.testing
    np.testing.assert_allclose(np.asarray(sig), np.asarray(expected), atol=1e-8, rtol=1e-8)


def test_zero_path_vanishes() -> None:
    """
    A constant path has zero increment, hence zero signature beyond level 0.
    """
    const_path = jnp.zeros((50, 2), dtype=jnp.float32)
    sig = get_signature(const_path, depth=4)
    assert jnp.allclose(sig, 0.0)


@pytest.mark.parametrize(
    "a, b",
    [
        (1.0, 1.0),
        (2.0, 1.0),
        (1.0, 3.0),
        (-1.0, 2.0),
    ],
)
def test_quadratic_path_signature(a: float, b: float) -> None:
    """
    Tests the signature of a 2D path x(t) = (a*t, b*t^2/2).
    The analytical signature is known and can be compared against.
    """
    T = 1.0
    num_steps = 1000
    depth = 2

    # Create the path x(t) = (a*t, b*t^2/2)
    t = jnp.linspace(0, T, num_steps)
    path_x = a * t
    path_y = b * t**2 / 2.0
    path = jnp.stack([path_x, path_y], axis=-1)

    # Compute the signature using the function
    sig = get_signature(path, depth=depth)

    # Analytical signature
    # S_1 = (a*T, b*T^2/2)
    # S_2 = (a^2*T^2/2, a*b*T^3/3, a*b*T^3/6, b^2*T^4/8)
    expected = jnp.array([a * T, b * T**2 / 2.0, a**2 * T**2 / 2.0, a * b * T**3 / 3.0, a * b * T**3 / 6.0, b**2 * T**4 / 8.0])

    np.testing.assert_allclose(np.asarray(sig), np.asarray(expected), atol=1e-5, rtol=1e-5)


def test_quicksig_signax_equivalence() -> None:
    """
    Test that the signature computed by QuickSig and Signax are equivalent.
    """
    path = generate_scalar_path(_test_key, n_features=2)
    quicksig_sig = get_signature(path, depth=2)
    signax_sig = signax.signature(path, depth=2)
    np.testing.assert_allclose(np.asarray(quicksig_sig), np.asarray(signax_sig), atol=1e-5, rtol=1e-5)
