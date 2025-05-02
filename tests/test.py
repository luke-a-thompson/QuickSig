import numpy as np
import jax.numpy as jnp
from quicksig.path_signature import batch_signature_pure_jax
import pytest
from tests.test_helpers import generate_scalar_path, signature_dim, _linear_path


@pytest.mark.parametrize("n_features", [1, 2, 3])
@pytest.mark.parametrize("num_steps", [10, 100])
def test_gbm_shape_and_positive(num_steps: int, n_features: int) -> None:
    """GBM path has expected shape and stays strictly positive."""
    ts, vals = generate_scalar_path(num_timesteps=num_steps, n_features=n_features)
    assert ts.shape == (num_steps,)
    assert vals.shape == (num_steps, n_features)
    assert np.all(vals > 0.0), "GBM should remain positive"


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_signature_shape(channels: int, depth: int) -> None:
    """Signature tensor dimension matches algebraic formula."""
    _, vals = generate_scalar_path(n_features=channels)
    path = jnp.asarray(vals)[None, :, :]  # (1, seq, channels)
    sig = batch_signature_pure_jax(path, depth=depth)
    expected_dim = signature_dim(channels, depth)
    assert sig.shape == (1, expected_dim)


def test_linear_path_exactness() -> None:
    """
    For a straight-line 1-D path x(t)=αt with total increment Δx,
    the level-k iterated integral equals (Δx)^k / k!.
    """
    Δx = -0.08840573
    num_steps = 20
    depth = 3
    path = _linear_path(0.0, Δx, num_steps=num_steps, channels=1)
    sig = np.asarray(batch_signature_pure_jax(path, depth=depth))[0]

    # Ground‑truth closed form
    expected = np.array([Δx, Δx**2 / 2.0, Δx**3 / 6.0], dtype=np.float32)
    np.testing.assert_allclose(sig, expected, atol=1e-6, rtol=1e-6)


def test_zero_path_vanishes() -> None:
    """
    A constant path has zero increment, hence zero signature beyond level 0.
    """
    const_path = jnp.zeros((1, 50, 2), dtype=jnp.float32)
    sig = batch_signature_pure_jax(const_path, depth=4)
    assert jnp.allclose(sig, 0.0)
