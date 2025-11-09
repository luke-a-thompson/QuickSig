import jax
import jax.numpy as jnp
import pytest
from quicksig.control_lifts.log_signature import compute_log_signature
from quicksig.analytics.signature_sizes import (
    get_log_signature_dim,
    get_signature_dim,
)
import signax
from typing import Literal

_test_key = jax.random.PRNGKey(42)


@pytest.mark.parametrize("scalar_path_fixture", [(1, 10), (2, 10)], indirect=True)
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("log_signature_type", ["Tensor words", "Lyndon words"])
def test_log_signature_shape_full(
    scalar_path_fixture: jax.Array,
    depth: int,
    log_signature_type: Literal["Tensor words", "Lyndon words"],
) -> None:
    """Log signature tensor dimension matches algebraic formula."""
    path = scalar_path_fixture
    channels = path.shape[1]
    log_sig = compute_log_signature(
        path,
        depth=depth,
        log_signature_type=log_signature_type,
        mode="full",
    )
    log_sig_array = jnp.concatenate([x.flatten() for x in log_sig.signature])

    if log_signature_type == "Tensor words":
        expected_dim = get_signature_dim(depth, channels)
    else:  # Lyndon words
        expected_dim = get_log_signature_dim(depth, channels)

    expected_shape = (expected_dim,)
    assert log_sig_array.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {log_sig_array.shape}"
    )


@pytest.mark.parametrize("scalar_path_fixture", [(1, 10), (2, 10)], indirect=True)
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("log_signature_type", ["Tensor words", "Lyndon words"])
def test_log_signature_shape_stream(
    scalar_path_fixture: jax.Array,
    depth: int,
    log_signature_type: Literal["Tensor words", "Lyndon words"],
) -> None:
    """Log signature tensor dimension matches algebraic formula."""
    path = scalar_path_fixture
    num_steps, channels = path.shape
    log_sigs = compute_log_signature(
        path,
        depth=depth,
        log_signature_type=log_signature_type,
        mode="stream",
    )

    assert len(log_sigs) == num_steps - 1
    log_sig_array = jnp.stack(
        [jnp.concatenate([x.flatten() for x in l.signature]) for l in log_sigs]
    )

    if log_signature_type == "Tensor words":
        expected_dim = get_signature_dim(depth, channels)
    else:
        expected_dim = get_log_signature_dim(depth, channels)

    expected_shape = (num_steps - 1, expected_dim)
    assert log_sig_array.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {log_sig_array.shape}"
    )


@pytest.mark.parametrize("scalar_path_fixture", [(1, 10), (2, 10)], indirect=True)
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("log_signature_type", ["Tensor words", "Lyndon words"])
def test_log_signature_shape_incremental(
    scalar_path_fixture: jax.Array,
    depth: int,
    log_signature_type: Literal["Tensor words", "Lyndon words"],
) -> None:
    """Log signature tensor dimension matches algebraic formula."""
    path = scalar_path_fixture
    num_steps, channels = path.shape
    log_sigs = compute_log_signature(
        path,
        depth=depth,
        log_signature_type=log_signature_type,
        mode="incremental",
    )

    assert len(log_sigs) == num_steps - 1
    log_sig_array = jnp.stack(
        [jnp.concatenate([x.flatten() for x in l.signature]) for l in log_sigs]
    )

    if log_signature_type == "Tensor words":
        expected_dim = get_signature_dim(depth, channels)
    else:
        expected_dim = get_log_signature_dim(depth, channels)

    expected_shape = (num_steps - 1, expected_dim)
    assert log_sig_array.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {log_sig_array.shape}"
    )


# Signax does not support 1D paths
@pytest.mark.parametrize("scalar_path_fixture", [(2, 10), (3, 10)], indirect=True)
@pytest.mark.parametrize("depth", [2, 3])
def test_quicksig_signax_equivalence_full(scalar_path_fixture: jax.Array, depth: int) -> None:
    """
    Test that the log signature computed by QuickSig and Signax are equivalent.
    """
    path = scalar_path_fixture
    quicksig_log_sig = compute_log_signature(
        path, depth=depth, log_signature_type="Lyndon words", mode="full"
    )
    quicksig_log_sig = jnp.concatenate([x.flatten() for x in quicksig_log_sig.signature])

    signax_log_sig = signax.logsignature(path, depth=depth, stream=False)

    assert jnp.allclose(quicksig_log_sig, signax_log_sig, atol=1e-5, rtol=1e-5)


# Signax does not support 1D paths
@pytest.mark.parametrize("scalar_path_fixture", [(2, 10), (3, 10)], indirect=True)
@pytest.mark.parametrize("depth", [2, 3])
def test_quicksig_signax_equivalence_stream(scalar_path_fixture: jax.Array, depth: int) -> None:
    """
    Test that the log signature computed by QuickSig and Signax are equivalent.
    """
    path = scalar_path_fixture
    quicksig_log_sigs = compute_log_signature(
        path, depth=depth, log_signature_type="Lyndon words", mode="stream"
    )
    quicksig_log_sigs = jnp.stack(
        [jnp.concatenate([x.flatten() for x in l.signature]) for l in quicksig_log_sigs]
    )

    signax_log_sigs = signax.logsignature(path, depth=depth, stream=True)

    assert jnp.allclose(quicksig_log_sigs, signax_log_sigs, atol=1e-5, rtol=1e-5)
