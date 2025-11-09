import jax
import jax.numpy as jnp
import pytest
from jax.scipy.linalg import expm as jexpm

from quicksig.integrators.log_ode import log_ode
from quicksig.control_lifts.log_signature import duval_generator
from quicksig.control_lifts.log_signature import compute_log_signature
from quicksig.vector_field_lifts.lie_lift import form_lyndon_brackets


def _so3_generators() -> jax.Array:
    A1 = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    A2 = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    A3 = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    return jnp.stack([A1, A2, A3], axis=0)  # [3, 3, 3]


def test_logode_zero_control_identity() -> None:
    """With zero coefficients, log-ODE should return the same normalized state."""
    A: jax.Array = _so3_generators()
    depth: int = 2
    dim: int = A.shape[0]
    words_by_len: list[jax.Array] = duval_generator(depth, dim)
    bracket_basis: jax.Array = form_lyndon_brackets(A, depth)  # [L, 3, 3]

    # Zero coefficients per level matching words_by_len
    lie_coefficients_by_len: list[jax.Array] = [
        jnp.zeros((w.shape[0],), dtype=jnp.float32)
        if w.size != 0
        else jnp.zeros((0,), dtype=jnp.float32)
        for w in words_by_len
    ]

    y0: jax.Array = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    y_next: jax.Array = log_ode(bracket_basis, lie_coefficients_by_len, words_by_len, y0)

    # y0 already unit norm; expect exact equality within tolerance
    assert jnp.allclose(y_next, y0, rtol=1e-7)


def test_logode_linear_1d_matches_matrix_exponential() -> None:
    """In 1D with depth=1, log-ODE equals exp(Δ A) @ y0."""
    # Single generator in R^2: 90-degree rotation generator
    A0 = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.float32)
    A: jax.Array = A0[jnp.newaxis, ...]  # [1, 2, 2]
    depth: int = 1
    dim: int = 1
    words_by_len: list[jax.Array] = duval_generator(depth, dim)  # one word [[0]]
    bracket_basis: jax.Array = form_lyndon_brackets(A, depth)  # [1, 2, 2] == A0

    delta: float = 0.3
    lie_coefficients_by_len: list[jax.Array] = [jnp.array([delta], dtype=jnp.float32)]

    y0: jax.Array = jnp.array([1.0, 0.0], dtype=jnp.float32)
    y_logode: jax.Array = log_ode(bracket_basis, lie_coefficients_by_len, words_by_len, y0)

    expected: jax.Array = jexpm(delta * A0) @ y0
    expected = expected / jnp.linalg.norm(expected)

    assert jnp.allclose(y_logode, expected, rtol=1e-6)


@pytest.mark.parametrize("brownian_path_fixture", [(1, 200)], indirect=True)
def test_logode_brownian_segmentation_invariance(brownian_path_fixture: jax.Array) -> None:
    """Sequential windowed application equals whole-interval application for commuting case (dim=1)."""
    W: jax.Array = brownian_path_fixture  # shape (N+1, 1)
    depth: int = 1  # depth=1 and dim=1 => commuting flows so product of exponentials equals single exponential
    dim: int = 1
    # Single 2x2 skew-symmetric generator
    A0 = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.float32)
    A: jax.Array = A0[jnp.newaxis, ...]  # [1, 2, 2]
    words_by_len: list[jax.Array] = duval_generator(depth, dim)
    bracket_basis: jax.Array = form_lyndon_brackets(A, depth)

    y0: jax.Array = jnp.array([1.0, 0.0], dtype=jnp.float32)

    # Whole interval
    log_sig_full = compute_log_signature(W, depth, "Lyndon words", mode="full")
    y_full: jax.Array = log_ode(bracket_basis, log_sig_full.signature, words_by_len, y0)

    # Windowed
    window: int = 10
    y_win: jax.Array = y0
    N: int = W.shape[0] - 1
    for s in range(0, N, window):
        e = min(s + window, N)
        seg: jax.Array = W[s : e + 1, :]
        log_sig_seg = compute_log_signature(seg, depth, "Lyndon words", mode="full")
        y_win = log_ode(bracket_basis, log_sig_seg.signature, words_by_len, y_win)

    assert jnp.allclose(y_full, y_win, rtol=1e-5)


@pytest.mark.parametrize("brownian_path_fixture", [(3, 300)], indirect=True)
def test_logode_brownian_segmentation_invariance_commuting_high_depth(
    brownian_path_fixture: jax.Array,
) -> None:
    """Higher depth and multi-dim path on Euclidean (commuting diagonal generators).

    Since generators commute, product of exponentials along windows equals
    the exponential of the sum, so whole-interval equals windowed trajectory.
    """
    W: jax.Array = brownian_path_fixture  # shape (N+1, 3)
    depth: int = 3
    dim: int = 3

    # Build commuting skew-symmetric generators acting on disjoint 2D planes in R^6
    # Each is an independent 2x2 rotation block; they commute and are norm-preserving.
    R2 = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.float32)
    zeros2 = jnp.zeros((2, 2), dtype=jnp.float32)
    # A1 acts on coordinates (0,1), A2 on (2,3), A3 on (4,5)
    A1 = jnp.block([[R2, zeros2, zeros2], [zeros2, zeros2, zeros2], [zeros2, zeros2, zeros2]])
    A2 = jnp.block([[zeros2, zeros2, zeros2], [zeros2, R2, zeros2], [zeros2, zeros2, zeros2]])
    A3 = jnp.block([[zeros2, zeros2, zeros2], [zeros2, zeros2, zeros2], [zeros2, zeros2, R2]])
    A: jax.Array = jnp.stack([A1, A2, A3], axis=0)  # [3, 6, 6]

    words_by_len: list[jax.Array] = duval_generator(depth, dim)
    bracket_basis: jax.Array = form_lyndon_brackets(A, depth)

    y0: jax.Array = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=jnp.float32)
    y0 = y0 / jnp.linalg.norm(y0)

    # Whole interval
    log_sig_full = compute_log_signature(W, depth, "Lyndon words", mode="full")
    y_full: jax.Array = log_ode(bracket_basis, log_sig_full.signature, words_by_len, y0)

    # Windowed
    window: int = 25
    y_win: jax.Array = y0
    N: int = W.shape[0] - 1
    for s in range(0, N, window):
        e = min(s + window, N)
        seg: jax.Array = W[s : e + 1, :]
        log_sig_seg = compute_log_signature(seg, depth, "Lyndon words", mode="full")
        y_win = log_ode(bracket_basis, log_sig_seg.signature, words_by_len, y_win)

    assert jnp.allclose(y_full, y_win, rtol=1e-5)
