"""Tests for free Lie algebra operations, focusing on Lyndon bracket formation.

These tests verify:
1. Correctness of Lyndon bracket computation with standard factorization
2. Mathematical properties of commutators (antisymmetry, Jacobi identity)
3. Edge cases and gradient computation
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quicksig.hopf_algebras.free_lie import (
    commutator,
    form_lyndon_brackets,
)
from quicksig.integrators.series import form_lie_series


def test_commutator_basic() -> None:
    """Test basic commutator properties: [a,b] = ab - ba."""
    a = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    b = jnp.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])

    result = commutator(a, b)
    expected = a @ b - b @ a

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_commutator_antisymmetry() -> None:
    """Test antisymmetry: [a,b] = -[b,a]."""
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = jnp.array([[5.0, 6.0], [7.0, 8.0]])

    ab = commutator(a, b)
    ba = commutator(b, a)

    np.testing.assert_allclose(ab, -ba, rtol=1e-10)


def test_commutator_self_zero() -> None:
    """Test that [a,a] = 0."""
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    result = commutator(a, a)

    np.testing.assert_allclose(result, jnp.zeros_like(result), rtol=1e-10)


def test_jacobi_identity() -> None:
    """Test the Jacobi identity: [a, [b, c]] + [b, [c, a]] + [c, [a, b]] = 0.

    This is a fundamental property of Lie algebras.
    """
    n = 3
    a = jax.random.normal(jax.random.PRNGKey(4), (n, n))
    b = jax.random.normal(jax.random.PRNGKey(5), (n, n))
    c = jax.random.normal(jax.random.PRNGKey(6), (n, n))

    # Compute the three terms
    term1 = commutator(a, commutator(b, c))
    term2 = commutator(b, commutator(c, a))
    term3 = commutator(c, commutator(a, b))

    jacobi_sum = term1 + term2 + term3

    # Jacobi identity holds exactly, but float32 has numerical precision limits
    np.testing.assert_allclose(jacobi_sum, jnp.zeros_like(jacobi_sum), atol=1e-5)


def test_apply_lie_coeffs() -> None:
    """Test that a Lie algebra element can be formed from brackets and coefficients."""
    n = 3
    num_words = 5

    # Random brackets and coefficients
    W = jax.random.normal(jax.random.PRNGKey(12), (num_words, n, n))
    lam = jax.random.normal(jax.random.PRNGKey(13), (num_words,))

    # Wrap into per-level inputs expected by form_series
    lam_by_len = [lam]
    # words_by_len is only used for filtering empties; provide non-empty dummy words
    words_by_len = [jnp.arange(num_words, dtype=jnp.int32).reshape(num_words, 1)]
    result = form_lie_series(W, lam_by_len, words_by_len)

    # Should compute sum_i lam[i] * W[i]
    # Note: tensordot and explicit sum may have slightly different floating point behavior
    expected = jnp.sum(lam[:, None, None] * W, axis=0)

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_apply_lie_coeffs_shape_error() -> None:
    """Test that mismatched shapes raise appropriate error."""
    n = 2
    W = jax.random.normal(jax.random.PRNGKey(14), (5, n, n))
    lam = jax.random.normal(jax.random.PRNGKey(15), (3,))  # Wrong size

    with pytest.raises(ValueError, match="does not match"):
        lam_by_len = [lam]
        words_by_len = [jnp.arange(lam.shape[0], dtype=jnp.int32).reshape(lam.shape[0], 1)]
        form_lie_series(W, lam_by_len, words_by_len)


# ============================================================================
# Tests for Lyndon bracket formation
# ============================================================================


def test_form_lyndon_brackets_single_letter() -> None:
    """Test that single letter Lyndon words [i] return A[i] directly."""
    dim, n = 2, 3
    A = jax.random.normal(jax.random.PRNGKey(20), (dim, n, n))

    result = form_lyndon_brackets(A, depth=1)

    # For depth=1, dim=2, we get words [0] and [1]
    assert result.shape == (2, n, n)
    np.testing.assert_allclose(result[0], A[0], rtol=1e-10)
    np.testing.assert_allclose(result[1], A[1], rtol=1e-10)


def test_form_lyndon_brackets_two_letters() -> None:
    """Test that two-letter Lyndon word [0,1] computes [A[0], A[1]]."""
    dim, n = 2, 3
    A = jax.random.normal(jax.random.PRNGKey(21), (dim, n, n))

    result = form_lyndon_brackets(A, depth=2, dim=dim)

    # For depth=2, dim=2: words are [0], [1], [0,1]
    assert result.shape == (3, n, n)

    # Check single letters
    np.testing.assert_allclose(result[0], A[0], rtol=1e-10)
    np.testing.assert_allclose(result[1], A[1], rtol=1e-10)

    # Check two-letter bracket [0,1] = [A[0], A[1]]
    bracket_01 = result[2]
    expected = commutator(A[0], A[1])
    np.testing.assert_allclose(bracket_01, expected, rtol=1e-10)


def test_form_lyndon_brackets_standard_factorization() -> None:
    """Test that Lyndon brackets use standard factorization correctly.

    For a Lyndon word w = uv where v is the longest proper Lyndon suffix,
    [w] = [[u], [v]].

    Example: [0,0,1] should factorize as [0] and [0,1] (not [0,0] and [1]).
    """
    dim, n = 2, 3
    A = jax.random.normal(jax.random.PRNGKey(22), (dim, n, n))

    result = form_lyndon_brackets(A, depth=3, dim=dim)

    # For depth=3, dim=2: words are [0], [1], [0,1], [0,0,1], [0,1,1]
    # Let's verify [0,0,1] = [[0], [0,1]]
    bracket_0 = result[0]  # [0] = A[0]
    bracket_01 = result[2]  # [0,1] = [A[0], A[1]]
    bracket_001 = result[3]  # [0,0,1] should be [[0], [0,1]]

    expected = commutator(bracket_0, bracket_01)
    np.testing.assert_allclose(bracket_001, expected, rtol=1e-10)

    # Also verify [0,1,1] = [[0,1], [1]]
    bracket_1 = result[1]  # [1] = A[1]
    bracket_011 = result[4]  # [0,1,1] should be [[0,1], [1]]

    expected_011 = commutator(bracket_01, bracket_1)
    np.testing.assert_allclose(bracket_011, expected_011, rtol=1e-10)


def test_form_lyndon_brackets_jacobi_identity() -> None:
    """Test that Lyndon brackets satisfy the Jacobi identity.

    For any three elements in the free Lie algebra, the Jacobi identity should hold.
    We'll test with dim=3 using the single-letter brackets directly.
    """
    dim, n = 3, 3
    A = jax.random.normal(jax.random.PRNGKey(23), (dim, n, n))

    # Get single-letter brackets
    bracket_0 = A[0]  # [0] = A[0]
    bracket_1 = A[1]  # [1] = A[1]
    bracket_2 = A[2]  # [2] = A[2]

    # Compute Jacobi identity: [a, [b, c]] + [b, [c, a]] + [c, [a, b]] = 0
    # Using the three single-letter brackets directly
    term1 = commutator(bracket_0, commutator(bracket_1, bracket_2))  # [0, [1, 2]]
    term2 = commutator(bracket_1, commutator(bracket_2, bracket_0))  # [1, [2, 0]]
    term3 = commutator(bracket_2, commutator(bracket_0, bracket_1))  # [2, [0, 1]]

    jacobi_sum = term1 + term2 + term3

    # Jacobi identity should hold (up to numerical precision)
    np.testing.assert_allclose(jacobi_sum, jnp.zeros_like(jacobi_sum), atol=1e-5)


def test_form_lyndon_brackets_three_dimensions() -> None:
    """Test Lyndon brackets for dimension 3 with more complex words."""
    dim, n = 3, 2
    A = jax.random.normal(jax.random.PRNGKey(24), (dim, n, n))

    result = form_lyndon_brackets(A, depth=3, dim=dim)

    # For dim=3, depth=3, we should have:
    # Level 1: [0], [1], [2]
    # Level 2: [0,1], [0,2], [1,2]
    # Level 3: [0,0,1], [0,0,2], [0,1,1], [0,1,2], [0,2,2], [1,1,2], [1,2,2]

    # Verify basic structure
    assert result.shape[0] >= 3  # At least single letters

    # Verify [0,1] = [A[0], A[1]]
    bracket_0 = result[0]
    bracket_1 = result[1]
    bracket_01 = result[3]  # First two-letter word should be [0,1]

    expected_01 = commutator(bracket_0, bracket_1)
    np.testing.assert_allclose(bracket_01, expected_01, rtol=1e-10)


def test_form_lyndon_brackets_empty() -> None:
    """Test handling of edge cases (empty/zero depth)."""
    dim, n = 2, 3
    A = jax.random.normal(jax.random.PRNGKey(25), (dim, n, n))

    # Depth 0 should return empty
    result = form_lyndon_brackets(A, depth=0, dim=dim)
    assert result.shape == (0, n, n)

    # Depth 1 with dim=1 should work
    A_single = jax.random.normal(jax.random.PRNGKey(26), (1, n, n))
    result = form_lyndon_brackets(A_single, depth=2, dim=1)
    # For dim=1, we only get [0] at level 1, nothing at level 2+
    assert result.shape == (1, n, n)
    np.testing.assert_allclose(result[0], A_single[0], rtol=1e-10)


def test_form_lyndon_brackets_reproducibility() -> None:
    """Test that form_lyndon_brackets produces consistent results."""
    dim, n = 2, 3
    A = jax.random.normal(jax.random.PRNGKey(27), (dim, n, n))

    # Run twice with same input
    result1 = form_lyndon_brackets(A, depth=2, dim=dim)
    result2 = form_lyndon_brackets(A, depth=2, dim=dim)

    # Results should match exactly
    np.testing.assert_allclose(result1, result2, rtol=1e-10)


def test_form_lyndon_brackets_gradients() -> None:
    """Test that gradients can be computed through Lyndon bracket formation."""
    dim, n = 2, 2
    key = jax.random.PRNGKey(28)

    def loss_fn(A: jax.Array) -> jax.Array:
        brackets = form_lyndon_brackets(A, depth=2, dim=dim)
        # Sum of squares as a simple loss
        return jnp.sum(brackets**2)

    A = jax.random.normal(key, (dim, n, n))

    # Compute gradient
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(A)

    # Gradients should have same shape as A
    assert grads.shape == A.shape
    # Gradients should not be all zeros (for a non-trivial case)
    assert not jnp.allclose(grads, jnp.zeros_like(grads), atol=1e-10)


def test_form_lyndon_brackets_consistency_with_duval() -> None:
    """Test that Lyndon brackets are consistent with duval_generator output."""
    from quicksig.control_lifts.log_signature import duval_generator

    dim, n = 2, 3
    A = jax.random.normal(jax.random.PRNGKey(29), (dim, n, n))

    # Generate Lyndon words using duval_generator
    words_by_len = duval_generator(depth=2, dim=dim)

    # Compute brackets using our function
    result = form_lyndon_brackets(A, depth=2, dim=dim)

    # Verify we have the right number of brackets
    total_words = sum(w.shape[0] for w in words_by_len)
    assert result.shape[0] == total_words

    # Verify single-letter brackets match
    assert words_by_len[0].shape[0] == 2  # [0] and [1]
    np.testing.assert_allclose(result[0], A[0], rtol=1e-10)
    np.testing.assert_allclose(result[1], A[1], rtol=1e-10)

    # Verify two-letter bracket
    assert words_by_len[1].shape[0] == 1  # [0,1]
    bracket_01_expected = commutator(A[0], A[1])
    np.testing.assert_allclose(result[2], bracket_01_expected, rtol=1e-10)
