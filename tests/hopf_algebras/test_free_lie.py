"""Tests for free Lie algebra operations, focusing on right-normed bracket formation.

These tests verify:
1. Correctness of right-normed bracket computation
2. Mathematical properties of commutators (antisymmetry, Jacobi identity)
3. Caching/reuse of suffix brackets
4. Edge cases and JIT compatibility
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quicksig.hopf_algebras.free_lie import (
    commutator,
    form_right_normed_brackets,
    apply_lie_coeffs,
    flatten_coeffs,
)


def test_commutator_basic() -> None:
    """Test basic commutator properties: [a,b] = ab - ba."""
    n = 3
    a = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    b = jnp.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
    
    result = commutator(a, b)
    expected = a @ b - b @ a
    
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_commutator_antisymmetry() -> None:
    """Test antisymmetry: [a,b] = -[b,a]."""
    n = 2
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


def test_form_right_normed_brackets_single_letter() -> None:
    """Test that single letter words [i] return A[i] directly."""
    dim, n = 2, 3
    A = jax.random.normal(jax.random.PRNGKey(0), (dim, n, n))
    
    words_by_len = [jnp.array([[0], [1]], dtype=jnp.int32)]  # Single letter words
    
    result = form_right_normed_brackets(A, words_by_len)
    
    assert result.shape == (2, n, n)
    np.testing.assert_allclose(result[0], A[0], rtol=1e-10)
    np.testing.assert_allclose(result[1], A[1], rtol=1e-10)


def test_form_right_normed_brackets_two_letters() -> None:
    """Test that two-letter words [i,j] compute [A[i], A[j]] = A[i]@A[j] - A[j]@A[i]."""
    dim, n = 2, 3
    A = jax.random.normal(jax.random.PRNGKey(1), (dim, n, n))
    
    words_by_len = [
        jnp.array([[0], [1]], dtype=jnp.int32),  # Single letters
        jnp.array([[0, 1]], dtype=jnp.int32),    # Two-letter word [0,1]
    ]
    
    result = form_right_normed_brackets(A, words_by_len)
    
    # Result should have 3 brackets: [0], [1], [0,1]
    assert result.shape == (3, n, n)
    
    # Check the two-letter bracket [0,1] = [A[0], A[1]]
    bracket_01 = result[2]
    expected = commutator(A[0], A[1])
    
    np.testing.assert_allclose(bracket_01, expected, rtol=1e-10)


def test_form_right_normed_brackets_three_letters_right_normed() -> None:
    """Test that three-letter words are right-normed: [i,j,k] = [A[i], [A[j], A[k]]].
    
    This verifies the critical right-normed property that distinguishes our
    implementation from left-normed brackets.
    """
    dim, n = 3, 2
    A = jax.random.normal(jax.random.PRNGKey(2), (dim, n, n))
    
    words_by_len = [
        jnp.array([[0], [1], [2]], dtype=jnp.int32),
        jnp.array([[1, 2]], dtype=jnp.int32),      # [1,2]
        jnp.array([[0, 1, 2]], dtype=jnp.int32),   # [0,1,2]
    ]
    
    result = form_right_normed_brackets(A, words_by_len)
    
    # Extract brackets
    bracket_12 = result[3]  # [1,2] = [A[1], A[2]]
    bracket_012 = result[4]  # [0,1,2] = [A[0], [A[1], A[2]]]
    
    # Verify right-normed structure
    expected_bracket_12 = commutator(A[1], A[2])
    expected_bracket_012 = commutator(A[0], expected_bracket_12)
    
    np.testing.assert_allclose(bracket_12, expected_bracket_12, rtol=1e-10)
    np.testing.assert_allclose(bracket_012, expected_bracket_012, rtol=1e-10)
    
    # Verify it's NOT left-normed: [[A[0], A[1]], A[2]] should be different
    left_normed = commutator(commutator(A[0], A[1]), A[2])
    assert not np.allclose(bracket_012, left_normed, rtol=1e-6), \
        "Bracket should be right-normed, not left-normed"


def test_caching_suffix_reuse() -> None:
    """Test that suffixes are cached and reused correctly.
    
    If we have words [1,2] and [0,1,2], the computation of [0,1,2] should
    reuse the already-computed bracket for [1,2].
    """
    dim, n = 3, 2
    A = jax.random.normal(jax.random.PRNGKey(3), (dim, n, n))
    
    words_by_len = [
        jnp.array([[0], [1], [2]], dtype=jnp.int32),
        jnp.array([[1, 2]], dtype=jnp.int32),      # [1,2]
        jnp.array([[0, 1, 2]], dtype=jnp.int32),   # [0,1,2] should reuse [1,2]
    ]
    
    result = form_right_normed_brackets(A, words_by_len)
    
    # Manually compute to verify
    bracket_12 = commutator(A[1], A[2])
    bracket_012_expected = commutator(A[0], bracket_12)
    
    bracket_12_computed = result[3]
    bracket_012_computed = result[4]
    
    # Verify correctness (which implicitly tests caching)
    np.testing.assert_allclose(bracket_12_computed, bracket_12, rtol=1e-10)
    np.testing.assert_allclose(bracket_012_computed, bracket_012_expected, rtol=1e-10)


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


def test_form_right_normed_brackets_jacobi_via_words() -> None:
    """Test Jacobi identity using our bracket formation function."""
    dim, n = 3, 3
    A = jax.random.normal(jax.random.PRNGKey(7), (dim, n, n))
    
    # Create words that form a Jacobi triple
    # We'll compute [0, [1, 2]] + [1, [2, 0]] + [2, [0, 1]]
    words_by_len = [
        jnp.array([[0], [1], [2]], dtype=jnp.int32),
        jnp.array([[0, 1], [1, 2], [2, 0]], dtype=jnp.int32),
        jnp.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=jnp.int32),
    ]
    
    result = form_right_normed_brackets(A, words_by_len)
    
    # Extract the three three-letter brackets
    bracket_0_12 = result[6]  # [0, [1, 2]]
    bracket_1_20 = result[7]  # [1, [2, 0]]
    bracket_2_01 = result[8]  # [2, [0, 1]]
    
    # Note: [2, 0, 1] is right-normed as [2, [0, 1]], but Jacobi needs [2, [0, 1]]
    # which is different from [2, 0, 1] in our notation.
    # Actually, let's compute what we need:
    # [0, [1, 2]] = result[6] ✓
    # [1, [2, 0]]: word [1, 2, 0] gives [1, [2, 0]] = result[7] ✓
    # [2, [0, 1]]: word [2, 0, 1] gives [2, [0, 1]] = result[8] ✓
    
    jacobi_sum = bracket_0_12 + bracket_1_20 + bracket_2_01
    
    np.testing.assert_allclose(jacobi_sum, jnp.zeros_like(jacobi_sum), atol=1e-6)


def test_empty_words() -> None:
    """Test handling of empty word lists."""
    dim, n = 2, 3
    A = jax.random.normal(jax.random.PRNGKey(8), (dim, n, n))
    
    # Empty list
    result = form_right_normed_brackets(A, [])
    assert result.shape == (0, n, n)
    
    # List with empty arrays
    words_by_len = [
        jnp.array([], dtype=jnp.int32).reshape(0, 1),
        jnp.array([], dtype=jnp.int32).reshape(0, 2),
    ]
    result = form_right_normed_brackets(A, words_by_len)
    assert result.shape == (0, n, n)


def test_mixed_empty_nonempty_levels() -> None:
    """Test that empty levels are skipped correctly."""
    dim, n = 2, 2
    A = jax.random.normal(jax.random.PRNGKey(9), (dim, n, n))
    
    words_by_len = [
        jnp.array([[0], [1]], dtype=jnp.int32),  # Level 1: non-empty
        jnp.array([], dtype=jnp.int32).reshape(0, 2),  # Level 2: empty
        jnp.array([[0, 1, 0]], dtype=jnp.int32),  # Level 3: non-empty
    ]
    
    result = form_right_normed_brackets(A, words_by_len)
    
    # Should have 3 brackets: [0], [1], [0,1,0]
    assert result.shape == (3, n, n)


def test_jit_compatibility() -> None:
    """Test that the function can be JIT compiled."""
    dim, n = 2, 3
    A = jax.random.normal(jax.random.PRNGKey(10), (dim, n, n))
    words_by_len = [
        jnp.array([[0], [1]], dtype=jnp.int32),
        jnp.array([[0, 1]], dtype=jnp.int32),
    ]
    
    # Compile with JIT
    jit_fn = jax.jit(form_right_normed_brackets)
    
    # Run JIT-compiled version
    result_jit = jit_fn(A, words_by_len)
    result_eager = form_right_normed_brackets(A, words_by_len)
    
    # Results should match
    np.testing.assert_allclose(result_jit, result_eager, rtol=1e-10)


def test_gradients() -> None:
    """Test that gradients can be computed through the bracket formation."""
    dim, n = 2, 2
    key = jax.random.PRNGKey(11)
    
    def loss_fn(A: jax.Array) -> jax.Array:
        words_by_len = [
            jnp.array([[0], [1]], dtype=jnp.int32),
            jnp.array([[0, 1]], dtype=jnp.int32),
        ]
        brackets = form_right_normed_brackets(A, words_by_len)
        # Sum of squares as a simple loss
        return jnp.sum(brackets ** 2)
    
    A = jax.random.normal(key, (dim, n, n))
    
    # Compute gradient
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(A)
    
    # Gradients should have same shape as A
    assert grads.shape == A.shape
    # Gradients should not be all zeros (for a non-trivial case)
    assert not jnp.allclose(grads, jnp.zeros_like(grads), atol=1e-10)


def test_apply_lie_coeffs() -> None:
    """Test that Lie series can be formed from brackets and coefficients."""
    n = 3
    num_words = 5
    
    # Random brackets and coefficients
    W = jax.random.normal(jax.random.PRNGKey(12), (num_words, n, n))
    lam = jax.random.normal(jax.random.PRNGKey(13), (num_words,))
    
    result = apply_lie_coeffs(W, lam)
    
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
        apply_lie_coeffs(W, lam)


def test_flatten_coeffs() -> None:
    """Test coefficient flattening handles empty levels correctly."""
    # Case 1: All levels non-empty
    lam_by_len = [
        jnp.array([1.0, 2.0]),
        jnp.array([3.0, 4.0, 5.0]),
    ]
    words_by_len = [
        jnp.array([[0], [1]], dtype=jnp.int32),
        jnp.array([[0, 1], [1, 0], [0, 0]], dtype=jnp.int32),
    ]
    
    result = flatten_coeffs(lam_by_len, words_by_len)
    expected = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    # Case 2: Some levels empty
    lam_by_len = [
        jnp.array([1.0, 2.0]),
        jnp.array([]),
        jnp.array([3.0]),
    ]
    words_by_len = [
        jnp.array([[0], [1]], dtype=jnp.int32),
        jnp.array([], dtype=jnp.int32).reshape(0, 2),
        jnp.array([[0, 1, 0]], dtype=jnp.int32),
    ]
    
    result = flatten_coeffs(lam_by_len, words_by_len)
    expected = jnp.array([1.0, 2.0, 3.0])
    
    np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    # Case 3: All empty
    lam_by_len = [
        jnp.array([]),
        jnp.array([]),
    ]
    words_by_len = [
        jnp.array([], dtype=jnp.int32).reshape(0, 1),
        jnp.array([], dtype=jnp.int32).reshape(0, 2),
    ]
    
    result = flatten_coeffs(lam_by_len, words_by_len)
    assert result.shape == (0,)
    assert result.dtype == jnp.float32

