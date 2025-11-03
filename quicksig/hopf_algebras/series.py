import jax
import jax.numpy as jnp

def lie_polynomial(bracket_basis: jax.Array, lie_coefficients: jax.Array) -> jax.Array:
    """Form the Lie series matrix: C = sum_w lam_w * W[w].
    
    A lie polynomial is a linear combination of Lie brackets (commutators).
    It is a truncation of an infinite Lie series.
    
    Works with any bracket basis (e.g., right-normed or Lyndon) as long as the
    coefficients are in the same ordering as the brackets.

    Args:
        bracket_basis: [L, n, n] array of Lie brackets (e.g., from form_lyndon_brackets).
        lie_coefficients: [L] array of coefficients (e.g., flattened log signature in Lyndon coordinates).

    Returns:
        [n, n] array of the Lie series matrix.
        
    Example:
        For log signature in Lyndon coordinates:
        >>> log_sig = compute_log_signature(path, depth, log_signature_type="Lyndon words")
        >>> brackets = form_lyndon_brackets(A, depth, dim)
        >>> coeffs = flatten_coeffs(log_sig.signature, duval_generator(depth, dim))
        >>> lie_elem = lie_polynomial(brackets, coeffs)
    """
    if bracket_basis.shape[0] != lie_coefficients.shape[0]:
        raise ValueError(f"Coefficient count {lie_coefficients.shape[0]} does not match number of brackets {bracket_basis.shape[0]}.")
    return jnp.tensordot(lie_coefficients, bracket_basis, axes=1)  # [n, n]