import jax
import jax.numpy as jnp

def lie_polynomial(right_normed_basis: jax.Array, lie_coefficients: jax.Array) -> jax.Array:
    """Form the Lie series matrix: C = sum_w lam_w * W[w].
    
    A lie polynomial is a linear combination of right-normed brackets.
    It is a truncation of an infinite Lie series.

    Args:
        right_normed_basis: [L, n, n] array of right-normed brackets.
        lie_coefficients: [L] array of coefficients.

    Returns:
        [n, n] array of the Lie series matrix.
    """
    if right_normed_basis.shape[0] != lie_coefficients.shape[0]:
        raise ValueError(f"Coefficient count {lie_coefficients.shape[0]} does not match number of words {right_normed_basis.shape[0]}.")
    return jnp.tensordot(lie_coefficients, right_normed_basis, axes=1)  # [n, n]