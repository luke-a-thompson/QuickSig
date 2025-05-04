import jax
import jax.numpy as jnp


def batch_tensor_product(x: jax.Array, y: jax.Array) -> jax.Array:
    """GPU-optimized batched tensor product preserving batch dimension.

    Args:
        x (jax.Array): Shape (..., n)
        y (jax.Array): Shape (..., n)

    Returns:
        jax.Array: The batched tensor product of x and y.
    """
    return jnp.einsum("...i,...j->...ij", x, y, optimize=True)


def batch_seq_tensor_product(x: jax.Array, y: jax.Array) -> jax.Array:
    """
    Outer product of the trailing dimensions while preserving the
    leading (batch, sequence) axes.

    Shapes
    -------
    x : (B, S, *A)          # *A is any tuple of ≥1 dims
    y : (B, S, *B_)         # *B_ is any tuple of ≥1 dims

    Returns
    --------
    (B, S, *A, *B_)
    """
    # trailing ranks
    a_rank = x.ndim - 2
    b_rank = y.ndim - 2

    # add singleton axes **once** instead of in a Python loop
    x_bcast = jnp.reshape(x, x.shape + (1,) * b_rank)
    y_bcast = jnp.reshape(y, y.shape[:2] + (1,) * a_rank + y.shape[2:])

    return x_bcast * y_bcast


# @partial(jax.jit, static_argnames="depth")
def batch_restricted_exp(x: jax.Array, depth: int) -> tuple[jax.Array, ...]:
    r"""
    Return the truncated tensor-exponential terms
    $$\frac{(\mathrm{base\_tensor})^{\otimes k}}{k!}\quad(k=1,\dots,\text{max\_order}).$$


    Args:
        x: ArrayLike shape (..., n)
        depth: int. The truncation order of the restricted tensor exponential, usually denoted k in literature.

    Returns:
        A tuple of length max_order, where the k-th entry is x^{⊗(k+1)}/(k+1)!.

        terms[k-1] is the k-th order term \frac{x^{\otimes k}}{k!} so has shape `(..., n, n, …, n)` with `k` copies of the last dimension.
    """
    terms = [x]
    for k in range(1, depth):
        divisor = k + 1
        next_factor = x / divisor
        next_power = batch_tensor_product(terms[-1], next_factor)
        terms.append(next_power)
    return tuple(terms)
