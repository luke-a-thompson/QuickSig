import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from functools import partial


@jax.jit
def batch_otimes_pure_jax(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """GPU-optimized batched tensor product preserving batch dimension.

    Args:
        x: ArrayLike shape (..., n)
        y: ArrayLike shape (..., m)

    Returns:
        ArrayLike: The batched tensor product of x and y.
    """
    return jnp.einsum("...i,...j->...ij", x, y, optimize=True)


@jax.jit
def batch_seq_otimes_pure_jax(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """GPU-optimized tensor product preserving both batch and sequence dimensions."""
    xdim = x.ndim
    ydim = y.ndim
    for i in range(ydim - 2):
        x = jnp.expand_dims(x, axis=-1)
    for i in range(xdim - 2):
        y = jnp.expand_dims(y, axis=2)
    return x * y


@partial(jax.jit, static_argnames="depth")
def batch_restricted_exp_pure_jax(input: ArrayLike, depth: int) -> list[ArrayLike]:
    """Computes restricted exponential with full GPU parallelization."""
    ret = [input]
    for i in range(2, depth + 1):
        ret.append(batch_otimes_pure_jax(ret[-1], input / i))
    return ret
