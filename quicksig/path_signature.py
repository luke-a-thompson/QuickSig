import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from functools import partial
from quicksig.batch_ops import batch_restricted_exp_pure_jax, batch_seq_otimes_pure_jax


@partial(jax.jit, static_argnames=["depth", "stream"])
def batch_signature_pure_jax(path: ArrayLike, depth: int, stream: bool = False) -> ArrayLike:
    """Highly optimized signature computation maximizing GPU parallelism.

    Key optimizations:
    - Pre-computes scaled increments for all depths
    - Uses cumsum for parallel sequence processing
    - Replaces sequential operations with parallel matrix ops
    - Fully JIT-compiled for maximum GPU utilization
    """
    batch_size, seq_len, n_features = path.shape
    path_increments = path[:, 1:] - path[:, :-1]

    stacked = [jnp.cumsum(path_increments, axis=1)]

    exp_term = batch_restricted_exp_pure_jax(path_increments[:, 0], depth=depth)

    if depth > 1:
        path_increment_divided = jnp.stack([path_increments / i for i in range(2, depth + 1)], axis=0)
    else:
        path_increment_divided = None

    for depth_index in range(1, depth):
        current = stacked[0][:, :-1] + path_increment_divided[depth_index - 1, :, 1:]
        for j in range(depth_index - 1):
            current = stacked[j + 1][:, :-1] + batch_seq_otimes_pure_jax(current, path_increment_divided[depth_index - j - 2, :, 1:])
        current = batch_seq_otimes_pure_jax(current, path_increments[:, 1:])
        current = jnp.concatenate([jnp.expand_dims(exp_term[depth_index], axis=1), current], axis=1)
        stacked.append(jnp.cumsum(current, axis=1))

    if not stream:
        return jnp.concatenate([jnp.reshape(c[:, -1], (batch_size, n_features ** (1 + idx))) for idx, c in enumerate(stacked)], axis=1)
    else:
        return jnp.concatenate([jnp.reshape(r, (batch_size, seq_len - 1, n_features ** (1 + idx))) for idx, r in enumerate(stacked)], axis=2)
