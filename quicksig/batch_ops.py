import jax
import jax.numpy as jnp


def batch_tensor_product(x: jax.Array, y: jax.Array) -> jax.Array:
    """GPU-optimized batched tensor product preserving batch dimension.
    Computes an outer product of the trailing dimensions of x and y,
    broadcasting over leading (batch) dimensions.
    For example, if x is (..., M) and y is (..., N), result is (..., M, N).
    If x is (..., M, K) and y is (..., N), result is (..., M, K, N).

    Args:
        x (jax.Array): Shape (..., d1) or (..., d1, d2, ...)
        y (jax.Array): Shape (..., d_final)

    Returns:
        jax.Array: The batched tensor product of x and y.
    """
    x_ndim_orig = x.ndim
    y_ndim_orig = y.ndim

    # Expand x by y's feature dimensions
    # If y is (B, N), y_ndim_orig-1 = 1. x becomes (..., d1, 1)
    # If y is (B, N, K), y_ndim_orig-1 = 2. x becomes (..., d1, 1, 1)
    for _ in range(y_ndim_orig - 1):
        x = jnp.expand_dims(x, axis=-1)
    for _ in range(x_ndim_orig - 1):
        y = jnp.expand_dims(y, axis=1)  # Inserts singleton dimension after the first (assumed batch)

    return x * y


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


def batch_tensor_log(sig_flat: jax.Array, depth: int, n_features: int) -> jax.Array:
    r"""
    Compute the log of a batched tensor.
    """
    B = sig_flat.shape[0]
    # Compute the dimension of each level: $$[\mathrm{n\_features}, \mathrm{n\_features}^2, \ldots, \mathrm{n\_features}^{\mathrm{depth}} ]$$
    level_sizes = n_features ** jnp.arange(1, depth + 1)
    splits = jnp.cumsum(level_sizes[:-1]) if depth > 1 else jnp.array([], dtype=sig_flat.dtype)
    # S_levels: list of [S_1, S_2, ..., S_depth], each S_p is (batch, n_features^p)
    sig_levels = jnp.split(sig_flat, splits, axis=-1)  # list of (batch, d**p)

    # S_levels[k] = $$T^{\otimes k}, \quad k=1,\dots,\text{depth}$$
    sig_levels = [lvl.reshape((B,) + (n_features,) * (k + 1)) for k, lvl in enumerate(sig_levels)]

    def tensor_mul(x: list[jax.Array], y: list[jax.Array], depth: int, S_levels_shapes: list[jax.Array]) -> list[jax.Array]:
        # S_levels_shapes is a list of zero tensors with the correct shapes for each level, e.g. initial S_levels.
        out = [jnp.zeros_like(S_levels_shapes[k]) for k in range(depth)]
        # out[i] will be the (i+1)-th order component.
        # Smallest order of X_p \otimes Y_q is 1+1=2 (if X, Y start at order 1).
        # So out[0] (order 1 component) is zero.
        for i in range(1, depth):  # i is the index for out, e.g., out[i] is order i+1
            # order_out = i + 1
            # We need sum of X_p \otimes Y_q where p+q = order_out.
            # X[j] is order j+1. Y[k] is order k+1.
            # (j+1) + (k+1) = i+1  => j+k = i-1.
            # j runs from 0 to i-1. k = i-1-j.
            acc = jnp.zeros_like(out[i])
            for j in range(i):  # j from 0 to i-1
                if j < len(x) and (i - 1 - j) < len(y):  # Ensure terms exist
                    term = batch_tensor_product(x[j], y[i - 1 - j])
                    acc = acc + term
            out[i] = acc
        return out

    result = [jnp.zeros_like(t) for t in sig_levels]
    tensor_exp_lvl = sig_levels

    for n in range(1, depth + 1):
        coef = (-1.0) ** (n - 1) / n  # $$c_k = (-1)^{k-1}/k$$
        result = [r + coef * p for r, p in zip(result, tensor_exp_lvl)]  # $$L_k \leftarrow L_k + \frac{(-1)^{k-1}}{k} \cdot T^{\otimes k}$$, $$L_k$$ is accumulates for the sum
        if n < depth:  # $$S^{⊗(n+1)}$$
            tensor_exp_lvl = tensor_mul(tensor_exp_lvl, sig_levels, depth, sig_levels)

    # --- flatten and concatenate ------------------------------------
    log_flat = jnp.concatenate([lvl.reshape(B, -1) for lvl in result], axis=-1)
    return log_flat


if __name__ == "__main__":
    from quicksig.path_signature import batch_signature_pure_jax

    key = jax.random.PRNGKey(0)
    path = jax.random.normal(key, shape=(2, 100, 4))
    signature = batch_signature_pure_jax(path, depth=5)
    print(signature.shape)
    tensor_log = batch_tensor_log(signature, 5, 4)
    print(tensor_log.shape)
