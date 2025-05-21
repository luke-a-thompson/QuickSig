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

    $$\exp(x) = \sum_{k=0}^{\infty} \frac{x^{\otimes k}}{k!}$$

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
        next_power = batch_tensor_product(terms[-1], next_factor)  # $$x^{\otimes (k+1)} / (k+1)!$$
        terms.append(next_power)
    return tuple(terms)


def batch_cauchy_prod(x: list[jax.Array], y: list[jax.Array], depth: int, S_levels_shapes: list[jax.Array]) -> list[jax.Array]:
    r"""
    Computes the degree-m component of the graded tensor-concatenation product
    or Cauchy convolution product in the truncated free tensor algebra.
    $$
    Z^{(m)} \;=\;\sum_{p+q=m} X^{(p)} \otimes Y^{(q)}
    $$
    This is the degree-m truncation of the full (Cauchy) product
    $$
    X \cdot Y = \sum_{p,q\ge0} X^{(p)} \otimes Y^{(q)}.
    $$
    """

    # out[i] holds $$Z^{(i+1)}, Z = x \otimes y$$
    out = [jnp.zeros_like(S_levels_shapes[k]) for k in range(depth)]
    # order-1 term is zero as there is no way to split $$1 = (p+1)+(q+1)$$ with $$p,q ≥ 0$$
    for i in range(1, depth):  # i is the index for out, e.g., out[i] is order i+1

        # we want $$Z^{(i+1)} = \sum_{(j+1)+(k+1)=i+1} X^{(j+1)}⊗Y^{(k+1)}$$
        # i.e. we want to sum over all ways to split $$i+1 = (j+1)+(k+1)$$ with $$j,k ≥ 0$$
        acc = jnp.zeros_like(out[i])
        for j in range(i):
            if j < len(x) and (i - 1 - j) < len(y):  # Ensure terms exist
                k = i - 1 - j
                term = batch_tensor_product(x[j], y[k])
                acc = acc + term  # $$X^{(j+1)}⊗Y^{(k+1)}$$
        out[i] = acc
    return out


def batch_tensor_log(sig_levels: list[jax.Array], n_features: int) -> list[jax.Array]:
    """Compute the log of a batched tensor."""
    assert isinstance(sig_levels, list), "sig_levels must be a list of jax.Array signatures per level"
    B = sig_levels[0].shape[0]

    # Reshape each level to have the proper tensor structure
    sig_levels = [lvl.reshape((B,) + (n_features,) * (k + 1)) for k, lvl in enumerate(sig_levels)]

    result = [jnp.zeros_like(t) for t in sig_levels]
    tensor_exp_lvl = sig_levels

    for n in range(1, len(sig_levels) + 1):
        coef = (-1.0) ** (n - 1) / n  # $$c_k = (-1)^{k-1}/k$$
        result = [res + coef * p for res, p in zip(result, tensor_exp_lvl)]  # $$L_k \leftarrow L_k + \frac{(-1)^{k-1}}{k} \cdot T^{\otimes k}$$, $$L_k$$ this accumulates for the sum
        if n < len(sig_levels):  # $$S^{⊗(n+1)}$$
            # Math note: We must use the Cauchy product because the atomic tensor product is not defined for lists of signatures.
            tensor_exp_lvl = batch_cauchy_prod(tensor_exp_lvl, sig_levels, len(sig_levels), sig_levels)

    # --- flatten and concatenate ------------------------------------
    return [lvl.reshape(B, -1) for lvl in result]
