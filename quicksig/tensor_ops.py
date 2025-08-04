import jax
import jax.numpy as jnp


def tensor_product(x: jax.Array, y: jax.Array) -> jax.Array:
    """Computes the outer product of two tensors.

    This is a pure feature-space operation. It does not align batch dimensions.
    For inputs `x` with shape `A` and `y` with shape `B`, the result has shape `A + B`.

    Args:
        x (jax.Array): The first tensor.
        y (jax.Array): The second tensor.

    Returns:
        jax.Array: The tensor product of x and y.
    """
    x_bcast = jnp.reshape(x, x.shape + (1,) * y.ndim)  # Make col vector
    y_bcast = jnp.reshape(y, (1,) * x.ndim + y.shape)  # Make row vector
    return x_bcast * y_bcast


def seq_tensor_product(x: jax.Array, y: jax.Array) -> jax.Array:
    """
    Outer product of the trailing dimensions while preserving the
    leading sequence axis.

    Shapes
    -------
    x : (S, *A)          # *A is any tuple of ≥1 dims
    y : (S, *B_)         # *B_ is any tuple of ≥1 dims

    Returns
    --------
    (S, *A, *B_)
    """
    # trailing ranks
    a_rank = x.ndim - 1
    b_rank = y.ndim - 1

    # add singleton axes **once** instead of in a Python loop
    x_bcast = jnp.reshape(x, x.shape + (1,) * b_rank)
    y_bcast = jnp.reshape(y, y.shape[:1] + (1,) * a_rank + y.shape[1:])  # Only take first dim (sequence)

    return x_bcast * y_bcast


def restricted_tensor_exp(x: jax.Array, depth: int) -> list[jax.Array]:
    r"""
    Return the truncated tensor-exponential terms

    $$\exp(x) = \sum_{k=0}^{\infty} \frac{x^{\otimes k}}{k!}$$

    Args:
        x: ArrayLike shape (..., n)
        depth: int. The truncation order of the restricted tensor exponential, usually denoted k in literature.

    Returns:
        A tuple of length max_order, where the k-th entry is $$x^{⊗(k+1)}/(k+1)!$$.

        terms[k-1] is the k-th order term $$\frac{x^{\otimes k}}{k!}$$ so has shape `(..., n, n, …, n)` with `k` copies of the last dimension.
    """
    terms = [x]
    for k in range(1, depth):
        divisor = k + 1
        next_factor = x / divisor
        next_power = tensor_product(terms[-1], next_factor)  # $$x^{\otimes (k+1)} / (k+1)!$$
        terms.append(next_power)
    return terms


def cauchy_convolution(x: list[jax.Array], y: list[jax.Array], depth: int, S_levels_shapes: list[jax.Array]) -> list[jax.Array]:
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
                term = tensor_product(x[j], y[k])
                acc = acc + term  # $$X^{(j+1)}⊗Y^{(k+1)}$$
        out[i] = acc
    return out


def tensor_log(sig_levels: list[jax.Array], n_features: int, flatten_output: bool = True) -> list[jax.Array]:
    """Compute the log of a tensor."""
    assert isinstance(sig_levels, list), "sig_levels must be a list of jax.Array signatures per level"

    # Reshape each level to have the proper tensor structurey
    sig_levels = [lvl.reshape((n_features,) * (k + 1)) for k, lvl in enumerate(sig_levels)]

    result = [jnp.zeros_like(t) for t in sig_levels]
    tensor_exp_lvl = sig_levels

    for n in range(1, len(sig_levels) + 1):
        coef = (-1.0) ** (n - 1) / n  # $$c_k = (-1)^{k-1}/k$$
        # $$L_k \leftarrow L_k + \tfrac{(-1)^{k-1}}{k} \cdot T^{\otimes k}$$, $$L_k$$ this accumulates for the sum
        result = [res + coef * p for res, p in zip(result, tensor_exp_lvl)]
        if n < len(sig_levels):  # $$S^{⊗(n+1)}$$
            # Math note: We must use the Cauchy product because the atomic tensor product is not defined for lists of signatures.
            tensor_exp_lvl = cauchy_convolution(tensor_exp_lvl, sig_levels, len(sig_levels), sig_levels)

    # --- flatten ------------------------------------
    if flatten_output:
        return [lvl.reshape(-1) for lvl in result]
    return result
