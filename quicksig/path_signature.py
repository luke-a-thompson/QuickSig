import jax
import jax.numpy as jnp
from quicksig.batch_ops import batch_restricted_exp, batch_seq_tensor_product


def batch_signature(path: jax.Array, depth: int, stream: bool = False) -> list[jax.Array]:
    r"""Computes the truncated path signature
    $$\operatorname{Sig}_{0,T}(X)=\bigl(S^{(1)}_{0,T},\,S^{(2)}_{0,T},\ldots,S^{(m)}_{0,T}\bigr),\qquad m=\text{depth}.$$
    The constant term $$S^{(0)}_{0,T}=1$$ is omitted.

    Args:
        path: A JAX array of shape $$(\text{batch\_size}, \text{seq\_len}, \text{n\_features})$$
              representing the path segment $$X_{0:T}.$$
        depth: Maximum signature depth (denoted $$m$$) to compute.
        stream: If $$\text{True}$$, returns the partial signatures
                $$\operatorname{Sig}_{0,t}(X)$$ for each $$t$$; otherwise returns only
                $$\operatorname{Sig}_{0,T}(X).$$
                For example, with $$m=2$$ and $$\text{n\_features}=3$$,
                setting $$\text{stream}=\text{True}$$ gives shape
                $$(\text{batch\_size}, \text{seq\_len}-1,\;3+9=12),$$ since
                $$\dim S^{(1)}=3,\ \dim S^{(2)}=9.$$

    Returns:
        Let $$d=\text{n\_features}$$ and $$D_p=d^{\,p}.$$  The output dimension is $$\sum_{p=1}^{m}D_p.$$
        If $$\text{stream}=\text{False}$$:
            shape $$(\text{batch\_size},\;\sum_{p=1}^{m}D_p).$$
        If $$\text{stream}=\text{True}$$:
            shape $$(\text{batch\_size},\;\text{seq\_len}-1,\;\sum_{p=1}^{m}D_p).$$
    """
    assert depth > 0 and isinstance(depth, int), "Depth must be a positive integer."
    batch_size, seq_len, n_features = path.shape
    assert seq_len > 1, "Sequence length must be greater than 1."

    # Path increments: $$\Delta X_i = X_i - X_{i-1}$$ for $$i = 1, \ldots, N$$
    path_increments = path[:, 1:, :] - path[:, :-1, :]

    # Level 1 signature $$S^1_{0,t} = \textstyle{\sum_{i=1}^{t}} \Delta X_i$$
    # E.g. t=5 is cumsum 0 to 5
    depth_1_stream = jnp.cumsum(path_increments, axis=1)
    incremental_signatures = [depth_1_stream]

    if depth == 1:
        if stream:
            return [depth_1_stream]
        else:
            return [depth_1_stream[:, -1, :]]

    # Precompute $$S^k_{0,1} = (\Delta X_1)^{\otimes k}$$ for $$k = 1, \ldots, \text{depth}$$, len = k
    first_inc_tensor_exp_terms = batch_restricted_exp(path_increments[:, 0, :], depth=depth)

    # Precompute scaled increments: $$\Delta X_t / k$$ for $$k = 2, \ldots, \text{depth}$$
    divisors = jnp.arange(2, depth + 1, dtype=path_increments.dtype).reshape(depth - 1, 1, 1, 1)
    path_increment_divided = jnp.expand_dims(path_increments, axis=0) / divisors

    for k in range(1, depth):
        # Initialize accumulator:
        # $$\text{Aux}^{(1)}_t = S^{k-1}_{0,t-1} + (\Delta X_t) / k!, \quad \forall t = 1, \ldots, N - 1$$
        # Due to numpy indexing, :-1 and 1: are just numbers being added, the list-like indexing means vectorised addition over all seq_len - not actually adding 2 slices
        sig_accm = incremental_signatures[0][:, :-1, :] + path_increment_divided[k - 1, :, 1:, :]

        for j in range(k - 1):
            # $$ p = j + 1 $$
            prev_signature_level_term = incremental_signatures[j + 1][:, :-1, :]  # $$S^p_{0,t-1}$$

            scaled_increment = path_increment_divided[k - j - 2, :, 1:, :]  # $$\frac{ΔX_t}{(k-p+1)!}$$

            #  $$Aux^{(p)}_t  = S^{(k-p)}_{0,t-1} + Aux^{(p-1)}_t ⊗ \frac{ΔX_t}{(k-p+1)!}, \quad \forall p = 1, \dots, k-1$$
            #  This is a recursive tensor product, so each addition is just one $$\Delta X_t$$ scaled by where we are in the recursion
            sig_accm = prev_signature_level_term + batch_seq_tensor_product(sig_accm, scaled_increment)

        sig_accm = batch_seq_tensor_product(sig_accm, path_increments[:, 1:, :])

        # Concatenate the first increment (timestep) with the rest of the signature
        first_inc_expanded = jnp.expand_dims(first_inc_tensor_exp_terms[k], axis=1)  # Shape: [batch_size, 1, n_features ** (depth_index + 1)]
        sig_accm = jnp.concatenate([first_inc_expanded, sig_accm], axis=1)  # Shape: [batch_size, depth_index + 1, n_features ** (depth_index + 1)]

        # The depth-k signature up to time t
        incremental_signatures.append(jnp.cumsum(sig_accm, axis=1))

    if not stream:
        return [jnp.reshape(c[:, -1], (batch_size, n_features ** (1 + idx))) for idx, c in enumerate(incremental_signatures)]
    else:
        return [jnp.reshape(r, (batch_size, seq_len - 1, n_features ** (1 + idx))) for idx, r in enumerate(incremental_signatures)]
