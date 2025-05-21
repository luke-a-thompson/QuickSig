import jax
import jax.numpy as jnp
from quicksig.batch_ops import batch_restricted_exp, batch_seq_tensor_product


def flatten_signature(incremental_signatures: list[jax.Array], batch_size: int, n_features: int, stream: bool = False) -> jax.Array:
    """Flattens a list of incremental signatures into a single array.

    Args:
        incremental_signatures: A list of signature arrays at different depths.
        batch_size: Batch size of the input.
        n_features: Number of features in the input.
        stream: If True, returns the stream of signatures for each time step.
                If False, returns only the final signature.

    Returns:
        A flattened jax.Array of signatures.
    """
    if stream:
        return jnp.concatenate(incremental_signatures, axis=2)
    else:
        return jnp.concatenate([jnp.reshape(c[:, -1], (batch_size, n_features ** (1 + idx))) for idx, c in enumerate(incremental_signatures)], axis=1)


def batch_signature(path: jax.Array, depth: int, stream: bool = False, flatten: bool = True) -> jax.Array | list[jax.Array]:
    r"""Computes the path signature $$S(X)_{0,T} = (S^1_{0,T}, S^2_{0,T}, \ldots, S^{\text{depth}}_{0,T})$$.
    We omit the constant term $$S^0=1$$.

    Args:
        path: a jax.Array of shape (batch_size, seq_len, n_features), representing $$X_t$$.
        depth: the depth (maximum level) of the signature to compute.
        stream: If True, returns the stream of signatures $$S(X)_{0,t}$$ for each $$t$$.
                If False, returns only the final signature $$S(X)_{0,T}$$.
                For example, if depth=2 and n_features=3, stream=True will return a tensor
                of shape (batch_size, seq_len-1, 3+9=12), where 12 is dim($$S^1$$)+dim($$S^2$$).
        flatten: If True, returns a flattened array. If False, returns the list of incremental signatures.

    Returns:
        If flatten=True:
            Let D_p = text{n_features}^p. Total signature dimension is $$\sum_{p=1}^{\text{depth}} D_p$$.
            If stream=False:
                jax.Array of shape (batch_size, $$\sum_{p=1}^{\text{depth}} D_p$$)
            If stream=True:
                jax.Array of shape (batch_size, seq_len - 1, $$\sum_{p=1}^{\text{depth}} D_p$$)
                (Note: seq_len - 1 is the number of increments/intervals, denoted $$N$$ below)
        If flatten=False:
            List of jax.Array, where each element is a signature of a particular depth.
            Each array has shape:
            If stream=False:
                For depth k: (batch_size, n_features^k)
            If stream=True:
                For depth k: (batch_size, seq_len - 1, n_features^k)
    """
    assert depth > 0 and isinstance(depth, int), "Depth must be a positive integer."
    batch_size, seq_len, n_features = path.shape
    assert seq_len > 1, "Sequence length must be greater than 1."

    # Path increments: $$\Delta X_i = X_i - X_{i-1}$$ for $$i=1, \ldots, T$$
    # Shape: [batch_size, seq_len - 1, n_features]
    path_increments = path[:, 1:] - path[:, :-1]

    # Level 1 signature $$S^1_{0,t} = \sum_{i=1}^{t} \Delta X_i$$
    # E.g. t=5 is cumsum 0 to 5
    depth_1_stream = jnp.cumsum(path_increments, axis=1)
    incremental_signatures = [depth_1_stream]

    if depth == 1:
        if flatten:
            if stream:
                # Full time‑series signature for $$S^1_{0,T}$$
                return incremental_signatures[0]
            else:
                # Only the end‑point signature for $$S^1_T$$
                return incremental_signatures[0][:, -1, :]  # Shape (batch_size, n_features)
        else:
            if not stream:
                # Convert to final signatures only when not streaming and not flattened
                incremental_signatures[0] = incremental_signatures[0][:, -1, :]
            return incremental_signatures

    # For higher depths:
    # Precompute $$S^k_{0,1} = (\Delta X_1)^{\otimes k} / k!, \quad \forall k=1...depth$$
    # Shape: tuple of arrays, k-th element (batch_size, n_features^k)
    first_inc_tensor_exp_terms: tuple[jax.Array, ...] = batch_restricted_exp(path_increments[:, 0, :], depth=depth)

    # Precompute scaled path increments $$k$$ for the each depth $$\Delta X_t / k, \quad \forall k = 2, \ldots, \text{depth}$$
    divisors = jnp.arange(2, depth + 1, dtype=path_increments.dtype).reshape(depth - 1, 1, 1, 1)
    # path_increment_divided[idx] corresponds to $$\Delta X_t / (\text{idx}+2)$$
    path_increment_divided = jnp.expand_dims(path_increments, axis=0) / divisors

    for depth_index in range(1, depth):
        # ------------------------------------------------------------------
        # We compute the kth‑order signature increment
        #
        #   $$ΔS^{(k)}_{t-1,t} := S^{(k)}_{0,t} − S^{(k)}_{0,t-1} = ∑_{r=0}^{k-1} S^{(k-1-r)}_{0,t-1} ⊗ \frac{(ΔX_t)^{⊗(r+1)}}{(r+1)!}$$
        #
        # via the recursion:
        #   $$Aux^{(1)}_t  = S^{(k-1)}_{0,t-1} + \frac{(ΔX_t)^{⊗k}}{k!}$$
        #
        #   $$Aux^{(p)}_t  = S^{(k-p)}_{0,t-1} + Aux^{(p-1)}_t ⊗ \frac{(ΔX_t)^{⊗(k-p+1)}}{(k-p+1)!}, \quad \forall  p = 2, …, k-1$$
        #
        # After k−1 recursions $$Aux^{(k-1)}_t$$ holds the full shuffle sum:
        #
        #   $$\sum_{r=0}^{k-1} S^{(k-1-r)}_{0,t-1} ⊗ \frac{(ΔX_t)^{⊗(r+1)}}{(r+1)!}$$.
        #
        # Multiplying by $$ΔX_t$$ yields $$ΔS^{(k)}_{t-1,t}$$.
        # ------------------------------------------------------------------
        sig_accm = incremental_signatures[0][:, :-1, :] + path_increment_divided[depth_index - 1, :, 1:, :]

        for j in range(depth_index - 1):
            prev_signature_level_term = incremental_signatures[j + 1][:, :-1, :]  # $$S^p_{0,t-1}$$

            scaled_increment = path_increment_divided[depth_index - j - 2, :, 1:, :]  # $$\frac{(ΔX_t)^{⊗(k-p+1)}}{(k-p+1)!}$$

            #  $$Aux^{(p)}_t  = S^{(k-p)}_{0,t-1} + Aux^{(p-1)}_t ⊗ \frac{(ΔX_t)^{⊗(k-p+1)}}{(k-p+1)!}$$
            sig_accm = prev_signature_level_term + batch_seq_tensor_product(sig_accm, scaled_increment)
        sig_accm = batch_seq_tensor_product(sig_accm, path_increments[:, 1:, :])

        # Concatenate the first increment (timestep) with the rest of the signature
        first_inc_expanded = jnp.expand_dims(first_inc_tensor_exp_terms[depth_index], axis=1)  # Shape: [batch_size, 1, n_features ** (depth_index + 1)]
        sig_accm = jnp.concatenate([first_inc_expanded, sig_accm], axis=1)  # Shape: [batch_size, depth_index + 1, n_features ** (depth_index + 1)]

        # incremental_signatures[k] is the depth-k signature up to time t
        incremental_signatures.append(jnp.cumsum(sig_accm, axis=1))

    if not flatten:
        if not stream:
            # Convert to final signatures only when not streaming and not flattened
            return [sig[:, -1, :] for sig in incremental_signatures]
        else:
            return incremental_signatures
    else:
        if stream:
            return jnp.concatenate(incremental_signatures, axis=2)
        else:
            return jnp.concatenate([jnp.reshape(c[:, -1], (batch_size, n_features ** (1 + idx))) for idx, c in enumerate(incremental_signatures)], axis=1)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    path = jax.random.normal(key, shape=(2, 100, 3))
    signature = batch_signature(path, depth=5)
    print(signature.shape)

    # Example of unflattened signatures
    unflattened_signatures = batch_signature(path, depth=3, flatten=False)
    print(f"Number of unflattened signature levels: {len(unflattened_signatures)}")
    for i, sig in enumerate(unflattened_signatures):
        print(f"Depth {i+1} signature shape: {sig.shape}")
