import jax
import jax.numpy as jnp
from quicksig.tensor_ops import restricted_tensor_exp, seq_tensor_product
from typing import Literal, overload
from quicksig.signatures.signature_types import Signature
from functools import partial


def _compute_incremental_levels(path_increments: jax.Array, depth: int) -> list[jax.Array]:
    """Compute cumulative signature levels up to each t for k=1..depth.

    Returns a list where entry k-1 has shape [T, n_features ** k].
    """
    # Level 1 stream: cumulative sum of increments
    depth_1_stream = jnp.cumsum(path_increments, axis=0)
    incremental_signatures: list[jax.Array] = [depth_1_stream]

    # Precompute S^k_{0,1} = (ΔX_1)^{⊗ k}
    first_inc_tensor_exp_terms = restricted_tensor_exp(path_increments[0, :], depth=depth)

    # Precompute scaled increments: ΔX_t / k for k = 2..depth
    divisors = jnp.arange(2, depth + 1, dtype=path_increments.dtype).reshape(depth - 1, 1, 1)
    path_increment_divided = jnp.expand_dims(path_increments, axis=0) / divisors

    for k in range(1, depth):
        # Aux^{(1)}_t = S^1_{0,t-1} + ΔX_t/(k+1)
        sig_accm = incremental_signatures[0][:-1, :] + path_increment_divided[k - 1, 1:, :]

        for j in range(k - 1):
            prev_signature_level_term = incremental_signatures[j + 1][:-1, :]
            scaled_increment = path_increment_divided[k - j - 2, 1:, :]
            sig_accm = prev_signature_level_term + seq_tensor_product(sig_accm, scaled_increment)

        # ΔS^{k+1}_t = Aux^{(k)}_t ⊗ ΔX_t
        sig_accm = seq_tensor_product(sig_accm, path_increments[1:, :])

        # prepend first timestep term
        first_inc_expanded = jnp.expand_dims(first_inc_tensor_exp_terms[k], axis=0)
        sig_accm = jnp.concatenate([first_inc_expanded, sig_accm], axis=0)

        incremental_signatures.append(jnp.cumsum(sig_accm, axis=0))

    return incremental_signatures


@overload
def compute_path_signature(
    path: jax.Array,
    depth: int,
    mode: Literal["full"],
    index_start: int = 0,
) -> Signature: ...


@overload
def compute_path_signature(
    path: jax.Array,
    depth: int,
    mode: Literal["stream", "incremental"],
    index_start: int = 0,
) -> list[Signature]: ...


@partial(jax.jit, static_argnames=["depth", "mode"])
def compute_path_signature(
    path: jax.Array,
    depth: int,
    mode: Literal["full", "stream", "incremental"],
    index_start: int = 0,
) -> Signature | list[Signature]:
    r"""Computes the truncated path signature
    $$\operatorname{Sig}_{0,T}(X)=\bigl(S^{(1)}_{0,T},\,S^{(2)}_{0,T},\ldots,S^{(m)}_{0,T}\bigr),\qquad m=\text{depth}.$$
    The constant term $$S^{(0)}_{0,T}=1$$ is omitted.

    Args:
        path: A JAX array of shape $$(\text{seq\_len}, \text{n\_features})$$
              representing the path segment $$X_{0:T}.$$
        depth: Maximum signature depth (denoted $$m$$) to compute.
        stream: If $$\text{True}$$, returns the partial signatures
                $$\operatorname{Sig}_{0,t}(X)$$ for each $$t$$; otherwise returns only
                $$\operatorname{Sig}_{0,T}(X).$$
                For example, with $$m=2$$ and $$\text{n\_features}=3$$,
                setting $$\text{stream}=\text{True}$$ gives shape
                $$(\text{seq\_len}-1,\;3+9=12),$$ since
                $$\dim S^{(1)}=3,\ \dim S^{(2)}=9.$$

    Returns:
        Let $$d=\text{n\_features}$$ and $$D_p=d^{\,p}.$$  The output dimension is $$\sum_{p=1}^{m}D_p.$$
        If $$\text{stream}=\text{False}$$:
            shape $$(\sum_{p=1}^{m}D_p).$$
        If $$\text{stream}=\text{True}$$:
            shape $$(\;\text{seq\_len}-1,\;\sum_{p=1}^{m}D_p).$$
    Note:
        Intervals are expressed in global sample indices using `index_start`:
        - full: `(index_start, index_start + N)`
        - stream: `(index_start, index_start + t)` for each t
        - incremental: `(index_start + i, index_start + i + 1)` for each i
    """
    assert depth > 0 and isinstance(depth, int), f"Depth must be a positive integer, got {depth}."
    if path.ndim == 1:
        raise ValueError(
            f"QuickSig requires 2D arrays of shape [seq_len, n_features]. Got shape: {path.shape}. \n Consider using path.reshape(-1, 1)."
        )
    seq_len, n_features = path.shape
    if seq_len <= 1:
        if mode == "full":
            zero_terms = [
                jnp.zeros((n_features ** (i + 1),), dtype=path.dtype) for i in range(depth)
            ]
            return Signature(
                signature=zero_terms,
                interval=(index_start, index_start + seq_len),
            )
        elif mode in ("stream", "incremental"):
            return []
        else:
            raise ValueError(f"Invalid mode: {mode}")

    # Non-degenerate path: compute increments once
    path_increments = path[1:, :] - path[:-1, :]

    match mode:
        case "incremental":
            return [
                Signature(
                    signature=restricted_tensor_exp(path_increments[i, :], depth=depth),
                    interval=(index_start + i, index_start + i + 1),
                )
                for i in range(path_increments.shape[0])
            ]
        case "full":
            incremental_signatures = _compute_incremental_levels(path_increments, depth)
            final_levels = [jnp.array(c[-1]) for c in incremental_signatures]
            return Signature(
                signature=final_levels,
                interval=(index_start, index_start + path.shape[0]),
            )
        case "stream":
            incremental_signatures = _compute_incremental_levels(path_increments, depth)
            return [
                Signature(
                    signature=[term[i, :] for term in incremental_signatures],
                    interval=(index_start, index_start + i + 1),
                )
                for i in range(path_increments.shape[0])
            ]
        case _:
            raise ValueError(f"Invalid mode: {mode}")
