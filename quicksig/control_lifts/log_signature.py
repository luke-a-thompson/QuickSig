from functools import partial
import jax
from quicksig.tensor_ops import tensor_log
from quicksig.control_lifts.path_signature import compute_path_signature
from quicksig.control_lifts.signature_types import Signature, LogSignature

from typing import Literal, overload
from collections import defaultdict
import jax.numpy as jnp


@overload
def compute_log_signature(
    path: jax.Array,
    depth: int,
    log_signature_type: Literal["Tensor words", "Lyndon words"],
    mode: Literal["full"],
) -> LogSignature: ...


@overload
def compute_log_signature(
    path: jax.Array,
    depth: int,
    log_signature_type: Literal["Tensor words", "Lyndon words"],
    mode: Literal["stream", "incremental"],
) -> list[LogSignature]: ...


@partial(jax.jit, static_argnames=["depth", "log_signature_type", "mode"])
def compute_log_signature(
    path: jax.Array,
    depth: int,
    log_signature_type: Literal["Tensor words", "Lyndon words"],
    mode: Literal["full", "stream", "incremental"],
) -> LogSignature | list[LogSignature]:
    n_features = path.shape[-1]
    signature_result = compute_path_signature(path, depth, mode=mode)

    def _get_log_signature(signature: Signature) -> LogSignature:
        if log_signature_type == "Tensor words":
            log_sig_tensors = tensor_log(signature.signature, n_features)
        elif log_signature_type == "Lyndon words":
            indices = duval_generator(depth, n_features)
            log_signature_expanded = tensor_log(
                signature.signature, n_features, flatten_output=False
            )
            log_sig_tensors = compress(log_signature_expanded, indices)
        else:
            raise ValueError(f"Invalid log signature type: {log_signature_type}")

        return LogSignature(
            signature=log_sig_tensors,
            interval=signature.interval,
            basis_name=log_signature_type,
        )

    if isinstance(signature_result, list):
        return [_get_log_signature(sig) for sig in signature_result]
    else:
        return _get_log_signature(signature_result)


@partial(jax.jit, static_argnames=["depth", "dim"])
def duval_generator(depth: int, dim: int) -> list[jax.Array]:
    """Generates lists of words (integer sequences) for each level up to a specified depth.
    These words typically correspond to the Lyndon word basis.
    Ref: https://www.lyndex.org/algo.php
    """
    if dim == 1:
        first_level_word = [jnp.array([[0]], dtype=jnp.int32)]
        higher_level_empty_words = [jnp.empty((0, i + 1), dtype=jnp.int32) for i in range(1, depth)]
        return first_level_word + higher_level_empty_words

    list_of_words = defaultdict(list)
    word = [-1]
    while word:
        word[-1] += 1
        m = len(word)
        list_of_words[m - 1].append(jnp.array(word))
        while len(word) < depth:
            word.append(word[-m])
        while word and word[-1] == dim - 1:
            word.pop()

    return [jnp.stack(list_of_words[i]) for i in range(depth)]


def index_select(input: jax.Array, indices: jax.Array) -> jax.Array:
    """
    Select entries in m-level tensor based on given indices
    This function will help compressing log-signatures
    """
    if input.ndim == 0:
        return jnp.zeros(indices.shape[0], dtype=input.dtype)

    dim_first_axis = input.shape[0]
    ndim_input_tensor = input.ndim
    n_components_in_indices = indices.shape[1]

    if n_components_in_indices > ndim_input_tensor:
        return jnp.zeros(indices.shape[0], dtype=input.dtype)

    flattened = input.ravel("F")

    strides_array = jnp.array([dim_first_axis**i for i in range(n_components_in_indices)])

    def _select(one_index_row: jax.Array) -> jax.Array:
        position = jnp.sum(one_index_row * strides_array)
        return flattened[position]

    return jax.vmap(_select)(indices)


def compress(expanded_terms: list[jax.Array], lyndon_indices: list[jax.Array]) -> list[jax.Array]:
    result_compressed = []
    for term, term_lyndon_indices in zip(expanded_terms, lyndon_indices):
        compressed_term = index_select(term, term_lyndon_indices)
        result_compressed.append(compressed_term)
    return result_compressed



