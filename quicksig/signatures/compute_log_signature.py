from functools import partial
import jax
from quicksig.tensor_ops import tensor_log
from quicksig.signatures.compute_path_signature import compute_path_signature
from quicksig.signatures.signature_types import Signature, LogSignature

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
    """
    Compute the log signature of a path.

    Args:
        path: A JAX array of shape (seq_len, n_features).
        depth: The depth of the log signature.
        log_signature_type: Type of log signature, "expanded" or "lyndon".
        mode:
            If "stream", returns a stream of log signatures.
            If "incremental", returns the log signature of each increment.
            If "full", returns the full log signature.

    Returns:
        A list of JAX arrays representing the log signature.
        If stream is True, each array will have a leading time dimension.
    """

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

    Args:
        depth (int): The maximum length of words to generate. This corresponds to the
               depth of the signature or log-signature. Must be > 0.
        dim (int): The dimension in path space, also representing the size of the
             alphabet (0 to dim-1) from which words are constructed. Must be >= 1.

    Returns:
        list[jax.Array]: A list of JAX arrays with len(list) == depth.
        The k-th JAX array contains words of length `k+1`.
        Each word is a row in the JAX array, so `result[k]` has shape `(num_words_at_length_k+1, k+1)`.
        If no words of a certain length are generated (e.g., for `dim=1` and length > 1), the corresponding JAX array will have shape `(0, k+1)`.
    """
    if dim == 1:
        # For dim=1, only one Lyndon word of length 1 exists: [0], all higher-order Lyndon word lists are empty.
        first_level_word = [jnp.array([[0]], dtype=jnp.int32)]  # Word [0], length 1
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

    Args:
        input: size (dim, dim, ..., dim)
        indices: size (dim, n)
    Return:
        A 1D jnp.ndarray
    """

    # Handle scalar input: if log-signature term is scalar (e.g., 0.0 for higher orders),its projection onto Lyndon basis is all zeros.
    if input.ndim == 0:
        return jnp.zeros(indices.shape[0], dtype=input.dtype)

    dim_first_axis = input.shape[0]
    ndim_input_tensor = input.ndim
    n_components_in_indices = indices.shape[
        1
    ]  # Number of components in each Lyndon word (row of indices)

    # If Lyndon words have more components than the tensor has dimensions,
    # the tensor is degenerate w.r.t this basis. All projections are zero.
    if n_components_in_indices > ndim_input_tensor:
        return jnp.zeros(indices.shape[0], dtype=input.dtype)

    # Flatten matrix A in Fortran-style
    flattened = input.ravel("F")

    # Strides for Fortran-style indexing.
    # Assumes a tensor of rank `n_components_in_indices` where each dimension's size is `dim_first_axis`.
    # This is appropriate if n_components_in_indices == ndim_input_tensor.
    # If n_components_in_indices < ndim_input_tensor (e.g., 2-component words for a 3D tensor),
    # this selects from the "initial part" of the tensor, which is valid.
    strides_array = jnp.array([dim_first_axis**i for i in range(n_components_in_indices)])

    def _select(one_index_row: jax.Array) -> jax.Array:
        """one_index_row is a 1D jnp.ndarray (a single Lyndon word)"""
        # This computes the flat index for Fortran-style raveled arrays
        position = jnp.sum(one_index_row * strides_array)
        return flattened[position]

    return jax.vmap(_select)(indices)


def compress(expanded_terms: list[jax.Array], lyndon_indices: list[jax.Array]) -> list[jax.Array]:
    """
    Compress expanded log-signatures using Lyndon words.

    Args:
        expanded_terms: List of `jnp.ndarray`. Each element is a tensor,
                        e.g., expanded_terms[k] has shape (n_features, ..., n_features).
        lyndon_indices: List of `jnp.ndarray` generated by Lyndon words. Each element
                        lyndon_indices[k] has shape (num_lyndon_words_at_level_k, k+1).

    Returns:
        A list of compressed `jnp.ndarray`. Each element will have shape
        (num_lyndon_words_at_level_k,).
    """

    result_compressed = []
    for term, term_lyndon_indices in zip(expanded_terms, lyndon_indices):
        # term shape: (n_features, ..., n_features)
        # term_lyndon_indices shape: (num_lyndon_words, num_components)

        # Select components based on Lyndon indices for the current term
        compressed_term = index_select(term, term_lyndon_indices)
        # compressed_term shape: (num_lyndon_words,)
        result_compressed.append(compressed_term)

    return result_compressed
