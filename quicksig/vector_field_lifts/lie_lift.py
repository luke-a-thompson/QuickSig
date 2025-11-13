import jax
import jax.numpy as jnp
from quicksig.control_lifts.log_signature import duval_generator
from quicksig.vector_field_lifts.vector_field_lift_types import LyndonBrackets
from quicksig.hopf_algebras.free_lie import (
    find_split_points_vectorized,
    compute_lyndon_level_brackets,
)


def form_lyndon_brackets(
    A: jax.Array,
    depth: int,
) -> LyndonBrackets:
    """
    Form Lyndon brackets (commutators) for all Lyndon words up to given depth,
    returned per-level to mirror signature inputs.

    Uses the standard factorization: for a Lyndon word w = uv where v is the
    longest proper Lyndon suffix, [w] = [[u], [v]].

    Args:
        A: [dim, n, n] array where A[i] is the i-th Lie algebra basis element.
        depth: Maximum depth (word length) to compute brackets for.

    Returns:
        A list of length `depth`. Entry k contains the bracket matrices for
        Lyndon words of length k+1 with shape [Nk, n, n]. Empty levels yield
        a [0, n, n] array.
    """
    dim = A.shape[0]

    # Generate Lyndon words using duval_generator
    words_by_len = duval_generator(depth, dim)

    if not words_by_len:
        n = A.shape[-1]
        return LyndonBrackets([jnp.zeros((0, n, n), dtype=A.dtype) for _ in range(depth)])

    n = A.shape[-1]
    all_brackets: list[jax.Array] = []

    for word_len_idx, words in enumerate(words_by_len):
        if words.size == 0:
            all_brackets.append(jnp.zeros((0, n, n), dtype=A.dtype))
            continue

        word_length = word_len_idx + 1  # words at index k have length k+1

        # Compute brackets for this level
        if word_length == 1:
            # Level 1: just A[i] for each word
            level_brackets = A[words[:, 0]]  # [N1, n, n]
        else:
            # Level > 1: find splits and compute brackets
            splits = find_split_points_vectorized(words, words_by_len[:word_len_idx])
            level_brackets = compute_lyndon_level_brackets(
                words, splits, words_by_len[:word_len_idx], all_brackets, A
            )

        all_brackets.append(level_brackets)

    return LyndonBrackets(all_brackets)
