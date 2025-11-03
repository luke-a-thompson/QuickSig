import jax
import jax.numpy as jnp

def commutator(a: jax.Array, b: jax.Array) -> jax.Array:
    return a @ b - b @ a

def _find_suffix_bracket(
    suffix: jax.Array,
    prev_words: jax.Array,
    prev_brackets: jax.Array,
) -> jax.Array:
    """
    Find the bracket for a suffix word in the previous level's results.
    
    suffix: [suffix_len] array
    prev_words: [N, suffix_len] array of words from previous level
    prev_brackets: [N, n, n] array of brackets from previous level
    
    Returns: [n, n] bracket matrix
    """
    # Compare suffix with each word in prev_words
    matches = jnp.all(prev_words == suffix[None, :], axis=1)  # [N]
    # Find first match index
    match_idx = jnp.argmax(matches)
    # Extract the bracket (if no match, returns first bracket, but should always match)
    return prev_brackets[match_idx]


def _compute_level_brackets(
    words: jax.Array,
    A: jax.Array,
    prev_words: jax.Array,
    prev_brackets: jax.Array,
) -> jax.Array:
    """Compute brackets for one level of words, reusing suffixes from previous level."""
    
    def compute_bracket(i: int) -> jax.Array:
        word = words[i]
        # For [i1, i2, ..., ik], compute [i1, [i2, [..., ik]]]
        # Suffix is [i2, ..., ik] which we look up from prev_brackets
        suffix = word[1:]  # [k]
        suffix_bracket = _find_suffix_bracket(suffix, prev_words, prev_brackets)
        return commutator(A[word[0]], suffix_bracket)
    
    return jax.vmap(compute_bracket)(jnp.arange(words.shape[0]))


def form_right_normed_brackets(
    A: jax.Array,
    words_by_len: list[jax.Array],
) -> jax.Array:
    """
    Form right-normed brackets (commutators) for all words, which serve as
    the basis matrices for constructing Lie series. Uses caching to avoid
    recomputing shared suffixes. JIT-compatible.

    A:             [dim, n, n]
    words_by_len:  list; words_by_len[k] has shape [Nk, k+1] with ints in [0, dim)

    Returns:
        W: [L, n, n] stacked matrices representing right-normed brackets for all words
           in the given order. L = sum_k Nk. If no words, returns shape [0, n, n].
    """
    if not words_by_len:
        n = A.shape[-1]
        return jnp.zeros((0, n, n), dtype=A.dtype)

    n = A.shape[-1]
    all_brackets: list[jax.Array] = []
    
    # Initialize with empty array for level 0 (will be replaced)
    prev_brackets = jnp.zeros((0, n, n), dtype=A.dtype)
    prev_words = jnp.zeros((0, 1), dtype=jnp.int32)  # Empty, shape matches length 1 words

    for word_len_idx, words in enumerate(words_by_len):
        if words.size == 0:
            continue

        word_length = word_len_idx + 1  # words at index k have length k+1
        
        # Compute brackets for this level
        if word_length == 1:
            # Level 1: just A[i] for each word
            level_brackets = A[words[:, 0]]  # [N1, n, n]
        else:
            # Level > 1: reuse suffixes from previous level
            level_brackets = _compute_level_brackets(
                words, A, prev_words, prev_brackets
            )
        
        all_brackets.append(level_brackets)
        # Update cache for next level
        prev_brackets = level_brackets
        prev_words = words

    if not all_brackets:
        return jnp.zeros((0, n, n), dtype=A.dtype)

    return jnp.concatenate(all_brackets, axis=0)  # [L, n, n]

def flatten_coeffs(
    lam_by_len: list[jax.Array],
    words_by_len: list[jax.Array],
) -> jax.Array:
    """Concatenate coefficients only for non-empty word buckets (alignment-safe)."""
    lams = [lam for lam, words in zip(lam_by_len, words_by_len) if words.size != 0]
    return jnp.concatenate(lams, axis=0) if lams else jnp.zeros((0,), dtype=jnp.float32)

def apply_lie_coeffs(W: jax.Array, lam_flat: jax.Array) -> jax.Array:
    """Form the Lie series matrix: C = sum_w lam_w * W[w]."""
    if W.shape[0] != lam_flat.shape[0]:
        raise ValueError(f"Coefficient count {lam_flat.shape[0]} does not match number of words {W.shape[0]}.")
    return jnp.tensordot(lam_flat, W, axes=1)  # [n, n]

