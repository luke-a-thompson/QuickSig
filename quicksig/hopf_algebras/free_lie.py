import jax
import jax.numpy as jnp
from quicksig.signatures.compute_log_signature import duval_generator


def commutator(a: jax.Array, b: jax.Array) -> jax.Array:
    return a @ b - b @ a


def _find_split_points_vectorized(
    words: jax.Array,
    prev_words_by_len: list[jax.Array],
) -> jax.Array:
    """
    Find split points for all words at once using vectorized operations.
    Returns array of split points [n_words].
    """
    n_words = words.shape[0]
    word_len = words.shape[1]
    splits = []

    for i in range(n_words):
        word = words[i]
        # Try splits from right to left (longest suffix first)
        split_found = word_len - 1  # Default to last possible split
        for split in range(word_len - 1, 0, -1):
            prefix = word[:split]
            suffix = word[split:]
            suffix_len = suffix.shape[0]
            prefix_len = prefix.shape[0]

            # Both prefix and suffix must be Lyndon words (or single letters)
            # Check if suffix exists in previous words
            suffix_ok = False
            if suffix_len <= len(prev_words_by_len):
                suffix_words = prev_words_by_len[suffix_len - 1]
                if suffix_words.size > 0:
                    matches = jnp.all(suffix_words == suffix[None, :], axis=1)
                    suffix_ok = bool(jnp.any(matches))
            elif suffix_len == 1:
                # Single letter is always valid
                suffix_ok = True

            # Check if prefix exists (or is single letter)
            prefix_ok = False
            if prefix_len <= len(prev_words_by_len):
                prefix_words = prev_words_by_len[prefix_len - 1]
                if prefix_words.size > 0:
                    matches = jnp.all(prefix_words == prefix[None, :], axis=1)
                    prefix_ok = bool(jnp.any(matches))
            elif prefix_len == 1:
                # Single letter is always valid
                prefix_ok = True

            # Standard factorization: both parts must be valid Lyndon words
            if suffix_ok and prefix_ok:
                split_found = split
                break

        splits.append(split_found)

    return jnp.array(splits, dtype=jnp.int32)


def _compute_lyndon_level_brackets(
    words: jax.Array,
    splits: jax.Array,
    prev_words_by_len: list[jax.Array],
    prev_brackets_by_len: list[jax.Array],
    A: jax.Array,
) -> jax.Array:
    """Compute brackets for a level of Lyndon words."""
    n_words = words.shape[0]
    n = A.shape[-1]
    level_brackets: list[jax.Array] = []

    # Process each word individually (Python loop, but brackets computed in JAX)
    for i in range(n_words):
        word = words[i]
        split = int(splits[i])  # Convert to Python int

        prefix = word[:split]
        suffix = word[split:]
        suffix_len = suffix.shape[0]

        # Find suffix bracket using vectorized lookup
        suffix_words = prev_words_by_len[suffix_len - 1]
        matches = jnp.all(suffix_words == suffix[None, :], axis=1)
        suffix_idx = jnp.argmax(matches)
        suffix_bracket = prev_brackets_by_len[suffix_len - 1][suffix_idx]

        # Get prefix bracket
        if split == 1:
            bracket = commutator(A[prefix[0]], suffix_bracket)
        else:
            prefix_words = prev_words_by_len[split - 1]
            prefix_matches = jnp.all(prefix_words == prefix[None, :], axis=1)
            prefix_idx = jnp.argmax(prefix_matches)
            prefix_bracket = prev_brackets_by_len[split - 1][prefix_idx]
            bracket = commutator(prefix_bracket, suffix_bracket)

        level_brackets.append(bracket)

    return jnp.stack(level_brackets)


def form_lyndon_brackets(
    A: jax.Array,
    depth: int,
    dim: int | None = None,
) -> jax.Array:
    """
    Form Lyndon brackets (commutators) for all Lyndon words up to given depth.

    Uses the standard factorization: for a Lyndon word w = uv where v is the
    longest proper Lyndon suffix, [w] = [[u], [v]].

    Args:
        A: [dim, n, n] array where A[i] is the i-th Lie algebra basis element.
        depth: Maximum depth (word length) to compute brackets for.
        dim: Dimension of the alphabet. If None, inferred from A.shape[0].

    Returns:
        W: [L, n, n] stacked Lyndon bracket matrices for all words in order.
           L = total number of Lyndon words up to depth.
    """
    if dim is None:
        dim = A.shape[0]

    # Generate Lyndon words using duval_generator
    words_by_len = duval_generator(depth, dim)

    if not words_by_len:
        n = A.shape[-1]
        return jnp.zeros((0, n, n), dtype=A.dtype)

    n = A.shape[-1]
    all_brackets: list[jax.Array] = []

    for word_len_idx, words in enumerate(words_by_len):
        if words.size == 0:
            continue

        word_length = word_len_idx + 1  # words at index k have length k+1

        # Compute brackets for this level
        if word_length == 1:
            # Level 1: just A[i] for each word
            level_brackets = A[words[:, 0]]  # [N1, n, n]
        else:
            # Level > 1: find splits and compute brackets
            splits = _find_split_points_vectorized(words, words_by_len[:word_len_idx])
            level_brackets = _compute_lyndon_level_brackets(
                words, splits, words_by_len[:word_len_idx], all_brackets, A
            )

        all_brackets.append(level_brackets)

    if not all_brackets:
        return jnp.zeros((0, n, n), dtype=A.dtype)

    return jnp.concatenate(all_brackets, axis=0)  # [L, n, n]


def form_right_normed_brackets(
    A: jax.Array,
    words_by_len: list[jax.Array],
) -> jax.Array:
    """
    Form right-normed brackets (commutators) for all words using matrix Lie algebra
    basis elements. Uses caching to avoid recomputing shared suffixes. JIT-compatible.

    Works for Lie algebras with matrix representations (matrix Lie groups like SO(n),
    SE(3), or homogeneous spaces like Stiefel manifolds via their symmetry group's
    Lie algebra representation).

    For ODEs dx/dt = f(x) on homogeneous spaces: A contains FIXED Lie algebra
    generators (e.g., so(n) for O(n), se(3) for SE(3)). Vector field f(x) provides
    position-dependent coefficients λ_i(x) such that f(x) = Σ_i λ_i(x) * A[i].
    This function precomputes brackets [A[i], [A[j], ...]] which are then multiplied
    by coefficients via apply_lie_coeffs().

    A:             [dim, n, n] array where A[i] is the i-th Lie algebra basis element.
                   Fixed and independent of ODE state x.
    words_by_len:  list; words_by_len[k] has shape [Nk, k+1] with ints in [0, dim)

    Returns:
        W: [L, n, n] stacked right-normed bracket matrices for all words in order.
           L = sum_k Nk. If no words, returns shape [0, n, n].
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
            suffix = words[:, 1:]  # [N, k] for words of length k+1
            # Find matching brackets
            matches = jnp.all(prev_words == suffix[:, None, :], axis=2)  # [N, M]
            match_indices = jnp.argmax(matches, axis=1)  # [N]
            suffix_brackets = prev_brackets[match_indices]  # [N, n, n]
            level_brackets = jax.vmap(commutator)(A[words[:, 0]], suffix_brackets)

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
