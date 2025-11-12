import math
from functools import partial

import jax


# @partial(jax.jit, static_argnames=("depth", "dim"))
def get_signature_dim(depth: int, dim: int) -> int:
    """Compute the total dimension of the signature space up to depth."""
    return sum(dim**k for k in range(1, depth + 1))


def _get_prime_factorization(n: int) -> dict[int, int]:
    factors: dict[int, int] = {}
    d = 2
    temp_n = n
    while d * d <= temp_n:
        while temp_n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            temp_n //= d
        d += 1
    if temp_n > 1:
        factors[temp_n] = factors.get(temp_n, 0) + 1
    return factors


def _mobius_mu(n: int) -> int:
    if n == 1:
        return 1
    prime_factors = _get_prime_factorization(n)
    for p in prime_factors:
        if prime_factors[p] > 1:
            return 0
    return (-1) ** len(prime_factors)


def _get_divisors(n: int) -> list[int]:
    divs = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(list(divs))


def _num_lyndon_words_of_length_k(num_symbols: int, length: int) -> int:
    if length == 0:
        return 0
    if num_symbols == 1:
        return 1 if length == 1 else 0
    divs = _get_divisors(length)
    total_sum = 0
    for d in divs:
        total_sum += int(_mobius_mu(length // d) * (num_symbols**d))
    return total_sum // length


@partial(jax.jit, static_argnames=("depth", "dim"))
def get_log_signature_dim(depth: int, dim: int) -> int:
    """Compute the total dimension of the log-signature space using Witt's formula."""
    return sum(_num_lyndon_words_of_length_k(dim, k) for k in range(1, depth + 1))


def _catalan(n: int) -> int:
    """Return the nth Catalan number C_n."""
    # C_n = (1/(n+1)) * binom(2n, n)
    # Use integer arithmetic to avoid precision issues.
    if n < 0:
        return 0
    if n == 0:
        return 1
    # Compute binomial(2n, n) iteratively
    c = 1
    for k in range(1, n + 1):
        c = c * (n + k) // k
    return c // (n + 1)


def _a000081_upto(max_n: int) -> list[int]:
    """Compute A000081(1..max_n): number of unlabeled rooted trees with n nodes.

    Uses the standard recurrence:
        a(1) = 1
        a(n) = (1/(n-1)) * sum_{k=1}^{n-1} ( sum_{d|k} d * a(d) ) * a(n-k),  n > 1

    Implementation detail:
    - Maintain s[k] = sum_{d|k} d * a(d) incrementally. When a(n) is found,
      we add n * a(n) to all multiples of n in s. This yields overall
      complexity O(N^2 + N log N), which is easily sufficient for signature
      depths used in practice.
    """
    if max_n <= 0:
        return []
    # a[0] unused; work 1-based for clarity
    a: list[int] = [0] * (max_n + 1)
    s: list[int] = [0] * (max_n + 1)
    a[1] = 1
    # d = 1 contributes 1 * a(1) to every k
    for m in range(1, max_n + 1):
        s[m] += 1
    for n in range(2, max_n + 1):
        acc = 0
        for k in range(1, n):
            acc += s[k] * a[n - k]
        a[n] = acc // (n - 1)
        # Update divisor-sum array for new a(n)
        inc = n * a[n]
        for m in range(n, max_n + 1, n):
            s[m] += inc
    # Return [a(1), a(2), ..., a(max_n)]
    return a[1:]


def get_mkw_signature_dim(depth: int, dim: int) -> int:
    """Total dimension of MKW (planar branched) signature coordinates up to depth.

    At level k (degree k), the number of basis elements equals:
        (# plane forests with k nodes) * dim^k
    There is a bijection between plane forests with k nodes and plane trees with k+1 nodes,
    hence the count equals Catalan(k) * dim^k.
    """
    levels: list[int] = []
    for k in range(1, depth + 1):
        levels.append(_catalan(k) * (dim**k))
    return sum(levels)


def get_bck_signature_dim(depth: int, dim: int) -> int:
    """Total dimension of BCK (unordered branched) signature coordinates up to depth.

    At level k (degree k), the number of basis elements equals:
        (# unordered rooted forests with k nodes) * dim^k
    There is a bijection between unordered rooted forests with k nodes and unordered rooted
    trees with k+1 nodes (add a super-root). Denote A000081(n) the number of rooted trees
    with n nodes. Then level-k count = A000081(k+1) * dim^k.
    """
    # Use the standard recurrence for A000081 rather than explicit enumeration.
    if depth <= 0:
        return 0
    counts: list[int] = _a000081_upto(depth + 1)  # counts[k] = A000081(k+1)
    levels: list[int] = []
    for k in range(1, depth + 1):
        levels.append(counts[k] * (dim**k))
    return sum(levels)
