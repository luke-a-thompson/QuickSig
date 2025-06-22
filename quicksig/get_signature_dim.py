import math
from functools import partial

import jax


@partial(jax.jit, static_argnames=("depth", "dim", "flatten"))
def get_signature_dim(depth: int, dim: int, flatten: bool = True) -> int | list[int]:
    """Compute the dimension of the signature space for a given depth and dimension.

    Args:
        depth (int): The depth of the signature.
        dim (int): The dimension of the path space.

    Returns:
        int: The dimension of the signature space.
    """
    if flatten:
        return sum(dim**k for k in range(1, depth + 1))
    else:
        return [dim**k for k in range(1, depth + 1)]


def get_prime_factorization(n: int) -> dict[int, int]:
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


def mobius_mu(n: int) -> int:
    if n == 1:
        return 1
    prime_factors = get_prime_factorization(n)
    for p in prime_factors:
        if prime_factors[p] > 1:
            return 0  # Has a squared prime factor
    return (-1) ** len(prime_factors)  # Product of k distinct primes


def get_divisors(n: int) -> list[int]:
    divs = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(list(divs))


def num_lyndon_words_of_length_k(num_symbols: int, length: int) -> int:
    if length == 0:
        return 0
    if num_symbols == 1:  # Alphabet has only one symbol e.g. "a"
        return 1 if length == 1 else 0  # Only "a" is a Lyndon word, "aa", "aaa" are not.

    divs = get_divisors(length)
    total_sum = 0
    for d in divs:
        total_sum += int(mobius_mu(length // d) * (num_symbols**d))
    return total_sum // length


@partial(jax.jit, static_argnames=("depth", "dim", "flatten"))
def get_log_signature_dim(depth: int, dim: int, flatten: bool = True) -> int | list[int]:
    """Compute the dimension of the log-signature space for a given depth and dimension using Witt's formula.

    Args:
        depth (int): The depth of the log-signature.
        dim (int): The dimension of the path space.

    Returns:
        int: The dimension of the log-signature space.
    """
    if flatten:
        return sum(num_lyndon_words_of_length_k(dim, k) for k in range(1, depth + 1))
    else:
        return [num_lyndon_words_of_length_k(dim, k) for k in range(1, depth + 1)]


if __name__ == "__main__":
    sig_dim = get_signature_dim(5, 2)
    print(sig_dim)
    log_sig_dim = get_log_signature_dim(5, 2)
    print(log_sig_dim)
