"""
This module contains functions for computing metrics and norms of signature levels.

In general we are concerned with three indices:
1. 𝛼 is the Hölder exponent of the signature level. 𝛼 = 1/p
2. p is the p-variation index. p = 1/𝛼.
3. H is the Hurst index. p > 1/H and 𝛼 < H. Thus, p > 1/a > 1/H
"""

import math


def get_holder_alpha(H: float, epsilon: float = 0.01) -> float:
    """
    Compute a valid Hölder exponent α = H - ε, ensuring α > 0.

    Args:
        H: True Hölder (or Hurst) exponent, must be > 0.
        epsilon: Small safety margin; default is 0.01.

    Returns:
        α = H - ε, guaranteed > 0.

    Raises:
        ValueError: If H ≤ 0 or H - ε ≤ 0.
    """
    if H <= 0:
        raise ValueError(f"H must be positive. Got H={H}.")
    if H > 1 / 2:
        return 1  # Young integration suffices
    alpha = H - epsilon
    if alpha <= 0:
        raise ValueError(f"Hölder exponent (H - ε) must be positive. Got α={alpha}.")
    return alpha


def get_minimal_signature_depth(H: float, epsilon: float = 0.01) -> int:
    """
    Compute the minimal signature depth N so that N * α > 1.

    Source: Definition 9.4, page 188, Friz & Vectoir 2010, Multidimensional Stochastic Processes as Rough Paths

    Args:
        H: Hurst exponent, must be > 0.
        epsilon: Small safety margin; default is 0.01.

    Returns:
        N: Minimal signature depth N so that N * α > 1.

    Raises:
    """
    alpha = get_holder_alpha(H, epsilon)
    return math.floor(1 / alpha)
