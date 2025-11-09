import math


def hurst_to_wz_friz_riedel_meshsize(epsilon: float, hurst: float, eta: float = 1e-8) -> float:
    """
    Friz-Riedel mesh size guaranteeing sup-norm error ≤ epsilon (a.s.).
    """
    alpha = 2.0 * hurst - 0.5 - eta
    if alpha <= 0:
        raise ValueError("Need H > 0.25 + eta/2 for a positive rate exponent.")
    return epsilon ** (1.0 / alpha)


def hurst_to_wz_friz_riedel_stepcount(
    epsilon: float, hurst: float, T: float = 1.0, eta: float = 1e-8
) -> int:
    """
    Minimum uniform-grid step count to reach epsilon accuracy on [0, T].
    """
    delta = hurst_to_wz_friz_riedel_meshsize(epsilon, hurst, eta)
    return math.ceil(T / delta)
