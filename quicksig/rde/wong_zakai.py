import math


def wz_friz_riedel_meshsize(epsilon: float, hurst: float, eta: float = 1e-8) -> float:
    """
    Friz-Riedel mesh size guaranteeing sup-norm error ≤ epsilon (a.s.).

    Given a partition of [0, T] into N steps, the mesh size is the maximum width of any subinterval
    (not necessarily uniform).

    Source: https://arxiv.org/abs/1108.1099

    Args:
        epsilon: desired error tolerance (0 < epsilon < 1)
        hurst: Hurst index H of the fractional Brownian driver (H > 0.25)
        eta: small slack to keep the exponent positive

    Returns:
        delta: maximum admissible mesh width
    """
    alpha = 2.0 * hurst - 0.5 - eta
    if alpha <= 0:
        raise ValueError("Need H > 0.25 + eta/2 for a positive rate exponent.")
    return epsilon ** (1.0 / alpha)


def wz_friz_riedel_stepcount(epsilon: float, hurst: float, T: float = 1.0, eta: float = 1e-8) -> int:
    """
    Minimum uniform-grid step count to reach epsilon accuracy on [0, T].

    Source: https://arxiv.org/abs/1108.1099

    Args:
        epsilon: desired error tolerance (0 < epsilon < 1)
        hurst: Hurst index H of the fractional Brownian driver (H > 0.25)
        T: length of the interval [0, T]
        eta: small slack to keep the exponent positive

    Returns:
        N: minimum uniform-grid step count
    """
    delta = wz_friz_riedel_meshsize(epsilon, hurst, eta)
    return math.ceil(T / delta)


if __name__ == "__main__":
    epsilon = 1e-3
    hurst = 0.4

    required_mesh_size = wz_friz_riedel_meshsize(epsilon, hurst)
    required_steps = wz_friz_riedel_stepcount(epsilon, hurst)

    print(
        f"""Calculating the Friz-Riedel mesh requirements for the Wong-Zakai ODE approximation to be within {epsilon} of the RDE solution w.r.t the inhomogenous rough path metric given H={hurst}:
            • Target error ε = {epsilon:.1e}, Hurst H = {hurst}
            • Maximum mesh size δ ≈ {required_mesh_size:.2e}
            • Required uniform steps N ≥ {required_steps}
        """
    )
