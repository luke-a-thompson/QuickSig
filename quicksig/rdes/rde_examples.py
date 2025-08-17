from dataclasses import dataclass
import jax
import jax.numpy as jnp
from quicksig.rdes.drivers import bm_driver, correlate_bm_driver_against_reference, riemann_liouville_driver
from quicksig.rdes.rde_types import Path
from typing import Callable, TypedDict
import diffrax as dfx
from diffrax import LinearInterpolation


@dataclass(frozen=True, slots=True)
class BonesiniParams:
    """
    Parameters for the Bonesini RDE.

    Args:
        hurst: Hurst parameter
        v_0: Forward volatility
        eta: Vol-of-vol parameter
        rho: Correlation between price and volatility Brownian motions
    """

    hurst: float
    v_0: float
    eta: float
    rho: float

    def __post_init__(self):
        if self.hurst <= 0 or self.hurst >= 1:
            raise ValueError(f"Hurst must be between 0 and 1. Got {self.hurst}")
        if self.v_0 <= 0:
            raise ValueError(f"v_0 must be positive. Got {self.v_0}")
        if self.eta <= 0:
            raise ValueError(f"eta must be positive. Got {self.eta}")
        if self.rho < -1 or self.rho > 1:
            raise ValueError(f"rho must be between -1 and 1. Got {self.rho}")


v_0 = 0.04


def bonesini_rde(
    key: jax.Array,
    timesteps: int,
    hurst: float,
    rho: float,
) -> dfx.Solution:
    """
    Generates a Bonesini RDE path.
    """
    keys = jax.random.split(key, 4)

    W_path = bm_driver(keys[0], timesteps, 1)
    B_path = bm_driver(keys[1], timesteps, 1)
    W_path_correlated = correlate_bm_driver_against_reference(W_path, B_path, rho)

    V_t_path = riemann_liouville_driver(keys[2], timesteps, hurst, W_path_correlated)
    del B_path

    leading_path = W_path.path
    lagging_path = jnp.concatenate([V_t_path.path[:1], V_t_path.path[:-1]], axis=0)

    ts = jnp.linspace(0.0, 1.0, timesteps + 1)
    leading_path = LinearInterpolation(ts=ts, ys=leading_path)
    lagging_path = LinearInterpolation(ts=ts, ys=lagging_path)

    term = dfx.MultiTerm(
        dfx.ODETerm(bonesini_black_scholes["g"]),
        dfx.ControlTerm(bonesini_black_scholes["sigma"], control=leading_path),
    )

    solver = dfx.Euler()
    solution = dfx.diffeqsolve(
        terms=term,
        solver=solver,
        t0=0.0,
        t1=1.0,
        dt0=1.0 / timesteps,
        y0=1.0,
        saveat=dfx.SaveAt(ts=jnp.linspace(0.0, 1.0, timesteps + 1)),
        max_steps=None,
    )

    return solution


bonesini_black_scholes = {
    "sigma": lambda s_t, v_t, t: (s_t * 1) * jnp.sqrt(v_t),
    "g": lambda s_t, v_t, t: -0.5 * s_t * v_0,
    "tau": lambda s_t, v_t, t: 0.0,
    "varsigma": lambda s_t, v_t, t: 0.0,
    "h": lambda s_t, v_t, t: 0.0,
}

if __name__ == "__main__":
    key = jax.random.key(42)
    timesteps = 1000
    hurst = 0.25
    rho = 0.0
    solution = bonesini_rde(key, timesteps, hurst, rho)
    print(solution.ys)
