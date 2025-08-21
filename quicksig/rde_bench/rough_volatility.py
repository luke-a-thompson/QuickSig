from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import Array
from quicksig.rdes.drivers import bm_driver, correlate_bm_driver_against_reference, riemann_liouville_driver
from typing import Callable
import diffrax as dfx
from diffrax import LinearInterpolation


@dataclass(frozen=True, slots=True)
class BonesiniModelSpec:
    """
    Specification for the Bonesini RDE.

    Args:
        name: Name of the model.
        state: State of the model.
        hurst: Hurst parameter.
        v_0: Forward volatility.
        nu: Vol-of-vol parameter.
        rho: Correlation between price and volatility Brownian motions.
        sigma: Multiplies dW_t in dS.
        g: dt drift in dS.
        tau: Multiplies dX_t in dV.
        varsigma: Multiplies dW_t in dV.
        h: dt drift in dV.

    Raises:
        ValueError: If the parameters are invalid.
    """

    name: str

    hurst: float
    v_0: float
    nu: float | None
    rho: float | None

    sigma: Callable[[Array, Array, float], Array] | None  # Multiplies dW_t in dS
    g: Callable[[Array, Array, float], Array] | None  # dt drift in dS

    tau: Callable[[Array, Array, float], Array] | None  # Multiplies dX_t in dV
    varsigma: Callable[[Array, Array, float], Array] | None  # Multiplies dW_t in dV
    h: Callable[[Array, Array, float], Array] | None  # dt drift in dV

    def __post_init__(self):
        if self.hurst <= 0 or self.hurst >= 1:
            raise ValueError(f"Hurst must be between 0 and 1. Got {self.hurst}")
        if self.v_0 < 0:
            raise ValueError(f"v_0 must be positive. Got {self.v_0}")
        if self.nu is not None and self.nu < 0:
            raise ValueError(f"nu must be non-negative. Got {self.nu}")
        if self.rho is not None and (self.rho < -1 or self.rho > 1):
            raise ValueError(f"rho must be between -1 and 1. Got {self.rho}")


def make_lead_lag_control(ts: jax.Array, X: jax.Array, W: jax.Array) -> dfx.LinearInterpolation:
    """
    Lead–lag construction for the 2D control Z = (X^lag, W).
    Inputs:
      ts: shape (N+1,)
      X, W: shape (N+1,) or (N+1, d) – here 1D per process is fine
    Returns:
      LinearInterpolation on staggered grid τ of length 2N+1 with values Z(τ_m) in R^2.
    """
    ts = jnp.asarray(ts)
    X = jnp.asarray(X).reshape(-1)
    W = jnp.asarray(W).reshape(-1)
    N = ts.shape[0] - 1
    Δ = ts[1] - ts[0]

    # Staggered times: τ_0=t_0; τ_{2k}=t_k, τ_{2k+1}=t_k+Δ/2, τ_{2N}=t_N
    τ_even = ts[:-1]
    τ_mid = ts[:-1] + 0.5 * Δ
    τ = jnp.concatenate([jnp.stack([τ_even, τ_mid], axis=1).reshape(-1), ts[-1:]], axis=0)  # (2N+1,)

    # Values:
    # at τ_{2k}   : (X_{t_k},   W_{t_k})
    # at τ_{2k+1} : (X_{t_k},   W_{t_{k+1}})
    # at τ_{2k+2} : (X_{t_{k+1}}, W_{t_{k+1}})
    Z_even = jnp.stack([X[:-1], W[:-1]], axis=1)  # (N, 2)
    Z_mid = jnp.stack([X[:-1], W[1:]], axis=1)  # (N, 2)
    Z_last = jnp.stack([X[-1], W[-1]])[None, :]  # (1, 2)
    Z = jnp.concatenate([jnp.reshape(jnp.stack([Z_even, Z_mid], axis=1), (2 * N, 2)), Z_last], axis=0)  # (2N+1, 2)

    return dfx.LinearInterpolation(ts=τ, ys=Z)


def build_terms_with_leadlag(model_spec: BonesiniModelSpec, Z_control: dfx.LinearInterpolation):
    def f0(t: float, y: jax.Array, args: tuple[float, float, float]) -> jax.Array:
        """
        ODE terms integrated against time (dt).
        """
        s, v = y[0], y[1]
        price_dt = model_spec.g(s, v, t) if model_spec.g else 0.0
        vol_dt = model_spec.h(s, v, t) if model_spec.h else 0.0
        # Row 0 corresponds to price equation, row 1 to volatility equation.
        # Row 0 receives price drift, row 1 receives volatility drift.
        return jnp.array([price_dt, vol_dt])

    def vf_Z(t: float, y: jax.Array, args: tuple[float, float, float]) -> jax.Array:
        """
        Control terms integrated against the lead-lag control Z.
        """
        s, v = y[0], y[1]
        # columns: [X, W]
        col_X = jnp.array(
            [
                0.0,
                (model_spec.tau(s, v, t) if model_spec.tau else 0.0),
            ]
        )
        col_W = jnp.array(
            [
                (model_spec.sigma(s, v, t) if model_spec.sigma else 0.0),
                (model_spec.varsigma(s, v, t) if model_spec.varsigma else 0.0),
            ]
        )
        # Control matrix [0, sigma; tau, varsigma]. Row 0 corresponds to price equation, row 1 to volatility equation.
        # Price receives 0 * dX, sigma * dW
        # Volatility receives tau * dX, varsigma * dW
        return jnp.stack([col_X, col_W], axis=1)  # (state dim, control dim)

    return dfx.MultiTerm(dfx.ODETerm(f0), dfx.ControlTerm(vf_Z, control=Z_control))


def get_bonesini_rde_params(
    key: jax.Array,
    noise_timesteps: int,
    model_spec: BonesiniModelSpec,
    s_0: float,
) -> tuple[jax.Array, dfx.MultiTerm]:
    """
    Generates a Bonesini RDE path.
    """
    key_W, key_B, key_V = jax.random.split(key, 3)
    ts_noise = jnp.linspace(0.0, 1.0, noise_timesteps + 1)

    # Brownian for price
    W_path = bm_driver(key_W, noise_timesteps, 1)
    W = jnp.squeeze(W_path.path)

    # Second driver (X): choose per model
    if model_spec.name.startswith("Black-Scholes"):
        X = jnp.zeros_like(W)  # dummy, not a stoch vol
    elif model_spec.name.startswith("Bergomi"):
        # X = W^V, correlated with W (price) by rho
        B_path = bm_driver(key_B, noise_timesteps, 1)
        Wv_corr = correlate_bm_driver_against_reference(W_path, B_path, model_spec.rho)
        X = jnp.squeeze(Wv_corr.path)
    elif model_spec.name.startswith("Rough Bergomi"):
        # X = RL integral of a Brownian correlated with W
        B_path = bm_driver(key_B, noise_timesteps, 1)
        W1_corr = correlate_bm_driver_against_reference(W_path, B_path, model_spec.rho)
        X = jnp.squeeze(riemann_liouville_driver(key_V, noise_timesteps, model_spec.hurst, W1_corr).path)
    else:
        raise ValueError("Unknown model for control construction.")

    Z = make_lead_lag_control(ts_noise, X=X, W=W)
    terms = build_terms_with_leadlag(model_spec, Z_control=Z)
    y_0 = jnp.array([s_0, 0.0])

    return y_0, terms


def get_bonesini_noise_drivers(
    key: jax.Array,
    noise_timesteps: int,
    model_spec: BonesiniModelSpec,
    s_0: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Generate initial condition and noise drivers (X, W) for the Bonesini RDE.
    Returns (y0, X, W) as arrays so we can vmap over them easily.
    """
    key_W, key_B, key_V = jax.random.split(key, 3)

    # Brownian for price
    W_path = bm_driver(key_W, noise_timesteps, 1)
    W = jnp.squeeze(W_path.path)

    # Second driver (X): choose per model
    if model_spec.name.startswith("Black-Scholes"):
        X = jnp.zeros_like(W)
    elif model_spec.name.startswith("Bergomi"):
        B_path = bm_driver(key_B, noise_timesteps, 1)
        Wv_corr = correlate_bm_driver_against_reference(W_path, B_path, model_spec.rho)
        X = jnp.squeeze(Wv_corr.path)
    elif model_spec.name.startswith("Rough Bergomi"):
        B_path = bm_driver(key_B, noise_timesteps, 1)
        W1_corr = correlate_bm_driver_against_reference(W_path, B_path, model_spec.rho)
        X = jnp.squeeze(riemann_liouville_driver(key_V, noise_timesteps, model_spec.hurst, W1_corr).path)
    else:
        raise ValueError("Unknown model for control construction.")

    y_0 = jnp.array([s_0, 0.0])
    return y_0, X, W


def solve_bonesini_rde_wong_zakai(y_0: jax.Array, terms: dfx.MultiTerm, noise_timesteps: int, rde_timesteps: int) -> dfx.Solution:
    if noise_timesteps > rde_timesteps:
        raise ValueError("Noise timesteps must be less than or equal to RDE timesteps.")

    ts_noise = jnp.linspace(0.0, 1.0, noise_timesteps + 1)
    ts_rde = jnp.linspace(0.0, 1.0, rde_timesteps + 1)

    solution = dfx.diffeqsolve(
        terms=terms,
        solver=dfx.Heun(),
        t0=0.0,
        t1=1.0,
        dt0=ts_rde[1] - ts_rde[0],
        y0=y_0,
        saveat=dfx.SaveAt(ts=ts_noise),
        stepsize_controller=dfx.ConstantStepSize(),
        max_steps=None,
    )

    return solution


def solve_bonesini_rde_from_drivers(
    y_0: jax.Array,
    X: jax.Array,
    W: jax.Array,
    model_spec: BonesiniModelSpec,
    noise_timesteps: int,
    rde_timesteps: int,
) -> dfx.Solution:
    ts_noise = jnp.linspace(0.0, 1.0, noise_timesteps + 1)
    Z = make_lead_lag_control(ts_noise, X=X, W=W)
    terms = build_terms_with_leadlag(model_spec, Z_control=Z)
    solution = solve_bonesini_rde_wong_zakai(y_0, terms, noise_timesteps, rde_timesteps)
    return solution


def make_black_scholes_model_spec(v_0: float) -> BonesiniModelSpec:
    """
    Makes a Black-Scholes model specification.
    """
    sigma = lambda s, v, t: s * jnp.sqrt(v_0)
    g = lambda s, v, t: -0.5 * s * v_0

    return BonesiniModelSpec(
        name="Black-Scholes",
        hurst=0.5,
        v_0=v_0,
        nu=0.0,
        rho=0.0,
        sigma=sigma,
        g=g,
        tau=None,
        varsigma=None,
        h=None,
    )


def make_bergomi_model_spec(v_0: float, rho: float) -> BonesiniModelSpec:
    """
    Makes a Bergomi model specification.
    """
    rho_bar = jnp.sqrt(1.0 - rho**2)

    sigma = lambda s, v, t: s * jnp.exp(v)
    g = lambda s, v, t: -0.5 * s * (jnp.exp(2 * v) + (rho * v * jnp.exp(v)))
    tau = lambda s, v, t: rho_bar * v
    varsigma = lambda s, v, t: rho * v
    h = lambda s, v, t: -0.5 * v

    return BonesiniModelSpec(
        name="Bergomi",
        hurst=0.5,
        v_0=v_0,
        nu=None,
        rho=rho,
        sigma=sigma,
        g=g,
        tau=tau,
        varsigma=varsigma,
        h=h,
    )


def make_rough_bergomi_model_spec(v_0: float, nu: float, hurst: float, rho: float) -> BonesiniModelSpec:
    # Hybrid RL normalisation: Var[V_t] = t^{2H}  ⇒  C = 2 nu^2

    # PRICE coefficients (not log-price)
    sigma = lambda s, v, t: s * jnp.sqrt(v_0) * jnp.exp(0.5 * nu * v - 0.25 * (nu**2) * (t ** (2.0 * hurst)))
    g = lambda s, v, t: -0.5 * s * v_0 * jnp.exp(nu * v - 0.5 * (nu**2) * (t ** (2.0 * hurst)))
    tau = lambda s, v, t: 1.0  # dV = dX

    return BonesiniModelSpec(
        name="Rough Bergomi",
        hurst=hurst,
        v_0=v_0,
        nu=nu,
        rho=rho,
        sigma=sigma,
        g=g,
        tau=tau,
        varsigma=None,
        h=None,
    )


if __name__ == "__main__":
    from quicksig.rde_bench.plot_rde import plot_bonesini_rde, plot_bonesini_monte_carlo

    black_scholes_model_spec = make_black_scholes_model_spec(v_0=0.04)
    bergomi_model_spec = make_bergomi_model_spec(v_0=0.0, rho=-0.848)
    rough_bergomi_model_spec = make_rough_bergomi_model_spec(v_0=0.04, nu=1.991, hurst=0.25, rho=-0.848)

    noise_timesteps = 1000
    rde_timesteps = 15000

    ## BLACK-SCHOLES
    # Generate drivers for multiple paths
    keys = jax.random.split(jax.random.key(42), 1000)
    y0_bs, X_bs, W_bs = jax.vmap(lambda key: get_bonesini_noise_drivers(key, noise_timesteps, black_scholes_model_spec, s_0=1.0))(keys)

    # Vmap over solving (build terms inside)
    solve_vmap_bs = jax.vmap(lambda y0, X, W: solve_bonesini_rde_from_drivers(y0, X, W, black_scholes_model_spec, noise_timesteps, rde_timesteps))
    solutions_bs = solve_vmap_bs(y0_bs, X_bs, W_bs)
    plot_bonesini_monte_carlo(solutions_bs, black_scholes_model_spec)

    ## BERGOMI
    # Generate drivers for multiple paths
    keys = jax.random.split(jax.random.key(42), 1000)
    y0_b, X_b, W_b = jax.vmap(lambda key: get_bonesini_noise_drivers(key, noise_timesteps, bergomi_model_spec, s_0=1.0))(keys)

    # Vmap over solving
    solve_vmap_b = jax.vmap(lambda y0, X, W: solve_bonesini_rde_from_drivers(y0, X, W, bergomi_model_spec, noise_timesteps, rde_timesteps))
    solutions_b = solve_vmap_b(y0_b, X_b, W_b)
    plot_bonesini_monte_carlo(solutions_b, bergomi_model_spec)

    ## ROUGH BERGOMI
    # Generate drivers for multiple paths
    keys = jax.random.split(jax.random.key(42), 1000)
    y0_rb, X_rb, W_rb = jax.vmap(lambda key: get_bonesini_noise_drivers(key, noise_timesteps, rough_bergomi_model_spec, s_0=100.0))(keys)

    # Vmap over solving
    solve_vmap_rb = jax.vmap(lambda y0, X, W: solve_bonesini_rde_from_drivers(y0, X, W, rough_bergomi_model_spec, noise_timesteps, rde_timesteps))
    solutions_rb = solve_vmap_rb(y0_rb, X_rb, W_rb)
    plot_bonesini_monte_carlo(solutions_rb, rough_bergomi_model_spec, use_log_price=True)
