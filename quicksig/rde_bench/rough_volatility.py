from dataclasses import dataclass
import jax
import jax.numpy as jnp
from quicksig.rdes.drivers import bm_driver, correlate_bm_driver_against_reference, riemann_liouville_driver
from quicksig.rdes.rde_types import Path
from typing import Callable, Literal, Optional
import diffrax as dfx
from diffrax import LinearInterpolation
import matplotlib.pyplot as plt


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
    state: Literal["Price", "PriceVol"]  # Price only vs coupled price and volatility
    v_input: Literal["constant", "lagged_RL", "state"]
    needs_X_control: bool

    hurst: float
    v_0: float
    nu: float | None
    rho: float | None

    sigma: Optional[Callable[[float, float, float], float]]  # Multiplies dW_t in dS
    g: Optional[Callable[[float, float, float], float]]  # dt drift in dS

    # Only used for PriceVol state
    tau: Optional[Callable[[float, float, float], float]]  # Multiplies dX_t in dV
    varsigma: Optional[Callable[[float, float, float], float]]  # Multiplies dW_t in dV
    h: Optional[Callable[[float, float, float], float]]  # dt drift in dV

    def __post_init__(self):
        if self.hurst <= 0 or self.hurst >= 1:
            raise ValueError(f"Hurst must be between 0 and 1. Got {self.hurst}")
        if self.v_0 < 0:
            raise ValueError(f"v_0 must be positive. Got {self.v_0}")
        if self.nu is not None and self.nu < 0:
            raise ValueError(f"nu must be non-negative. Got {self.nu}")
        if self.rho is not None and (self.rho < -1 or self.rho > 1):
            raise ValueError(f"rho must be between -1 and 1. Got {self.rho}")

def build_terms(
        model_spec: BonesiniModelSpec,
        W_control: dfx.AbstractPath,
        v_eval: Callable[[float], float],
        X_control: dfx.AbstractPath | None,
        ):
    """
    Builds diffrax terms for the Bonesini RDE.

    Args:
        model_spec: Specification for the Bonesini RDE.
        W_control: Control for the W Brownian motion.
        v_eval: Function to evaluate the volatility at a given time.
        X_control: Control for the X Brownian motion.

    Returns:
        - For state "Price" we return terms with ControlTerm(W) + ODETerm
        - For state "PriceVol" we return terms with ControlTerm(W) + ControlTerm(X) + ODETerm
    """
    match model_spec.state:
        case "Price":
            def vf_W(t: float, y: float, args: tuple[float, float, float]) -> float:
                """
                Vector field integrated against the W Brownian path (dW).
                """
                s = y
                v = v_eval(t)
                return model_spec.sigma(s, v, t) if model_spec.sigma is not None else 0.0
            def f0(t: float, y: float, args: tuple[float, float, float]) -> float:
                """
                Vector field integrated against time (dt). This is the drift term.
                """
                s = y
                v = v_eval(t)
                return model_spec.g(s, v, t) if model_spec.g is not None else 0.0
            
            terms = [dfx.ODETerm(f0), dfx.ControlTerm(vf_W, control=W_control)]
            return dfx.MultiTerm(*terms)
        case "PriceVol":
            def vf_W(t: float, y: jax.Array, args: tuple[float, float, float]) -> jax.Array:
                """
                Vector fields integrated against the W Brownian path (dW).
                """
                s, v_state = y[0], y[1]
                v = v_state
                price_term = model_spec.sigma(s, v, t) if model_spec.sigma is not None else 0.0
                vol_term = model_spec.varsigma(s, v, t) if model_spec.varsigma is not None else 0.0
                return jnp.array([price_term, vol_term])
            def vf_X(t: float, y: jax.Array, args: tuple[float, float, float]) -> jax.Array:
                """
                Vector fields integrated against the X Brownian path (dX).
                """
                s, v_state = y[0], y[1]
                v = v_state
                price_term = 0.0
                vol_term = model_spec.tau(s, v, t) if model_spec.tau is not None else 0.0
                return jnp.array([price_term, vol_term])
            def f0(t: float, y: jax.Array, args: tuple[float, float, float]) -> jax.Array:
                """
                Vector field integrated against time (dt). This is the drift term.
                """
                s, v_state = y[0], y[1]
                v = v_state
                price_term = model_spec.g(s, v, t) if model_spec.g is not None else 0.0
                vol_term = model_spec.h(s, v, t) if model_spec.h is not None else 0.0
                return jnp.array([price_term, vol_term])
            
            terms = [dfx.ODETerm(f0), dfx.ControlTerm(vf_W, control=W_control)]
            if model_spec.needs_X_control:
                if X_control is None:
                    raise ValueError("State=PriceVol with needs_X_control=True requires an X_control interpolation path.")
                terms.append(dfx.ControlTerm(vf_X, control=X_control))
            return dfx.MultiTerm(*terms)
        case _:
            raise ValueError(f"Invalid state: {model_spec.state}. Choose from 'Price' or 'PriceVol'.")


def bonesini_rde(
    key: jax.Array,
    timesteps: int,
    model_spec: BonesiniModelSpec,
    S0: float,
) -> dfx.Solution:
    """
    Generates a Bonesini RDE path.
    """
    key_W, key_B, key_V = jax.random.split(key, 3)

    ts = jnp.linspace(0.0, 1.0, timesteps + 1)
    delta_t = float(ts[1] - ts[0])

    # Leading Brownian W_t for S_t
    W_path = bm_driver(key_W, timesteps, 1)
    W_interp_leading = LinearInterpolation(ts=ts, ys=jnp.squeeze(W_path.path))

    X_interp = None
    V_interp_lagged = None
    if model_spec.v_input in ["lagged_RL", "state"]:
        B_path = bm_driver(key_B, timesteps, 1)
        if model_spec.hurst == 0.5:
            X_path = jnp.squeeze(B_path.path)
        else:
            W_path_correlated = correlate_bm_driver_against_reference(W_path, B_path, model_spec.rho)
            X_path = jnp.squeeze(riemann_liouville_driver(key_V, timesteps, model_spec.hurst, W_path_correlated).path)
        X_interp = LinearInterpolation(ts=ts, ys=X_path)
        if model_spec.v_input == "lagged_RL":
            V_path_lagged = jnp.concatenate([X_path[:1], X_path[:-1]], axis=0)
            V_interp_lagged = LinearInterpolation(ts=ts, ys=V_path_lagged)

    if model_spec.state == "Price":
        match model_spec.v_input:
            case "constant":
                v_eval = lambda t: model_spec.v_0
            case "lagged_RL":
                assert V_interp_lagged is not None, "V_interp_lagged is not set."
                v_eval = lambda t: V_interp_lagged.evaluate(t)
            case _:
                raise ValueError(f"For state='price', v_input must be 'constant' or 'lagged_RL'. Invalid v_input: {model_spec.v_input}.")
    else:
        v_eval = lambda t: jnp.array(0.0)

    match model_spec.v_input:
        case "constant":
            v_eval = lambda t: model_spec.v_0
        case "lagged_RL":
            assert V_interp_lagged is not None, "V_interp_lagged is not set."
            v_eval = lambda t: V_interp_lagged.evaluate(t)
        case "state":
            v_eval = lambda t: jnp.array(0.0)
        case _:
            raise ValueError(f"Invalid v_input: {model_spec.v_input}. Choose from 'constant', 'lagged_RL', or 'state'.")

    term = build_terms(model_spec=model_spec, W_control=W_interp_leading, v_eval=v_eval, X_control=X_interp)
    y_0 = S0 if model_spec.state == "Price" else jnp.array([S0, model_spec.v_0])

    solution = dfx.diffeqsolve(
        terms=term,
        solver=dfx.Heun(),
        t0=0.0,
        t1=1.0,
        dt0=delta_t,
        y0=y_0,
        saveat=dfx.SaveAt(ts=ts),
        stepsize_controller=dfx.ConstantStepSize(),
        max_steps=None,
    )

    return solution


def make_black_scholes_model_spec(v_0: float) -> BonesiniModelSpec:
    """
    Makes a Black-Scholes model specification.
    """
    return BonesiniModelSpec(
        name="Black-Scholes",
        state="Price",
        v_input="constant",
        needs_X_control=False,
        hurst = 0.5,
        v_0 = v_0,
        nu = 0.0,
        rho = 0.0,
        sigma = lambda s_t, v_t, t: (s_t * 1) * jnp.sqrt(v_t),
        g = lambda s_t, v_t, t: -0.5 * s_t * v_0,
        tau = lambda s_t, v_t, t: 0.0,
        varsigma = lambda s_t, v_t, t: 0.0,
        h = lambda s_t, v_t, t: 0.0,
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
        state="PriceVol",
        v_input="state",
        needs_X_control=True,
        hurst = 0.5,
        v_0 = v_0,
        nu = None,
        rho = rho,
        sigma = sigma,
        g = g,
        tau = tau,
        varsigma = varsigma,
        h = h,
    )

def make_rough_bergomi_model_spec(v_0: float, nu: float, hurst: float, rho: float) -> BonesiniModelSpec:
    gamma_term = jax.scipy.special.gamma(hurst + 0.5)
    C = (nu**2) / (gamma_term**2)

    sigma = lambda s, v, t: s * jnp.sqrt(v_0) * jnp.exp(nu * v -0.5 * C * (t**(2.0 * hurst)))
    g = lambda s, v, t: -0.5 * s * v_0 * jnp.exp((2.0 * nu * v) - C * (t ** (2.0 * hurst)))

    return BonesiniModelSpec(
        name="Rough Bergomi",
        state="Price",
        v_input="lagged_RL",
        needs_X_control=False,
        hurst = hurst,
        v_0 = v_0,
        nu = nu,
        rho = rho,
        sigma = sigma,
        g = g,
        tau = None,
        varsigma = None,
        h = None,
    )

if __name__ == "__main__":
    from quicksig.rde_bench.plot_rde import plot_bonesini_rde, plot_bonesini_monte_carlo

    black_scholes_model_spec = make_black_scholes_model_spec(v_0=0.04)
    bergomi_model_spec = make_bergomi_model_spec(v_0=0.0, rho=-0.848)
    rough_bergomi_model_spec = make_rough_bergomi_model_spec(v_0=0.04, nu=1.991, hurst=0.25, rho=-0.848)

    key = jax.random.key(43)
    timesteps = 5000

    solution_bs = bonesini_rde(key, timesteps, black_scholes_model_spec, S0=1.0)
    solution_b = bonesini_rde(key, timesteps, bergomi_model_spec, S0=1.0)

    solution_rb = bonesini_rde(key, timesteps, rough_bergomi_model_spec, S0=1.0)

    solutions = [solution_bs, solution_b, solution_rb]
    model_specs = [black_scholes_model_spec, bergomi_model_spec, rough_bergomi_model_spec]
    plot_bonesini_rde(solutions, model_specs)

    # Monte Carlo simulation over rough Bergomi, 100 different seeds, plot separately
    keys = jax.random.split(jax.random.key(42), 1000)
    bonesini_rde_vmap_rb = jax.vmap(lambda key: bonesini_rde(key, timesteps, rough_bergomi_model_spec, S0=1.0))
    solutions_rb = bonesini_rde_vmap_rb(keys)
    plot_bonesini_monte_carlo(solutions_rb, rough_bergomi_model_spec)

    # Monte Carlo simulation over Bergomi to observe the log-normal distribution
    keys_b = jax.random.split(jax.random.key(44), 1000)
    bonesini_rde_vmap_b = jax.vmap(lambda key: bonesini_rde(key, timesteps, bergomi_model_spec, S0=1.0))
    solutions_b = bonesini_rde_vmap_b(keys_b)
    plot_bonesini_monte_carlo(solutions_b, bergomi_model_spec)

    # # Monte Carlo simulation over Black-Scholes to observe the log-normal distribution
    # keys_bs = jax.random.split(jax.random.key(43), 1000) # Use more paths for a clearer distribution
    # bonesini_rde_vmap_bs = jax.vmap(lambda key: bonesini_rde(key, timesteps, black_scholes_model_spec, S0=1.0))
    # solutions_bs = bonesini_rde_vmap_bs(keys_bs)
    # plot_bonesini_monte_carlo(solutions_bs, black_scholes_model_spec)