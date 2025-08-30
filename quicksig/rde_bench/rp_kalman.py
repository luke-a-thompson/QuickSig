import jax
import jax.numpy as jnp
from jax import lax
import diffrax as dfx
from typing import cast
from quicksig.rde_bench.rough_volatility import (
    make_rough_bergomi_model_spec,
    build_terms_with_leadlag,
)
from dataclasses import dataclass, replace

key = jax.random.key(42)

jax.config.update("jax_debug_nans", True)


@jax.tree_util.register_pytree_node_class
@dataclass(slots=True)
class RBergomiEnsemble:
    # State (batched across N particles)
    S: jax.Array  # shape (N,)
    V: jax.Array  # shape (N,)

    X_driver: jax.Array
    W_driver: jax.Array

    # Unconstrained parameters per particle (batched)
    tH: jax.Array  # logit(H),   shape (N,)
    tNu: jax.Array  # log(eta),   shape (N,)
    tRho: jax.Array  # atanh(rho), shape (N,)
    tV0: jax.Array  # log(v0),    shape (N,)

    def tree_flatten(self):
        leaves = (self.S, self.V, self.X_driver, self.W_driver, self.tH, self.tNu, self.tRho, self.tV0)
        aux = None
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        S, V, X_driver, W_driver, tH, tNu, tRho, tV0 = leaves
        return cls(S, V, X_driver, W_driver, tH, tNu, tRho, tV0)

    def replace(self, **kwargs):
        return replace(self, **kwargs)


def constrain_theta(tH: jax.Array, tNu: jax.Array, tRho: jax.Array, tV0: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    H = jax.nn.sigmoid(tH)
    eta = jnp.exp(tNu)
    rho = jnp.tanh(tRho)
    v0 = jnp.exp(tV0)
    return H, eta, rho, v0


def unconstrain_theta(H: jax.Array, eta: jax.Array, rho: jax.Array, v0: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    tH = jnp.log(H / (1.0 - H))
    tNu = jnp.log(eta)
    tRho = jnp.arctanh(rho)
    tV0 = jnp.log(v0)
    return tH, tNu, tRho, tV0


def compute_fractional_brownian_increment(
    hurst_param: float | jax.Array,
    volatility_increments: jax.Array,
    time_grid: jax.Array,
    time_index: int | jax.Array,
    auxiliary_noise: jax.Array | None = None,
) -> jax.Array:
    """
    Compute one-step increment for Riemann-Liouville fractional Brownian motion.

    Parameters
    ----------
    hurst_param : float | jax.Array
        Hurst parameter H ∈ (0,1)
    volatility_increments : jax.Array
        Brownian increments for volatility driver, shape (K,)
    time_grid : jax.Array
        Discretized time points, shape (K+1,)
    time_index : int | jax.Array
        Current time step index k ∈ [0, K-1]
    auxiliary_noise : jax.Array | None, optional
        Additional noise terms for hybrid scheme, by default None

    Returns
    -------
    jax.Array
        Fractional Brownian increment ΔY_k = Y_{k+1} - Y_k
    """
    time_step = time_grid[1] - time_grid[0]
    alpha = hurst_param - 0.5

    integral_coeff = jnp.power(time_step, alpha) / (alpha + 1.0)
    integral_variance = jnp.power(time_step, 2.0 * alpha + 1.0) / (2.0 * alpha + 1.0)
    noise_coeff = jnp.sqrt(jnp.maximum(integral_variance - (integral_coeff * integral_coeff) * time_step, 0.0))

    current_index = jnp.asarray(time_index, dtype=jnp.int32)
    total_steps = volatility_increments.shape[0]

    next_increment = volatility_increments[current_index]
    prev_increment = jnp.where(current_index > 0, volatility_increments[current_index - 1], jnp.asarray(0.0, volatility_increments.dtype))
    if auxiliary_noise is None:
        aux_increment = jnp.asarray(0.0, volatility_increments.dtype)
    else:
        next_aux = auxiliary_noise[current_index]
        prev_aux = jnp.where(current_index > 0, auxiliary_noise[current_index - 1], jnp.asarray(0.0, auxiliary_noise.dtype))
        aux_increment = next_aux - prev_aux
    integral_increment = integral_coeff * (next_increment - prev_increment) + noise_coeff * aux_increment

    historical_indices = jnp.arange(1, total_steps + 1, dtype=jnp.int32)
    historical_float = historical_indices.astype(time_grid.dtype)

    weight_current = jnp.where(
        historical_indices >= 2,
        (jnp.power(historical_float, alpha + 1.0) - jnp.power(historical_float - 1.0, alpha + 1.0)) * (jnp.power(time_step, alpha) / (alpha + 1.0)),
        0.0,
    )
    weight_next_float = historical_float + 1.0
    weight_next = (jnp.power(weight_next_float, alpha + 1.0) - jnp.power(historical_float, alpha + 1.0)) * (jnp.power(time_step, alpha) / (alpha + 1.0))
    weight_coefficients = weight_next - weight_current

    history_indices = current_index - historical_indices
    valid_mask = (historical_indices <= current_index).astype(volatility_increments.dtype)
    safe_indices = jnp.clip(history_indices, 0, total_steps - 1)
    historical_contribution = jnp.sum(weight_coefficients * volatility_increments[safe_indices] * valid_mask)

    return jnp.sqrt(2.0 * hurst_param) * (integral_increment + historical_contribution)


def compute_correlated_price_increment(correlation: float | jax.Array, volatility_increment: jax.Array, independent_increment: jax.Array) -> jax.Array:
    """
    Compute correlated Brownian increment for price process.

    Parameters
    ----------
    correlation : float | jax.Array
        Correlation coefficient ρ ∈ (-1,1) between price and volatility
    volatility_increment : jax.Array
        Brownian increment for volatility process
    independent_increment : jax.Array
        Independent Brownian increment

    Returns
    -------
    jax.Array
        Correlated increment for price Brownian motion
    """
    return correlation * volatility_increment + jnp.sqrt(1.0 - correlation**2) * independent_increment


def create_leadlag_control(
    start_time: float | jax.Array,
    end_time: float | jax.Array,
    initial_vol_driver: jax.Array,
    vol_driver_increment: jax.Array,
    initial_price_driver: jax.Array,
    price_driver_increment: jax.Array,
) -> tuple[dfx.LinearInterpolation, jax.Array, jax.Array]:
    """
    Create lead-lag control path for rough path integration.

    Parameters
    ----------
    start_time, end_time : float | jax.Array
        Time interval [t0, t1]
    initial_vol_driver, vol_driver_increment : jax.Array
        Initial value and increment for volatility driver
    initial_price_driver, price_driver_increment : jax.Array
        Initial value and increment for price driver

    Returns
    -------
    control_path : dfx.LinearInterpolation
        Lead-lag control path for rough path integration
    final_vol_driver, final_price_driver : jax.Array
        Updated driver values
    """
    mid_time = 0.5 * (start_time + end_time)
    time_points = jnp.array([start_time, mid_time, end_time])
    final_vol_driver = initial_vol_driver + vol_driver_increment
    final_price_driver = initial_price_driver + price_driver_increment
    control_values = jnp.stack(
        [jnp.array([initial_vol_driver, initial_price_driver]), jnp.array([initial_vol_driver, final_price_driver]), jnp.array([final_vol_driver, final_price_driver])], axis=0
    )
    return dfx.LinearInterpolation(ts=time_points, ys=control_values), final_vol_driver, final_price_driver


def evolve_single_particle(
    current_price: jax.Array,
    current_volatility: jax.Array,
    hurst_param: jax.Array,
    vol_of_vol: jax.Array,
    correlation: jax.Array,
    initial_vol: jax.Array,
    vol_driver_state: jax.Array,
    price_driver_state: jax.Array,
    volatility_increments: jax.Array,
    independent_increment: jax.Array,
    time_grid: jax.Array,
    time_index: int | jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Evolve a single particle through one time step using rough Bergomi dynamics.

    Parameters
    ----------
    current_price, current_volatility : jax.Array
        Current state values
    hurst_param : jax.Array
        Hurst parameter H
    vol_of_vol : jax.Array
        Volatility of volatility parameter ν
    correlation : jax.Array
        Price-volatility correlation ρ
    initial_vol : jax.Array
        Initial volatility v0
    vol_driver_state, price_driver_state : jax.Array
        Current driver process states
    volatility_increments : jax.Array
        Brownian increments for volatility, shape (K,)
    independent_increment : jax.Array
        Independent Brownian increment for this step
    time_grid : jax.Array
        Time discretization
    time_index : int | jax.Array
        Current time step

    Returns
    -------
    next_price, next_volatility : jax.Array
        Updated state values
    next_vol_driver, next_price_driver : jax.Array
        Updated driver states
    """
    vol_driver_increment = compute_fractional_brownian_increment(hurst_param, volatility_increments, time_grid, time_index)
    price_driver_increment = compute_correlated_price_increment(correlation, volatility_increments[time_index], independent_increment)
    control_path, next_vol_driver, next_price_driver = create_leadlag_control(
        time_grid[time_index],
        time_grid[time_index + 1],
        vol_driver_state,
        vol_driver_increment,
        price_driver_state,
        price_driver_increment,
    )

    model_spec = make_rough_bergomi_model_spec(
        v_0=cast(float, initial_vol),
        nu=cast(float, vol_of_vol),
        hurst=cast(float, hurst_param),
        rho=cast(float, correlation),
    )
    sde_terms = build_terms_with_leadlag(model_spec, Z_control=control_path)

    initial_state = jnp.array([current_price, current_volatility])
    ode_solver = dfx.Heun()
    solver_state = ode_solver.init(sde_terms, time_grid[time_index], time_grid[time_index + 1], y0=initial_state, args=None)
    final_state, _, _, solver_state, _ = ode_solver.step(
        sde_terms, time_grid[time_index], time_grid[time_index + 1], initial_state, args=None, solver_state=solver_state, made_jump=False
    )
    next_price, next_volatility = final_state[0], final_state[1]

    return next_price, next_volatility, next_vol_driver, next_price_driver


evolve_particle_ensemble = jax.vmap(
    evolve_single_particle,
    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None),
    out_axes=(0, 0, 0, 0),
)


def compute_ensemble_kalman_gains(
    unconstrained_params: jax.Array,
    predicted_prices: jax.Array,
    predicted_volatilities: jax.Array,
    predicted_observations: jax.Array,
    observation_noise_variance: float,
    regularization: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Compute Ensemble Kalman Filter gain matrices.

    Parameters
    ----------
    unconstrained_params : jax.Array
        Unconstrained parameter matrix, shape (N, 4) with columns [tH, tNu, tRho, tV0]
    predicted_prices : jax.Array
        Predicted price values, shape (N,)
    predicted_volatilities : jax.Array
        Predicted volatility values, shape (N,)
    predicted_observations : jax.Array
        Predicted observations (log-prices), shape (N,)
    observation_noise_variance : float
        Observation noise variance R
    regularization : float, optional
        Tikhonov regularization parameter, by default 1e-6

    Returns
    -------
    parameter_gain : jax.Array
        Kalman gain for unconstrained parameters, shape (4,)
    state_gain : jax.Array
        Kalman gain for state variables [S, V], shape (2,)
    innovation_variance : jax.Array
        Innovation covariance (scalar)
    """
    num_particles = predicted_observations.shape[0]

    observation_mean = jnp.mean(predicted_observations)
    innovation_deviations = predicted_observations - observation_mean
    innovation_variance = (innovation_deviations @ innovation_deviations) / (num_particles - 1.0) + observation_noise_variance + regularization

    param_mean = jnp.mean(unconstrained_params, axis=0)
    param_deviations = unconstrained_params - param_mean
    param_innovation_covariance = (param_deviations.T @ innovation_deviations) / (num_particles - 1.0)
    parameter_gain = param_innovation_covariance / innovation_variance

    assert num_particles > 1, "Need more than one particle for state gain computation"
    state_matrix = jnp.stack([predicted_prices, predicted_volatilities], axis=1)
    state_mean = jnp.mean(state_matrix, axis=0)
    state_deviations = state_matrix - state_mean
    state_innovation_covariance = (state_deviations.T @ innovation_deviations) / (num_particles - 1.0)
    state_gain = state_innovation_covariance / innovation_variance

    return parameter_gain, state_gain, innovation_variance


def ensemble_kalman_filter_step(
    filter_state: tuple[RBergomiEnsemble, float, jax.Array, jax.Array, jax.Array, jax.Array],
    time_index: int | jax.Array,
) -> tuple[tuple[RBergomiEnsemble, float, jax.Array, jax.Array, jax.Array, jax.Array], dict[None, None]]:
    """
    Perform one step of the Ensemble Kalman Filter.

    Parameters
    ----------
    filter_state : tuple
        Current filter state containing (ensemble, R, time_grid, observations, vol_increments, indep_increments)
    time_index : int | jax.Array
        Current time step index

    Returns
    -------
    updated_state : tuple
        Updated filter state
    info : dict
        Additional information (empty)
    """
    ensemble, obs_noise_var, time_grid, observations, vol_increments, indep_increments = filter_state
    hurst_params, vol_of_vol_params, correlation_params, initial_vol_params = constrain_theta(ensemble.tH, ensemble.tNu, ensemble.tRho, ensemble.tV0)

    next_prices, next_volatilities, next_vol_drivers, next_price_drivers = evolve_particle_ensemble(
        ensemble.S,
        ensemble.V,
        hurst_params,
        vol_of_vol_params,
        correlation_params,
        initial_vol_params,
        ensemble.X_driver,
        ensemble.W_driver,
        vol_increments,
        indep_increments[time_index],
        time_grid,
        time_index,
    )
    predicted_observations = jnp.log(jnp.clip(next_prices, 1e-12))
    unconstrained_param_matrix = jnp.stack([ensemble.tH, ensemble.tNu, ensemble.tRho, ensemble.tV0], axis=1)
    param_gain, state_gain, _ = compute_ensemble_kalman_gains(unconstrained_param_matrix, next_prices, next_volatilities, predicted_observations, obs_noise_var, 1e-5)

    innovations = observations[time_index + 1] - predicted_observations

    param_updates = innovations[:, None] * param_gain[None, :]
    updated_tH = ensemble.tH + param_updates[:, 0]
    updated_tNu = ensemble.tNu + param_updates[:, 1]
    updated_tRho = ensemble.tRho + param_updates[:, 2]
    updated_tV0 = ensemble.tV0 + param_updates[:, 3]

    state_updates = innovations[:, None] * state_gain[None, :]
    updated_prices = jnp.clip(next_prices + state_updates[:, 0], 1e-12, 1e12)
    updated_volatilities = jnp.clip(next_volatilities + state_updates[:, 1], 1e-12, 1e12)

    updated_ensemble = RBergomiEnsemble(
        S=updated_prices,
        V=updated_volatilities,
        X_driver=next_vol_drivers,
        W_driver=next_price_drivers,
        tH=updated_tH,
        tNu=updated_tNu,
        tRho=updated_tRho,
        tV0=updated_tV0,
    )
    return (updated_ensemble, obs_noise_var, time_grid, observations, vol_increments, indep_increments), {}


def run_ensemble_kalman_filter(
    initial_ensemble: RBergomiEnsemble,
    time_grid: jax.Array,
    observations: jax.Array,
    observation_noise_variance: float,
    volatility_increments: jax.Array,
    independent_increments: jax.Array,
) -> RBergomiEnsemble:
    """
    Run the complete Ensemble Kalman Filter for parameter estimation.

    Parameters
    ----------
    initial_ensemble : RBergomiEnsemble
        Initial ensemble of particles
    time_grid : jax.Array
        Time discretization, shape (K+1,)
    observations : jax.Array
        Log-price observations, shape (K+1,)
    observation_noise_variance : float
        Observation noise variance R
    volatility_increments : jax.Array
        Brownian increments for volatility driver, shape (K,)
    independent_increments : jax.Array
        Independent Brownian increments, shape (K,)

    Returns
    -------
    RBergomiEnsemble
        Final ensemble after filtering
    """
    num_steps = time_grid.shape[0] - 1
    initial_state = (initial_ensemble, observation_noise_variance, time_grid, observations, volatility_increments, independent_increments)
    (final_state, _) = lax.scan(ensemble_kalman_filter_step, initial_state, jnp.arange(num_steps))
    final_ensemble, *_ = final_state
    return final_ensemble


@jax.jit
def simulate_observation_logS(
    S0: float,
    V0: float,
    H: float,
    eta: float,
    rho: float,
    v0: float,
    t: jax.Array,
    dW_V: jax.Array,
    dW_perp: jax.Array,
) -> jax.Array:
    K = t.shape[0] - 1
    solver = dfx.Heun()
    spec = make_rough_bergomi_model_spec(v_0=v0, nu=eta, hurst=H, rho=rho)

    def step(carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array], k: jax.Array) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
        S, V, X_drv, W_drv = carry
        vol_driver_increment = compute_fractional_brownian_increment(H, dW_V, t, k)
        price_driver_increment = compute_correlated_price_increment(rho, dW_V[k], dW_perp[k])
        control, X_next, W_next = create_leadlag_control(t[k], t[k + 1], X_drv, vol_driver_increment, W_drv, price_driver_increment)
        terms = build_terms_with_leadlag(spec, Z_control=control)
        y0 = jnp.array([S, V], dtype=jnp.float32)
        solver_state = solver.init(terms, t[k], t[k + 1], y0=y0, args=None)
        y1, _, _, solver_state, _ = solver.step(terms, t[k], t[k + 1], y0, args=None, solver_state=solver_state, made_jump=False)
        S1, V1 = y1[0], y1[1]
        y_log = jnp.log(jnp.clip(S1, 1e-12))
        return (S1, V1, X_next, W_next), y_log

    carry0 = (jnp.asarray(S0, dtype=jnp.float32), jnp.asarray(V0, dtype=jnp.float32), jnp.asarray(0.0, dtype=t.dtype), jnp.asarray(0.0, dtype=t.dtype))
    _, y_logs = lax.scan(step, carry0, jnp.arange(K))
    y0 = jnp.log(jnp.clip(jnp.asarray(S0, dtype=jnp.float32), 1e-12))
    return jnp.concatenate([y0[None], y_logs], axis=0)


def sample_unconstrained_priors(random_key: jax.Array, num_particles: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Sample weakly-informative priors in unconstrained parameter space.

    Parameters
    ----------
    random_key : jax.Array
        JAX random key for sampling
    num_particles : int
        Number of particles in ensemble

    Returns
    -------
    tH_samples, tNu_samples, tRho_samples, tV0_samples : jax.Array
        Unconstrained parameter samples, each with shape (num_particles,)
        tH: logit(H), tNu: log(ν), tRho: atanh(ρ), tV0: log(v0)
    """
    key_H, key_Nu, key_Rho, key_V0 = jax.random.split(random_key, 4)

    hurst_constrained = jax.random.uniform(key_H, (num_particles,), minval=0.05, maxval=0.45)
    tH_samples = jnp.log(hurst_constrained) - jnp.log1p(-hurst_constrained)

    tNu_samples = jax.random.uniform(key_Nu, (num_particles,), minval=jnp.log(0.3), maxval=jnp.log(3.0))

    correlation_constrained = jax.random.uniform(key_Rho, (num_particles,), minval=-0.95, maxval=0.95)
    tRho_samples = jnp.arctanh(correlation_constrained)

    tV0_samples = jax.random.uniform(key_V0, (num_particles,), minval=jnp.log(0.005), maxval=jnp.log(0.3))

    return tH_samples, tNu_samples, tRho_samples, tV0_samples


# ---- utility: run a single estimation for a given PRNG key ----
def run_single_estimation_with_key(
    base_key: jax.Array,
    K: int,
    N: int,
    R: float,
    H_star: float,
    nu_star: float,
    rho_star: float,
    v0_star: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Run one end-to-end estimation using the provided PRNG key.

    Returns a tuple of two dicts with parameter means (keys "H", "nu", "rho", "v0"):
    (enkf_estimates, prior_means).
    """
    # Time grid
    t = jnp.linspace(0.0, 1.0, K + 1)

    # Shared base Brownian increments (CRNs) and prior keys
    key_dw_V, key_dW_indep = jax.random.split(base_key, 2)
    dW_V = jax.random.normal(key_dw_V, (K,)) * jnp.sqrt(1.0 / K)
    dW_independent = jax.random.normal(key_dW_indep, (K,)) * jnp.sqrt(1.0 / K)

    # Initial state
    observation_S0 = 1.0
    observartion_V0 = v0_star

    # Simulated observations
    y_obs = simulate_observation_logS(observation_S0, observartion_V0, H_star, nu_star, rho_star, v0_star, t, dW_V, dW_independent)

    # Priors (deliberately noisy)
    tH0, tNu0, tRho0, tV00 = sample_unconstrained_priors(base_key, N)

    # Constrained-space prior means (baseline without filtering)
    hurst, nu, rho, v0 = constrain_theta(tH0, tNu0, tRho0, tV00)
    prior_means: dict[str, float] = {
        "H": float(jnp.mean(hurst)),
        "nu": float(jnp.mean(nu)),
        "rho": float(jnp.mean(rho)),
        "v0": float(jnp.mean(v0)),
    }

    ens0 = RBergomiEnsemble(
        S=jnp.full((N,), 1.0),
        V=jnp.full((N,), v0),
        X_driver=jnp.zeros((N,)),
        W_driver=jnp.zeros((N,)),
        tH=tH0,
        tNu=tNu0,
        tRho=tRho0,
        tV0=tV00,
    )

    # Run filter
    ensK = run_ensemble_kalman_filter(ens0, t, y_obs, R, dW_V, dW_independent)

    # Extract final parameter means
    Hf, tNuf, rhof, v0f = constrain_theta(ensK.tH, ensK.tNu, ensK.tRho, ensK.tV0)
    estimates: dict[str, float] = {
        "H": float(jnp.mean(Hf)),
        "nu": float(jnp.mean(tNuf)),
        "rho": float(jnp.mean(rhof)),
        "v0": float(jnp.mean(v0f)),
    }
    return estimates, prior_means


if __name__ == "__main__":
    key = jax.random.key(42)

    # Experiment configuration
    K = 4000
    N = 1000
    R = 1e-5
    num_trials = 5  # number of PRNG keys to iterate over

    # True parameters (θ*)
    H_star = 0.18
    nu_star = 1.40
    rho_star = -0.6
    v0_star = 0.04

    # Accumulate squared errors over trials (for MSE)
    sum_sq_err: dict[str, float] = {"H": 0.0, "nu": 0.0, "rho": 0.0, "v0": 0.0}
    sum_sq_err_prior: dict[str, float] = {"H": 0.0, "nu": 0.0, "rho": 0.0, "v0": 0.0}

    trial_keys = jax.random.split(key, num_trials)

    for i in range(num_trials):
        est, prior = run_single_estimation_with_key(trial_keys[i], K, N, R, H_star, nu_star, rho_star, v0_star)
        sum_sq_err["H"] += (est["H"] - H_star) ** 2
        sum_sq_err["nu"] += (est["nu"] - nu_star) ** 2
        sum_sq_err["rho"] += (est["rho"] - rho_star) ** 2
        sum_sq_err["v0"] += (est["v0"] - v0_star) ** 2
        sum_sq_err_prior["H"] += (prior["H"] - H_star) ** 2
        sum_sq_err_prior["nu"] += (prior["nu"] - nu_star) ** 2
        sum_sq_err_prior["rho"] += (prior["rho"] - rho_star) ** 2
        sum_sq_err_prior["v0"] += (prior["v0"] - v0_star) ** 2

    mse_err: dict[str, float] = {k: v / float(num_trials) for k, v in sum_sq_err.items()}
    mse_err_prior: dict[str, float] = {k: v / float(num_trials) for k, v in sum_sq_err_prior.items()}

    print("Mean squared parameter estimation error (MSE) over", num_trials, "keys:")
    print(f"  H   : {mse_err['H']:.6f} (prior: {mse_err_prior['H']:.6f})")
    print(f"  nu  : {mse_err['nu']:.6f} (prior: {mse_err_prior['nu']:.6f})")
    print(f"  rho : {mse_err['rho']:.6f} (prior: {mse_err_prior['rho']:.6f})")
    print(f"  v0  : {mse_err['v0']:.6f} (prior: {mse_err_prior['v0']:.6f})")
