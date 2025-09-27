import jax
import jax.numpy as jnp
import pytest

from quicksig.rde_bench.rough_volatility import (
    make_black_scholes_model_spec,
    make_bergomi_model_spec,
    make_rough_bergomi_model_spec,
    get_bonesini_noise_drivers,
    solve_bonesini_rde_from_drivers,
)


# --- Common utilities ---


def _simulate_model(
    model_spec,
    *,
    num_paths: int,
    noise_timesteps: int,
    rde_timesteps: int,
    s0: float,
    seed: int,
):
    key = jax.random.key(seed)
    keys = jax.random.split(key, num_paths)

    # Drivers
    y0s, Xs, Ws = jax.vmap(lambda k: get_bonesini_noise_drivers(k, noise_timesteps=noise_timesteps, model_spec=model_spec, s_0=s0))(keys)

    # Solve RDE (save on the noise grid via internal saveat)
    solve_vmap = jax.vmap(
        lambda y0, X, W: solve_bonesini_rde_from_drivers(
            y0,
            X,
            W,
            model_spec=model_spec,
            noise_timesteps=noise_timesteps,
            rde_timesteps=rde_timesteps,
        )
    )
    sols = solve_vmap(y0s, Xs, Ws)

    # `sols.ys` is shaped (num_paths, noise_timesteps+1, state_dim)
    Ys = sols.ys
    S_paths = Ys[..., 0]
    V_paths = Ys[..., 1]
    return S_paths, V_paths, Xs, Ws


# --- Black–Scholes ---


@pytest.fixture(scope="module")
def bs_mc():
    v0 = 0.04
    s0 = 1.0
    num_paths = 384
    N = 256
    M = 512
    spec = make_black_scholes_model_spec(v_0=v0)
    S_paths, V_paths, Xs, Ws = _simulate_model(
        spec,
        num_paths=num_paths,
        noise_timesteps=N,
        rde_timesteps=M,
        s0=s0,
        seed=2024,
    )
    return {
        "S": S_paths,
        "V": V_paths,
        "X": Xs,
        "W": Ws,
        "s0": s0,
        "v0": v0,
    }


def test_black_scholes_martingale(bs_mc) -> None:
    S = bs_mc["S"]
    s0 = float(bs_mc["s0"])  # scalar
    ST = S[:, -1]
    mean_ST = jnp.mean(ST)
    std_ST = jnp.std(ST, ddof=1)
    se = std_ST / jnp.sqrt(S.shape[0])
    assert jnp.abs(mean_ST - s0) <= 4.0 * se + 1e-2


def test_black_scholes_log_return_variance(bs_mc) -> None:
    S = bs_mc["S"]
    v0 = float(bs_mc["v0"])  # T=1
    s0 = float(bs_mc["s0"])  # scalar
    ST = S[:, -1]
    log_ret = jnp.log(ST / s0)
    var = jnp.var(log_ret, ddof=1)
    assert jnp.isclose(var, v0, rtol=0.15)


def test_black_scholes_log_normality(bs_mc) -> None:
    S = bs_mc["S"]
    s0 = float(bs_mc["s0"])  # scalar
    ST = S[:, -1]
    log_ret = jnp.log(ST / s0)
    z = (log_ret - jnp.mean(log_ret)) / jnp.std(log_ret, ddof=1)
    skew = jnp.mean(z**3)
    kurt = jnp.mean(z**4)
    # For Normal: skew ~ 0, kurtosis ~ 3
    assert jnp.abs(skew) < 0.2
    assert jnp.abs(kurt - 3.0) < 0.3


# --- Bergomi ---


@pytest.fixture(scope="module")
def bergomi_mc():
    rho = -0.7
    v0 = 0.0  # not used directly in Bergomi spec
    s0 = 1.0
    num_paths = 384
    N = 256
    M = 512
    spec = make_bergomi_model_spec(v_0=v0, rho=rho)
    S_paths, V_paths, Xs, Ws = _simulate_model(
        spec,
        num_paths=num_paths,
        noise_timesteps=N,
        rde_timesteps=M,
        s0=s0,
        seed=7,
    )
    return {
        "S": S_paths,
        "V": V_paths,
        "X": Xs,
        "W": Ws,
        "s0": s0,
        "rho": rho,
    }


def test_bergomi_martingale(bergomi_mc) -> None:
    S = bergomi_mc["S"]
    s0 = float(bergomi_mc["s0"])  # scalar
    ST = S[:, -1]
    mean_ST = jnp.mean(ST)
    std_ST = jnp.std(ST, ddof=1)
    se = std_ST / jnp.sqrt(S.shape[0])
    assert jnp.abs(mean_ST - s0) <= 4.0 * se + 1e-2


def test_bergomi_driver_correlation(bergomi_mc) -> None:
    X = bergomi_mc["X"]
    W = bergomi_mc["W"]
    rho = float(bergomi_mc["rho"])  # target
    dX = jnp.diff(X, axis=1).flatten()
    dW = jnp.diff(W, axis=1).flatten()
    corr = jnp.corrcoef(dX, dW)[0, 1]
    assert jnp.isclose(corr, rho, atol=0.12)


def test_bergomi_leverage_effect(bergomi_mc) -> None:
    S = bergomi_mc["S"]
    X = bergomi_mc["X"]
    rho = float(bergomi_mc["rho"])  # target sign/magnitude
    logS = jnp.log(S)
    dlogS = jnp.diff(logS, axis=1).flatten()
    dX = jnp.diff(X, axis=1).flatten()
    corr = jnp.corrcoef(dlogS, dX)[0, 1]
    assert jnp.sign(corr) == jnp.sign(rho)
    assert jnp.isclose(corr, rho, atol=0.15)


# --- Rough Bergomi ---


@pytest.fixture(scope="module")
def rough_bergomi_mc():
    v0 = 0.04
    nu = 1.991
    H = 0.25
    rho = -0.7
    s0 = 1.0
    num_paths = 320
    N = 256
    M = 512
    spec = make_rough_bergomi_model_spec(v_0=v0, nu=nu, hurst=H, rho=rho)
    S_paths, V_paths, Xs, Ws = _simulate_model(
        spec,
        num_paths=num_paths,
        noise_timesteps=N,
        rde_timesteps=M,
        s0=s0,
        seed=123,
    )
    return {
        "S": S_paths,
        "V": V_paths,
        "X": Xs,
        "W": Ws,
        "s0": s0,
        "v0": v0,
        "nu": nu,
        "H": H,
    }


def test_rough_bergomi_martingale(rough_bergomi_mc) -> None:
    S = rough_bergomi_mc["S"]
    s0 = float(rough_bergomi_mc["s0"])  # scalar
    ST = S[:, -1]
    mean_ST = jnp.mean(ST)
    std_ST = jnp.std(ST, ddof=1)
    se = std_ST / jnp.sqrt(S.shape[0])
    assert jnp.abs(mean_ST - s0) <= 4.0 * se + 1e-2


def test_rough_bergomi_vol_roughness_scaling(rough_bergomi_mc) -> None:
    V = rough_bergomi_mc["V"]
    H = float(rough_bergomi_mc["H"])  # hurst
    # Use pooled variances of lagged increments across all times/paths
    lags = jnp.array([1, 2, 4])
    vars_per_lag = []
    for L in list(lags):
        dV_L = (V[:, L:, ...] - V[:, :-L, ...]).flatten()
        vars_per_lag.append(jnp.var(dV_L, ddof=1))
    vars_per_lag = jnp.array(vars_per_lag)
    slope, _ = jnp.polyfit(jnp.log(lags.astype(float)), jnp.log(vars_per_lag), 1)
    assert jnp.isclose(slope, 2.0 * H, atol=0.25)


def test_rough_bergomi_exp_normalization(rough_bergomi_mc) -> None:
    V = rough_bergomi_mc["V"]
    nu = float(rough_bergomi_mc["nu"])  # vol-of-vol
    H = float(rough_bergomi_mc["H"])  # hurst
    T = V.shape[1] - 1
    # Check at several internal times
    for k in [int(T * 0.25), int(T * 0.5), int(T * 0.75), T]:
        t = k / T
        Vk = V[:, k]
        m = jnp.mean(jnp.exp(nu * Vk - 0.5 * (nu**2) * (t ** (2.0 * H))))
        assert jnp.isclose(m, 1.0, atol=0.1)
