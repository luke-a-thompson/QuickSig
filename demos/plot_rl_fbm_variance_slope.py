import jax
import jax.numpy as jnp

from quicksig.rde_bench.rp_kalman import compute_fractional_brownian_increment


def generate_rlfbm_path(H: float, dW: jax.Array, t: jax.Array, Z: jax.Array) -> jax.Array:
    """
    Generate a single Riemann–Liouville fBM path using compute_fractional_brownian_increment.

    Returns an array of shape (K+1,) with X_0 = 0 and cumulative sum of increments.
    """
    K = dW.shape[0]

    def compute_increment(k: jax.Array) -> jax.Array:
        return compute_fractional_brownian_increment(jnp.asarray(H, dtype=t.dtype), dW, t, k, Z)

    ks = jnp.arange(K)
    dX = jax.vmap(compute_increment)(ks)  # (K,)
    X = jnp.concatenate([jnp.zeros((1,), dtype=t.dtype), jnp.cumsum(dX)])  # (K+1,)
    return X


def simulate_variance_slope(seed: int = 0, timesteps: int = 1000, hurst: float = 0.2, num_paths: int = 500, T: float = 1.0) -> tuple[float, float]:
    """
    Simulate many RL-fBM paths and compute slope of log Var[X_t] vs log t.

    Returns (estimated_slope, expected_slope=2H).
    """
    key = jax.random.key(seed)
    K = timesteps
    Δ = T / float(K)

    t = jnp.linspace(0.0, T, K + 1)
    keys = jax.random.split(key, num_paths)

    def one_path(k: jax.Array) -> jax.Array:
        k_dw, k_z = jax.random.split(k)
        dW = jax.random.normal(k_dw, (K,)) * jnp.sqrt(Δ)  # (K,)
        Z = jax.random.normal(k_z, (K,))  # (K,)
        return generate_rlfbm_path(hurst, dW, t, Z)  # (K+1,)

    paths = jax.vmap(one_path)(keys)  # (M, K+1)

    variances = jnp.var(paths, axis=0, ddof=1)  # (K+1,)
    # Exclude t=0 where variance is 0
    t_pos = t[1:]
    v_pos = variances[1:]

    # Guard against potential numerical issues at extremely small times
    mask = v_pos > 0
    logt = jnp.log(t_pos[mask])
    logv = jnp.log(v_pos[mask])

    slope, _ = jnp.polyfit(logt, logv, 1)
    expected = 2.0 * hurst
    return float(slope), float(expected)


def main() -> None:
    slope, expected = simulate_variance_slope(seed=123, timesteps=1000, hurst=0.2, num_paths=500, T=1.0)
    err = abs(slope - expected)
    print(f"slope={slope:.4f}, expected=2H={expected:.4f}, abs_error={err:.4f}")


if __name__ == "__main__":
    main()
