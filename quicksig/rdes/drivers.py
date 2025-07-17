import jax
import jax.numpy as jnp


def bm_driver(key: jax.Array, timesteps: int, dim: int) -> jax.Array:
    """
    Generates a Brownian motion path.

    args:
        key:        JAX random key
        timesteps:  number of time steps
        dim:        dimension of the Brownian motion path
    returns:
        A JAX array of shape (timesteps + 1, dim) representing the Brownian motion path.
    """
    dt = 1.0 / timesteps
    increments = jax.random.normal(key, (timesteps, dim)) * jnp.sqrt(dt)
    path = jnp.cumsum(increments, axis=0)
    path = jnp.concatenate([jnp.zeros((1, dim)), path])
    return path


def fractional_bm_driver(key: jax.Array, timesteps: int, dim: int, hurst: float) -> jax.Array:
    """
    Generates sample paths of fractional Brownian Motion using the Davies Harte method with JAX.

    @author: Luke Thompson, PhD Student, University of Sydney
    @author: Justin Yu, M.S. Financial Engineering, Stevens Institute of Technology

    args:
        key:        JAX random key
        timesteps:  number of time steps within the timeframe
        hurst:      Hurst parameter
        dim:        dimension of the fBM path
    """

    def get_path(key: jax.Array, timesteps: int, hurst: float) -> jax.Array:
        gamma = lambda k, H: 0.5 * (jnp.abs(k - 1) ** (2 * H) - 2 * jnp.abs(k) ** (2 * H) + jnp.abs(k + 1) ** (2 * H))

        k_vals = jnp.arange(0, timesteps)
        g = gamma(k_vals, hurst)
        r = jnp.concatenate([g, jnp.array([0.0]), jnp.flip(g)[:-1]])

        # Step 1 (eigenvalues)
        j = jnp.arange(0, 2 * timesteps)
        k = 2 * timesteps - 1
        lk = jnp.fft.fft(r * jnp.exp(2 * jnp.pi * 1j * k * j * (1 / (2 * timesteps))))[::-1].real

        # Step 2 (get random variables)
        key1, key2, key3 = jax.random.split(key, 3)

        # Generate all random numbers at once
        rvs = jax.random.normal(key1, shape=(timesteps - 1, 2))
        v_0_0 = jax.random.normal(key2)
        v_n_0 = jax.random.normal(key3)

        Vj = jnp.zeros((2 * timesteps, 2))
        Vj = Vj.at[0, 0].set(v_0_0)
        Vj = Vj.at[timesteps, 0].set(v_n_0)

        indices1 = jnp.arange(1, timesteps)
        indices2 = jnp.arange(2 * timesteps - 1, timesteps, -1)

        Vj = Vj.at[indices1, :].set(rvs)
        Vj = Vj.at[indices2, :].set(rvs)

        # Step 3 (compute Z)
        wk = jnp.zeros(2 * timesteps, dtype=jnp.complex64)
        wk = wk.at[0].set(jnp.sqrt(lk[0] / (2 * timesteps)) * Vj[0, 0])
        wk = wk.at[1:timesteps].set(jnp.sqrt(lk[1:timesteps] / (4 * timesteps)) * (Vj[1:timesteps, 0] + 1j * Vj[1:timesteps, 1]))
        wk = wk.at[timesteps].set(jnp.sqrt(lk[timesteps] / (2 * timesteps)) * Vj[timesteps, 0])
        wk = wk.at[timesteps + 1 : 2 * timesteps].set(
            jnp.sqrt(lk[timesteps + 1 : 2 * timesteps] / (4 * timesteps)) * (jnp.flip(Vj[1:timesteps, 0]) - 1j * jnp.flip(Vj[1:timesteps, 1]))
        )

        Z = jnp.fft.ifft(wk)
        fGn = Z[0:timesteps].real
        fBm = jnp.cumsum(fGn) * (timesteps ** (-hurst))
        path = jnp.concatenate([jnp.array([0.0]), fBm])
        return path

    keys = jax.random.split(key, dim)
    paths = jax.vmap(get_path, in_axes=(0, None, None))(keys, timesteps, hurst)
    return paths.T


def riemann_liouville_driver(key: jax.Array, timesteps: int, hurst: float, bm_path: jax.Array) -> jax.Array:
    """
    Simulate a type-II (Riemann–Liouville) fractional Brownian motion (fBM)
    path using the κ = 1 hybrid scheme of Bennedsen-Lunde-Pakkanen
    (2017, Eq. (20)) on a *pre-computed* Brownian motion trajectory.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random seed used **only** for the additional Gaussian random
        variables required by the scheme (independent of the supplied
        Brownian path).
    bm_path : jax.Array
        Brownian motion trajectory **including** the initial point
        ``t = 0``.  Expected shape ``(timesteps+1, dim)`` where
        ``dim`` is the spatial dimension.  The driver derives the
        Brownian increments internally.
    hurst : float
        Hurst parameter ``H ∈ (0,1)``.  Inside the algorithm we use
        ``ζ = H − 0.5``.

    Returns
    -------
    jax.Array
        Simulated Riemann-Liouville fBM path of shape ``(timesteps+1, dim)``
        on the same grid as *bm_path*.
    """

    # Check dimensions
    assert bm_path.shape[0] == timesteps + 1, "bm_path must have shape (timesteps+1, dim)"
    dim: int = bm_path.shape[1] if bm_path.ndim == 2 else 1

    Δ: float = 1.0 / timesteps

    # Brownian increments ΔW_k on [t_{k-1}, t_k]
    ΔW: jax.Array = jnp.diff(bm_path if bm_path.ndim == 2 else bm_path[:, None], axis=0)  # (timesteps, dim)

    ζ: float = hurst - 0.5
    Cζ: jax.Array = jnp.sqrt(2 * ζ + 1)

    # Extra Gaussians Z_k independent of ΔW_k
    key, sub = jax.random.split(key)
    Z: jax.Array = jax.random.normal(sub, (timesteps, dim))

    # Coefficients for the recent integral I_k
    a: float = Δ**ζ / (ζ + 1)
    var_I: float = Δ ** (2 * ζ + 1) / (2 * ζ + 1)
    b: jax.Array = jnp.sqrt(var_I - (a**2) * Δ)

    I: jax.Array = a * ΔW + b * Z  # shape (timesteps, dim)

    # Pre-compute weights b_i^∗  (i = 2,…,N)
    i: jax.Array = jnp.arange(2, timesteps + 1)
    b_star: jax.Array = (Δ**ζ / (ζ + 1)) * (i ** (ζ + 1) - (i - 1) ** (ζ + 1))  # (timesteps-1,)

    # Prefix-convolution via an O(N) scan
    def step(carry: jax.Array, k: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute one step of the prefix-convolution (vector-valued)."""

        def update_hist() -> jax.Array:
            return carry + b_star[k - 2] * ΔW[timesteps - k]

        hist: jax.Array = jax.lax.cond(k > 1, update_hist, lambda: jnp.zeros_like(carry))

        GX_k: jax.Array = Cζ * (I[k - 1] + hist)
        return hist, GX_k

    # Initial carry is a zero vector of shape (dim,)
    _, GX_tail = jax.lax.scan(step, jnp.zeros((dim,)), jnp.arange(1, timesteps + 1))

    # Prepend initial zero to obtain the full path (timesteps+1, dim)
    return jnp.concatenate([jnp.zeros((1, dim)), GX_tail], axis=0)


if __name__ == "__main__":
    key = jax.random.key(0)
    bm_path = bm_driver(key, timesteps=1000, dim=1)  # (timesteps+1, dim)
    rl_path = riemann_liouville_driver(key, timesteps=1000, hurst=0.25, bm_path=bm_path)
    print(bm_path.shape)
    print(rl_path.shape)
    # import matplotlib.pyplot as plt

    # key = jax.random.key(0)
    # H = 0.25
    # N = 1000
    # dim = 1

    # # Generate keys for each driver
    # bm_key, fbm_key, rl_key = jax.random.split(key, 3)

    # # Generate paths
    # bm_path = bm_driver(bm_key, timesteps=N, dim=dim)
    # fbm_path = fractional_bm_driver(fbm_key, timesteps=N, dim=dim, hurst=H) * 1000
    # rl_path = riemann_liouville_driver(rl_key, bm_path=bm_path, hurst=H)

    # # Create the plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(bm_path, label="Brownian Motion")
    # plt.plot(fbm_path, label=f"fBM Davies-Harte (H={H})")
    # plt.plot(rl_path, label=f"Riemann-Liouville (H={H})")

    # plt.title("Comparison of RDE Drivers")
    # plt.xlabel("Time")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("drivers_comparison.png")
