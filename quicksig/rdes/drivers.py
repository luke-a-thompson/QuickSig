import jax
import jax.numpy as jnp
import jax.lax as lax
from quicksig.rdes.rde_types import Path


def bm_driver(key: jax.Array, timesteps: int, dim: int) -> Path:
    """
    Generates a Brownian motion path.

    args:
        key:        JAX random key
        timesteps:  number of time steps
        dim:        dimension of the Brownian motion path
    returns:
        A JAX array of shape (timesteps + 1, dim) representing the Brownian motion path.
    """
    # Interpret `timesteps` as the number of increments; output has timesteps+1 samples
    dt = 1.0 / timesteps
    increments = jax.random.normal(key, (timesteps, dim)) * jnp.sqrt(dt)
    path = jnp.concatenate([jnp.zeros((1, dim)), jnp.cumsum(increments, axis=0)], axis=0)
    # Half-open interval: number of samples equals end - start
    return Path(path, (0, timesteps + 1))


def correlate_bm_driver_against_reference(reference_path: Path, indep_path: Path, rho: float) -> Path:
    """
    Correlates a Brownian motion path against a reference path by their increments.

    Args:
        reference_path (Path): The reference path to correlate against.
        indep_path (Path): The independent path to correlate.
        rho (float): The correlation parameter.

    Raises:
        ValueError: If the reference path and indep path have different shapes.
        ValueError: If the correlation parameter is not between -1 and 1.

    Returns:
        Path: The correlated path.
    """

    if reference_path.path.shape != indep_path.path.shape:
        raise ValueError(f"Reference path and indep path must have the same shape. Got shapes {reference_path.path.shape} and {indep_path.path.shape}")
    if rho < -1 or rho > 1:
        raise ValueError(f"rho must be between -1 and 1. Got {rho}")

    # Get increments of the independent paths
    reference_increments = jnp.diff(reference_path.path, axis=0)
    indep_increments = jnp.diff(indep_path.path, axis=0)

    correlated_increments = rho * reference_increments + jnp.sqrt(1 - rho**2) * indep_increments

    initial_cond = indep_path.path[0, :]
    correlated_path = initial_cond + jnp.cumsum(correlated_increments, axis=0)
    correlated_path = jnp.concatenate([initial_cond[None, :], correlated_path], axis=0)
    return Path(correlated_path, indep_path.interval)


def correlated_bm_drivers(indep_bm_paths: Path, corr_matrix: jax.Array) -> Path:
    """
    Generates a new Brownian motion path that is correlated with a reference path.

    It takes two independent Brownian motion paths and a 2x2 correlation matrix.
    It returns a new path that has the desired correlation with the first path.
    The first path is returned unchanged. This function effectively replaces the
    second path with a correlated version.

    args:
        path1:       A JAX array of shape (timesteps + 1, dim) representing the first Brownian motion path.
        path2:       A JAX array of shape (timesteps + 1, dim) representing the second (independent) Brownian motion path.
        corr_matrix: 2x2 correlation matrix.
    returns:
        A JAX array of shape (timesteps + 1, dim) representing the new correlated Brownian motion path.
    """
    num_paths = indep_bm_paths.path.shape[0]
    if corr_matrix.shape != (num_paths, num_paths):
        raise ValueError(f"Received {num_paths} paths, but got a correlation matrix with shape {corr_matrix.shape}. Corr matrix must be shape (num_paths, num_paths).")
    if not jnp.allclose(jnp.diag(corr_matrix), 1.0):
        raise ValueError(f"The diagonal of the correlation matrix must be 1. Got {jnp.diag(corr_matrix)}")
    if not jnp.allclose(corr_matrix, corr_matrix.T):
        raise ValueError("The correlation matrix must be symmetric.")

    chol_matrix = jnp.linalg.cholesky(corr_matrix)

    # Get increments of the independent paths
    indep_increments = jnp.diff(indep_bm_paths.path, axis=1)

    # Correlate the increments
    correlated_increments = jnp.einsum("qtd,pq->ptd", indep_increments, chol_matrix)

    # Cumsum to get the new path
    new_path = jnp.cumsum(correlated_increments, axis=1)

    # Add zeros at the beginning to ensure Brownian motion starts at 0
    # Shape: (batch, time, nodes) -> add zeros at time=0
    zeros_shape = (new_path.shape[0], 1, new_path.shape[2])
    new_path = jnp.concatenate([jnp.zeros(zeros_shape), new_path], axis=1)

    return Path(new_path, indep_bm_paths.interval)


def fractional_bm_driver(key: jax.Array, timesteps: int, dim: int, hurst: float) -> Path:
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
    return Path(paths.T, (0, timesteps))


def riemann_liouville_driver(
    key: jax.Array,
    timesteps: int,
    hurst: float,
    bm_path,                      # Path: Brownian path for W1(t) INCLUDING t=0, shape (T+1, dim)
    use_fft: bool = True,
):
    """
    Hybrid scheme (kappa = 1) for the RL/type-II fBM driver used in rBergomi.

    Implements (for k = 1..T, Δ=1/T, α = H-1/2):

        Y_k = sqrt(2H) * [ I_k  +  sum_{i=2}^k w_i ΔW_{k+1-i} ],

    where
        I_k  ≈ a ΔW_k + b Z_k,
        a    = Δ^α / (α+1),
        b^2  = Δ^{2α+1}/(2α+1)  -  a^2 Δ,
        w_i  = Δ^α * ( i^{α+1} - (i-1)^{α+1} ) / (α+1).

    Returns a Path with Y_0 = 0 and Y_k on the input grid.
    """

    assert 0.0 < hurst < 0.5, "Hybrid κ=1 is used for H in (0, 1/2)."
    assert bm_path.num_timesteps == timesteps + 1, "bm_path must have shape (timesteps+1, dim)."

    dim = bm_path.ambient_dimension
    Δ   = 1.0 / timesteps
    α   = hurst - 0.5
    sqrt2H = jnp.sqrt(2.0 * hurst)

    # Brownian increments ΔW_k, k=1..T (var = Δ)
    dW = jnp.diff(bm_path.path, axis=0)                         # (T, dim)

    # Recent-interval integral I_k = a ΔW_k + b Z_k (Z ⟂ ΔW, i.i.d. N(0,1))
    a = (Δ ** α) / (α + 1.0)
    var_I = (Δ ** (2.0 * α + 1.0)) / (2.0 * α + 1.0)
    # numerical guard in case of float underflow:
    b = jnp.sqrt(jnp.maximum(var_I - (a * a) * Δ, 0.0))

    Z = jax.random.normal(key, shape=dW.shape)                  # (T, dim)
    I = a * dW + b * Z                                          # (T, dim)

    # Historical weights w_i for i=2..T
    i = jnp.arange(2, timesteps + 1, dtype=dW.dtype)            # (T-1,)
    w = (Δ ** α) * (i ** (α + 1.0) - (i - 1.0) ** (α + 1.0)) / (α + 1.0)  # (T-1,)

    # Convolution Y2_k = sum_{i=2}^k w_i ΔW_{k+1-i}, with Y2_1 = 0.
    # We compute this per dimension. For speed and exact indexing, use FFT.
    def conv_full_1d(w, x):
        # full convolution y[n] = sum_k w[k]*x[n-k]; we only need the first (T-1) outputs
        L = int(2 ** jnp.ceil(jnp.log2(w.shape[0] + x.shape[0] - 1)))
        wf = jnp.fft.rfft(jnp.pad(w, (0, L - w.shape[0])))
        xf = jnp.fft.rfft(jnp.pad(x, (0, L - x.shape[0])))
        y  = jnp.fft.irfft(wf * xf, n=L)[: w.shape[0] + x.shape[0] - 1]
        return y

    if use_fft:
        # x is ΔW[0:T-1] (i.e., ΔW_1..ΔW_{T-1}); Y2_k for k>=2 is y[k-2]
        def per_dim(x):
            y = conv_full_1d(w, x[:-1])                         # length 2T-3
            return jnp.concatenate([jnp.zeros((1,), x.dtype), y[: timesteps - 1]])  # (T,)

        Y2 = jnp.stack([per_dim(dW[:, d]) for d in range(dim)], axis=1)  # (T, dim)
    else:
        # O(T^2) fallback (fine for small T)
        def hist_scan(carry, k):  # k = 1..T
            # k==1 -> Y2=0; else Y2_k = sum_{j=0}^{k-2} w[j]*ΔW_{k-1-j}
            def body():
                j_idx = jnp.arange(0, k - 1)
                return (w[j_idx] @ dW[k - 1 - j_idx, :])  # (dim,)
            Y2k = lax.cond(k > 1, body, lambda: jnp.zeros((dim,), dW.dtype))
            return carry, Y2k

        _, Y2 = lax.scan(hist_scan, None, jnp.arange(1, timesteps + 1))
        # Y2 shape (T, dim)

    # Assemble Y_k values and prepend Y_0=0
    Y_tail = sqrt2H * (I + Y2)                                  # (T, dim)
    Y_path = jnp.concatenate([jnp.zeros((1, dim), Y_tail.dtype), Y_tail], axis=0)
    return Path(Y_path, bm_path.interval)

if __name__ == "__main__":
    # Example: generate and plot 1000 Riemann–Liouville drivers using vmap
    import matplotlib.pyplot as plt

    batch_size = 1000
    timesteps = 512
    dim = 1
    hurst = 0.5

    base_key = jax.random.PRNGKey(0)
    key_bm, key_rl = jax.random.split(base_key)
    bm_keys = jax.random.split(key_bm, batch_size)
    rl_keys = jax.random.split(key_rl, batch_size)

    batched_bm_paths = jax.vmap(bm_driver, in_axes=(0, None, None))(bm_keys, timesteps, dim)
    batched_rl_paths = jax.vmap(riemann_liouville_driver, in_axes=(0, None, None, 0))(rl_keys, timesteps, hurst, batched_bm_paths)

    rl_paths_np = jax.device_get(batched_rl_paths.path)

    plt.figure(figsize=(10, 6))
    for i in range(batch_size):
        plt.plot(rl_paths_np[i, :, 0], linewidth=0.5, alpha=0.15, color="tab:blue")
    plt.title(f"Riemann–Liouville drivers (H={hurst}, N={timesteps}, batch={batch_size})")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig("docs/assets/riemann_liouville_monte_carlo.png", dpi=150)
    plt.close()
