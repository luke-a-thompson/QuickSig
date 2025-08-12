import jax
import jax.numpy as jnp
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


def riemann_liouville_driver(key: jax.Array, timesteps: int, hurst: float, bm_path: Path) -> Path:
    """
    Simulate a type-II (Riemann-Liouville) fractional Brownian motion (fBM)
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
        ``ζ = H - 0.5``.

    Returns
    -------
    jax.Array
        Simulated Riemann-Liouville fBM path of shape ``(timesteps+1, dim)``
        on the same grid as *bm_path*.
    """

    # Check dimensions
    assert bm_path.num_timesteps == timesteps + 1, "bm_path must have shape (timesteps+1, dim)"
    dim: int = bm_path.ambient_dimension

    Δ: float = 1.0 / timesteps

    # Brownian increments ΔW_k on [t_{k-1}, t_k]
    ΔW: jax.Array = jnp.diff(bm_path.path, axis=0)  # (timesteps, dim)

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
    return Path(jnp.concatenate([jnp.zeros((1, dim)), GX_tail], axis=0), (0, timesteps + 1))

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    fbm_path = fractional_bm_driver(key, timesteps=100, dim=1, hurst=0.5)
    print(fbm_path)

    bm_path_for_rl = bm_driver(key, timesteps=100, dim=1)
    rl_path = riemann_liouville_driver(key, timesteps=100, hurst=0.5, bm_path=bm_path_for_rl)
    print(rl_path)