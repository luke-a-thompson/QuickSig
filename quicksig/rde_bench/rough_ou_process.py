import jax
import jax.numpy as jnp
import diffrax as dfx
from quicksig.rde.drivers import fractional_bm_driver
from quicksig.rde.rde_types import Path


def rough_ou_process(
    key: jax.Array,
    timesteps: int,
    dim: int,
    theta: float,
    mu: float,
    sigma: float,
    hurst: float,
    x0: float | None = None,
) -> Path:
    """
    Generates a rough Ornstein-Uhlenbeck process path using fractional Brownian motion and Heun solver.
    
    The rough OU process satisfies the SDE:
        dX_t = θ(μ - X_t)dt + σdW_t^H
    
    where:
        - θ (theta): rate of mean reversion
        - μ (mu): long-term mean
        - σ (sigma): volatility
        - W_t^H: fractional Brownian motion with Hurst parameter H
        - H (hurst): Hurst parameter in (0, 1)
    
    Uses diffrax with Heun's method for numerical integration.
    
    Args:
        key: JAX random key
        timesteps: number of time steps
        dim: dimension of the rough OU process
        theta: rate of mean reversion (must be positive)
        mu: long-term mean
        sigma: volatility (must be non-negative)
        hurst: Hurst parameter (must be in (0, 1))
        x0: initial value (defaults to mu if not provided)
    
    Returns:
        A Path object of shape (timesteps + 1, dim) representing the rough OU process.
    """
    if theta <= 0:
        raise ValueError(f"theta must be positive. Got {theta}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative. Got {sigma}")
    if not (0 < hurst < 1):
        raise ValueError(f"hurst must be in (0, 1). Got {hurst}")
    
    # Generate fractional Brownian motion path
    fbm_path = fractional_bm_driver(key, timesteps, dim, hurst)
    
    # Time grid
    ts = jnp.linspace(0.0, 1.0, timesteps + 1)
    
    # Initial condition
    if x0 is None:
        x0 = mu
    y0 = jnp.full((dim,), x0)
    
    # Define the drift term: f(t, y, args) = θ(μ - y)
    def drift(t: float, y: jax.Array, args: None) -> jax.Array:
        return theta * (mu - y)
    
    # Define the diffusion term: g(t, y, args) = σ (scalar multiplies the control)
    def diffusion(t: float, y: jax.Array, args: None) -> jax.Array:
        # Return a matrix of shape (dim, dim) where each row gets scaled by sigma
        return sigma * jnp.eye(dim)
    
    # Create linear interpolation control from fractional Brownian motion
    fbm_control = dfx.LinearInterpolation(ts=ts, ys=fbm_path.path)
    
    # Build the SDE terms
    terms = dfx.MultiTerm(
        dfx.ODETerm(drift),
        dfx.ControlTerm(diffusion, control=fbm_control)
    )
    
    # Solve using Heun's method
    solution = dfx.diffeqsolve(
        terms=terms,
        solver=dfx.Heun(),
        t0=0.0,
        t1=1.0,
        dt0=ts[1] - ts[0],
        y0=y0,
        saveat=dfx.SaveAt(ts=ts),
        stepsize_controller=dfx.ConstantStepSize(),
        max_steps=None,
    )
    
    return Path(solution.ys, (0, timesteps + 1))


if __name__ == "__main__":
    # Example: generate and plot 1000 rough OU processes using vmap
    import matplotlib.pyplot as plt
    
    batch_size = 1000
    timesteps = 512
    dim = 1
    theta = 0.5
    mu = 0.0
    sigma = 0.3
    hurst = 0.7  # H > 0.5 for long memory
    x0 = 1.0
    
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, batch_size)
    
    # Vectorize over multiple paths
    batched_rough_ou_paths = jax.vmap(
        rough_ou_process,
        in_axes=(0, None, None, None, None, None, None, None)
    )(keys, timesteps, dim, theta, mu, sigma, hurst, x0)
    
    rough_ou_paths_np = jax.device_get(batched_rough_ou_paths.path)
    
    plt.figure(figsize=(10, 6))
    for i in range(batch_size):
        plt.plot(rough_ou_paths_np[i, :, 0], linewidth=0.5, alpha=0.15, color="tab:blue")
    plt.axhline(y=mu, color='red', linestyle='--', linewidth=2, label=f'Mean μ={mu}')
    plt.title(f"Rough Ornstein-Uhlenbeck Process (θ={theta}, μ={mu}, σ={sigma}, H={hurst}, N={timesteps}, batch={batch_size})")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("docs/rde_bench/rough_ou_process_monte_carlo.png", dpi=150)
    plt.close()
    
    print(f"Generated {batch_size} rough OU process paths with Hurst={hurst}")
    print(f"Mean of final values: {rough_ou_paths_np[:, -1, 0].mean():.4f} (expected ≈ {mu})")
    print(f"Std of final values: {rough_ou_paths_np[:, -1, 0].std():.4f}")

