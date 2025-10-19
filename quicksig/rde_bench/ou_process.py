import jax
import jax.numpy as jnp
import diffrax as dfx
from quicksig.rde.drivers import bm_driver
from quicksig.rde.rde_types import Path


def ou_process(
    key: jax.Array,
    timesteps: int,
    dim: int,
    theta: float,
    mu: float,
    sigma: float,
    x0: float | None = None,
) -> Path:
    """
    Generates an Ornstein-Uhlenbeck process path using Brownian motion and Heun solver.
    
    The OU process satisfies the SDE:
        dX_t = θ(μ - X_t)dt + σdW_t
    
    where:
        - θ (theta): rate of mean reversion
        - μ (mu): long-term mean
        - σ (sigma): volatility
        - W_t: Brownian motion
    
    Uses diffrax with Heun's method for numerical integration.
    
    Args:
        key: JAX random key
        timesteps: number of time steps
        dim: dimension of the OU process
        theta: rate of mean reversion (must be positive)
        mu: long-term mean
        sigma: volatility (must be non-negative)
        x0: initial value (defaults to mu if not provided)
    
    Returns:
        A Path object of shape (timesteps + 1, dim) representing the OU process.
    """
    if theta <= 0:
        raise ValueError(f"theta must be positive. Got {theta}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative. Got {sigma}")
    
    # Generate Brownian motion path
    bm_path = bm_driver(key, timesteps, dim)
    
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
    
    # Create linear interpolation control from Brownian motion
    bm_control = dfx.LinearInterpolation(ts=ts, ys=bm_path.path)
    
    # Build the SDE terms
    terms = dfx.MultiTerm(
        dfx.ODETerm(drift),
        dfx.ControlTerm(diffusion, control=bm_control)
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
