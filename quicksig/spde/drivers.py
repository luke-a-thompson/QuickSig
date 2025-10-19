import math
import jax
import jax.numpy as jnp


def spacetime_white_noise(
    key: jax.Array,
    num_steps: int,
    grid_shape: tuple[int, ...],
    dt: float,
    dx: float | tuple[float, ...] = 1.0,
    sigma: float = 1.0,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """
    Generate discretized space-time white noise as independent Gaussian samples.

    The returned array has shape `sample_shape + (num_steps,) + grid_shape` and entries
    distributed as Normal(0, variance) with variance = sigma^2 * dt / prod(dx), which
    corresponds to the standard finite-difference scaling for space-time white noise
    (i.e., d-dimensional spatial grid with cell volume prod(dx)).

    Args:
        key: PRNG key for JAX randomness.
        num_steps: Number of time steps.
        grid_shape: Spatial grid shape as a tuple, e.g., (Nx,) or (Nx, Ny, ...).
        dt: Time step size (> 0).
        dx: Spatial grid spacing; scalar or tuple matching `grid_shape` length.
        sigma: Noise intensity multiplier.
        sample_shape: Optional leading batch/sample shape.

    Returns:
        JAX array of shape `sample_shape + (num_steps,) + grid_shape` containing the noise.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive. Got {num_steps}.")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive. Got {dt}.")
    if any(d <= 0 for d in grid_shape):
        raise ValueError(f"All grid dimensions must be positive. Got {grid_shape}.")

    spatial_dimensionality = len(grid_shape)
    # Cell volume in d-dimensions; scaling yields Var = sigma^2 * dt / volume
    if isinstance(dx, tuple):
        cell_volume: float = float(math.prod(dx))
    else:
        cell_volume = float(dx) ** spatial_dimensionality
    scale = sigma * math.sqrt(dt / cell_volume)

    shape: tuple[int, ...] = sample_shape + (num_steps,) + grid_shape
    noise: jax.Array = jax.random.normal(key, shape) * scale
    return noise


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Demo: 2D grid animated over time
    key = jax.random.PRNGKey(0)
    T, Nx, Ny = 200, 64, 64
    dt = 1.0 / T
    dx = (1.0 / Nx, 1.0 / Ny)

    dW = spacetime_white_noise(
        key=key,
        num_steps=T,
        grid_shape=(Nx, Ny),
        dt=dt,
        dx=dx,
        sigma=1.0,
    )
    dW_np = jax.device_get(dW)

    # Consistent color scale across frames
    v = float(jnp.percentile(jnp.abs(dW_np), 99.0))

    fig, ax = plt.subplots()
    im = ax.imshow(dW_np[0], cmap="RdBu_r", vmin=-v, vmax=v, origin="lower")
    ax.set_title("Space-time white noise (2D)")
    plt.colorbar(im, ax=ax)

    def update(frame: int):
        im.set_data(dW_np[frame])
        ax.set_xlabel(f"t index: {frame}")
        return (im,)

    ani = FuncAnimation(fig, update, frames=T, interval=40, blit=True)
    plt.tight_layout()
    try:
        ani.save("spacetime_white_noise.mp4", fps=25)
    except Exception:
        plt.savefig("spacetime_white_noise.png", dpi=150)
