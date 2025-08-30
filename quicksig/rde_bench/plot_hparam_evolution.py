import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Optional
from pathlib import Path

from quicksig.rde_bench.rp_kalman import (
    RBergomiEnsemble,
    constrain_theta,
    run_ensemble_kalman_filter,
    sample_unconstrained_priors,
    simulate_observation_logS,
    ensemble_kalman_filter_step,
)


def track_ensemble_evolution(
    ens0: RBergomiEnsemble,
    t: jax.Array,
    y_obs: jax.Array,
    R: float,
    dW_V: jax.Array,
    dW_indep: jax.Array,
) -> dict[str, jax.Array]:
    """
    Modified filter that tracks ensemble evolution at each time step.

    Returns:
        Dictionary with keys:
        - 'constrained_params': shape (K+1, N, 4) - [H, nu, rho, v0] for each particle at each time
        - 'unconstrained_params': shape (K+1, N, 4) - [tH, tNu, tRho, tV0] for each particle at each time
        - 'states': shape (K+1, N, 2) - [S, V] for each particle at each time
        - 'ensemble_means': shape (K+1, 4) - mean parameters at each time
        - 'ensemble_stds': shape (K+1, 4) - std parameters at each time
    """
    K = t.shape[0] - 1
    N = ens0.S.shape[0]

    # Storage arrays
    constrained_history = jnp.zeros((K + 1, N, 4))
    unconstrained_history = jnp.zeros((K + 1, N, 4))
    state_history = jnp.zeros((K + 1, N, 2))
    mean_history = jnp.zeros((K + 1, 4))
    std_history = jnp.zeros((K + 1, 4))

    # Store initial conditions
    H0, nu0, rho0, v0_0 = constrain_theta(ens0.tH, ens0.tNu, ens0.tRho, ens0.tV0)
    constrained_history = constrained_history.at[0].set(jnp.stack([H0, nu0, rho0, v0_0], axis=1))
    unconstrained_history = unconstrained_history.at[0].set(jnp.stack([ens0.tH, ens0.tNu, ens0.tRho, ens0.tV0], axis=1))
    state_history = state_history.at[0].set(jnp.stack([ens0.S, ens0.V], axis=1))
    mean_history = mean_history.at[0].set(jnp.array([jnp.mean(H0), jnp.mean(nu0), jnp.mean(rho0), jnp.mean(v0_0)]))
    std_history = std_history.at[0].set(jnp.array([jnp.std(H0), jnp.std(nu0), jnp.std(rho0), jnp.std(v0_0)]))

    # Run filter step by step
    ens = ens0
    for k in range(K):
        carry = (ens, R, t, y_obs, dW_V, dW_indep)
        (carry_next, _) = ensemble_kalman_filter_step(carry, k)
        ens, *_ = carry_next

        # Store results
        H_k, nu_k, rho_k, v0_k = constrain_theta(ens.tH, ens.tNu, ens.tRho, ens.tV0)
        constrained_history = constrained_history.at[k + 1].set(jnp.stack([H_k, nu_k, rho_k, v0_k], axis=1))
        unconstrained_history = unconstrained_history.at[k + 1].set(jnp.stack([ens.tH, ens.tNu, ens.tRho, ens.tV0], axis=1))
        state_history = state_history.at[k + 1].set(jnp.stack([ens.S, ens.V], axis=1))
        mean_history = mean_history.at[k + 1].set(jnp.array([jnp.mean(H_k), jnp.mean(nu_k), jnp.mean(rho_k), jnp.mean(v0_k)]))
        std_history = std_history.at[k + 1].set(jnp.array([jnp.std(H_k), jnp.std(nu_k), jnp.std(rho_k), jnp.std(v0_k)]))

    return {
        "constrained_params": constrained_history,
        "unconstrained_params": unconstrained_history,
        "states": state_history,
        "ensemble_means": mean_history,
        "ensemble_stds": std_history,
        "time_grid": t,
    }


def plot_parameter_cloud_evolution(
    evolution_data: dict[str, jax.Array], true_params: dict[str, float], param_pairs: Optional[list[tuple[str, str]]] = None, save_path: Optional[Path] = None
) -> None:
    """
    Plot 2D projections of parameter cloud evolution.

    Args:
        evolution_data: Output from track_ensemble_evolution
        true_params: Dictionary with true parameter values {'H': val, 'nu': val, 'rho': val, 'v0': val}
        param_pairs: List of parameter pairs to plot, defaults to all meaningful pairs
        save_path: Optional path to save figure
    """
    if param_pairs is None:
        param_pairs = [("H", "rho"), ("nu", "v0"), ("H", "nu"), ("rho", "v0")]

    param_names = ["H", "nu", "rho", "v0"]
    param_indices = {name: i for i, name in enumerate(param_names)}

    constrained_params = evolution_data["constrained_params"]  # (K+1, N, 4)
    time_grid = evolution_data["time_grid"]
    K = len(time_grid) - 1

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Color map for time evolution
    colors = plt.cm.viridis(np.linspace(0, 1, K + 1))

    for idx, (param1, param2) in enumerate(param_pairs):
        ax = axes[idx]

        p1_idx = param_indices[param1]
        p2_idx = param_indices[param2]

        # Plot particle clouds at selected time points
        time_points = [0, K // 4, K // 2, 3 * K // 4, K]

        for i, t_idx in enumerate(time_points):
            p1_vals = constrained_params[t_idx, :, p1_idx]
            p2_vals = constrained_params[t_idx, :, p2_idx]

            alpha = 0.3 if i < len(time_points) - 1 else 0.8
            size = 15 if i < len(time_points) - 1 else 25

            ax.scatter(p1_vals, p2_vals, c=[colors[t_idx]], alpha=alpha, s=size, label=f"t={time_grid[t_idx]:.2f}")

            # Add confidence ellipse for final time point
            if i == len(time_points) - 1:
                mean_p1, mean_p2 = jnp.mean(p1_vals), jnp.mean(p2_vals)
                cov_matrix = jnp.cov(jnp.stack([p1_vals, p2_vals]))
                eigenvals, eigenvecs = jnp.linalg.eigh(cov_matrix)

                # 95% confidence ellipse (2σ)
                angle = jnp.degrees(jnp.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                width, height = 2 * 2 * jnp.sqrt(eigenvals)  # 2σ ellipse

                ellipse = Ellipse((mean_p1, mean_p2), width, height, angle=angle, fill=False, color="red", linestyle="--", linewidth=2, label="95% confidence")
                ax.add_patch(ellipse)

        # Mark true parameter values
        ax.plot(true_params[param1], true_params[param2], "r*", markersize=15, label="True values", markeredgecolor="black", markeredgewidth=1)

        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_title(f"Parameter Cloud Evolution: {param1} vs {param2}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        out_dir = Path(__file__).resolve().parents[2] / "docs" / "assets"
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "hparam_cloud_evolution.png", dpi=300, bbox_inches="tight")

    plt.show()


def plot_ensemble_statistics_evolution(evolution_data: dict[str, jax.Array], true_params: dict[str, float], save_path: Optional[Path] = None) -> None:
    """
    Plot time series of ensemble mean and spread evolution.
    """
    param_names = ["H", "nu", "rho", "v0"]
    means = evolution_data["ensemble_means"]  # (K+1, 4)
    stds = evolution_data["ensemble_stds"]  # (K+1, 4)
    time_grid = evolution_data["time_grid"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, param in enumerate(param_names):
        ax = axes[i]

        # Plot ensemble mean
        ax.plot(time_grid, means[:, i], "b-", linewidth=2, label="Ensemble mean")

        # Plot confidence bands (mean ± 2σ)
        upper = means[:, i] + 2 * stds[:, i]
        lower = means[:, i] - 2 * stds[:, i]
        ax.fill_between(time_grid, lower, upper, alpha=0.3, color="blue", label="±2σ band")

        # Plot true value
        ax.axhline(true_params[param], color="red", linestyle="--", linewidth=2, label="True value")

        ax.set_xlabel("Time")
        ax.set_ylabel(param)
        ax.set_title(f"{param} Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        out_dir = Path(__file__).resolve().parents[2] / "docs" / "assets"
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "hparam_statistics_evolution.png", dpi=300, bbox_inches="tight")

    plt.show()


def create_parameter_cloud_animation(
    evolution_data: dict[str, jax.Array], true_params: dict[str, float], param_pair: tuple[str, str] = ("H", "rho"), save_path: Optional[Path] = None
) -> None:
    """
    Create animated visualization of parameter cloud evolution.
    """
    param_names = ["H", "nu", "rho", "v0"]
    param_indices = {name: i for i, name in enumerate(param_names)}

    constrained_params = evolution_data["constrained_params"]  # (K+1, N, 4)
    time_grid = evolution_data["time_grid"]
    K = len(time_grid) - 1

    param1, param2 = param_pair
    p1_idx = param_indices[param1]
    p2_idx = param_indices[param2]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Set up plot limits
    p1_all = constrained_params[:, :, p1_idx].flatten()
    p2_all = constrained_params[:, :, p2_idx].flatten()
    p1_range = jnp.max(p1_all) - jnp.min(p1_all)
    p2_range = jnp.max(p2_all) - jnp.min(p2_all)

    ax.set_xlim(jnp.min(p1_all) - 0.1 * p1_range, jnp.max(p1_all) + 0.1 * p1_range)
    ax.set_ylim(jnp.min(p2_all) - 0.1 * p2_range, jnp.max(p2_all) + 0.1 * p2_range)

    # Mark true values
    ax.plot(true_params[param1], true_params[param2], "r*", markersize=15, label="True values", markeredgecolor="black", markeredgewidth=1)

    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_title(f"Parameter Cloud Evolution: {param1} vs {param2}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Animation function
    scat = ax.scatter([], [], s=50, alpha=0.6)
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=12, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    def animate(frame: int) -> tuple[Any, Any]:
        p1_vals = constrained_params[frame, :, p1_idx]
        p2_vals = constrained_params[frame, :, p2_idx]

        # Update scatter plot
        scat.set_offsets(jnp.column_stack([p1_vals, p2_vals]))

        # Color by progress through time
        colors = plt.cm.viridis(frame / K)
        scat.set_color(colors)

        # Update time text
        time_text.set_text(f"Time: {time_grid[frame]:.3f}")

        return scat, time_text

    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=K + 1, interval=100, blit=True, repeat=True)

    if save_path:
        ani.save(save_path, writer="pillow", fps=10)
    else:
        out_dir = Path(__file__).resolve().parents[2] / "docs" / "assets"
        out_dir.mkdir(parents=True, exist_ok=True)
        ani.save(out_dir / f"hparam_cloud_animation_{param1}_{param2}.gif", writer="pillow", fps=10)

    plt.show()


def demonstrate_hparam_cloud_visualization() -> None:
    """
    Demonstration of hyperparameter cloud visualization capabilities.
    """
    print("Demonstrating hyperparameter cloud visualization...")

    # Setup parameters
    key = jax.random.key(42)
    K = 500  # Reduced for faster demo
    N = 100  # Reduced for faster demo
    R = 1e-5

    # True parameters
    true_params = {"H": 0.18, "nu": 1.40, "rho": -0.6, "v0": 0.04}

    # Generate synthetic data - focus on very early dynamics
    t = jnp.linspace(0.0, 0.1, K + 1)

    key_dw_V, key_dW_indep, key_prior = jax.random.split(key, 3)
    dW_V = jax.random.normal(key_dw_V, (K,)) * jnp.sqrt(1.0 / K)
    dW_independent = jax.random.normal(key_dW_indep, (K,)) * jnp.sqrt(1.0 / K)

    # Simulate observations
    y_obs = simulate_observation_logS(1.0, true_params["v0"], true_params["H"], true_params["nu"], true_params["rho"], true_params["v0"], t, dW_V, dW_independent)

    # Initialize ensemble with dispersed priors
    tH0, tNu0, tRho0, tV00 = sample_unconstrained_priors(key_prior, N)
    ens0 = RBergomiEnsemble(
        S=jnp.full((N,), 1.0),
        V=jnp.full((N,), true_params["v0"]),  # Start with true v0 for simplicity
        X_driver=jnp.zeros((N,)),
        W_driver=jnp.zeros((N,)),
        tH=tH0,
        tNu=tNu0,
        tRho=tRho0,
        tV0=tV00,
    )

    print("Tracking ensemble evolution...")
    evolution_data = track_ensemble_evolution(ens0, t, y_obs, R, dW_V, dW_independent)

    print("Creating visualizations...")

    # 1. Parameter cloud evolution plots
    plot_parameter_cloud_evolution(evolution_data, true_params)

    # 2. Ensemble statistics evolution
    plot_ensemble_statistics_evolution(evolution_data, true_params)

    # 3. Create animation for one parameter pair
    create_parameter_cloud_animation(evolution_data, true_params, ("H", "rho"))

    print("Visualization demonstration complete!")


if __name__ == "__main__":
    demonstrate_hparam_cloud_visualization()
