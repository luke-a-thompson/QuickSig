import matplotlib.pyplot as plt
import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.scipy.stats
from quicksig.rde_bench.rough_volatility import BonesiniModelSpec
from pathlib import Path


def plot_bonesini_rde(solution: dfx.Solution | list[dfx.Solution], model_spec: BonesiniModelSpec | list[BonesiniModelSpec], output_dir: Path | None = None) -> None:
    if isinstance(model_spec, BonesiniModelSpec):
        model_spec = [model_spec]

    if isinstance(solution, dfx.Solution):
        solution = [solution]

    if len(solution) != len(model_spec):
        raise ValueError("Number of solutions and model specifications must match.")

    for sol, spec in zip(solution, model_spec):
        ys = jnp.asarray(sol.ys)
        if jnp.isnan(ys).any():
            print(f"Warning: {spec.name} contains NaN")
            continue
        ts = jnp.asarray(sol.ts)
        S = ys if ys.ndim == 1 else ys[:, 0]
        log_S = jnp.log(S)
        plt.plot(ts, log_S, label=spec.name)

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Log-Price")
    
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "docs" / "rde_bench" / "rough_volatility"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "price_comparison.png")
    print(f"Saved price_comparison.png to {output_dir}")
    plt.close()


def plot_bonesini_monte_carlo(solution: dfx.Solution, model_spec: BonesiniModelSpec, plot_variance: bool = False, use_log_price: bool = True, output_dir: Path | None = None) -> None:
    fig, (ax_main, ax_marginal) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [3, 1]})

    ts_paths = jnp.asarray(solution.ts)
    ys_paths = jnp.asarray(solution.ys)

    if jnp.isnan(ys_paths).any():
        print(f"Warning: {model_spec.name} contains NaN")

    num_paths = ts_paths.shape[0]

    final_values = []
    initial_values = []
    ax_var = None
    for i in range(num_paths):
        ts = ts_paths[i]
        ys = ys_paths[i]
        S = ys if ys.ndim == 1 else ys[:, 0]
        price_data = jnp.log(S) if use_log_price else S
        ax_main.plot(ts, price_data, color="gray", alpha=0.6)
        final_values.append(price_data[-1])
        initial_values.append(price_data[0])

        # Optionally plot variance from second dimension on a twin y-axis
        if plot_variance and ys.ndim > 1 and ys.shape[1] > 1:
            if ax_var is None:
                ax_var = ax_main.twinx()
                ax_var.set_ylabel("Variance", color="tab:green")
                ax_var.tick_params(axis="y", labelcolor="tab:green")
            V = ys[:, 1]
            ax_var.plot(ts, V, color="tab:green", alpha=0.3)

    # Calculate means
    mean_initial = jnp.mean(jnp.array(initial_values))
    mean_final = jnp.mean(jnp.array(final_values))

    # Plot mean lines
    x_min, x_max = ax_main.get_xlim()
    price_label = "log" if use_log_price else "price"
    ax_main.axhline(y=mean_initial, color="red", linestyle="--", alpha=0.8, label=f"t=0 Mean ({price_label}): {mean_initial:.4f}")
    ax_main.axhline(y=mean_final, color="blue", linestyle="--", alpha=0.8, label=f"t=1 Mean ({price_label}): {mean_final:.4f}")
    ax_main.legend()

    ax_main.set_xlabel("Time")
    ax_main.set_ylabel("Log-Price" if use_log_price else "Price")
    ax_main.set_title(f"{model_spec.name} Monte Carlo")

    if final_values:
        ax_marginal.hist(final_values, bins=30, orientation="horizontal", color="gray", alpha=0.7, density=True)

        # Simple fat-tailedness check: compare to normal distribution
        final_array = jnp.array(final_values)
        mean_val = jnp.mean(final_array)
        std_val = jnp.std(final_array)

        # Generate normal distribution with same mean/std
        y_range = jnp.linspace(mean_val - 3 * std_val, mean_val + 3 * std_val, 100)
        normal_pdf = jnp.exp(-0.5 * ((y_range - mean_val) / std_val) ** 2) / (std_val * jnp.sqrt(2 * jnp.pi))

        # Plot normal distribution for comparison
        ax_marginal.plot(normal_pdf, y_range, color="red", linestyle="--", alpha=0.8, label="Normal")

        # Simple fat-tailedness indicator: compare tail probabilities
        tail_threshold = 2 * std_val
        empirical_tail_prob = jnp.mean(jnp.abs(final_array - mean_val) > tail_threshold)
        normal_tail_prob = 2 * (1 - jax.scipy.stats.norm.cdf(tail_threshold, 0, std_val))

        ax_marginal.set_xlabel("Density")
        ax_marginal.set_ylabel("Log-Price" if use_log_price else "Price")
        title = f"t=1 Marginal ({price_label})"
        ax_marginal.set_title(title)
        ax_marginal.legend()
        y_min, y_max = ax_main.get_ylim()
        ax_marginal.set_ylim(y_min, y_max)

    plt.tight_layout()
    
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "docs" / "rde_bench" / "rough_volatility"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{model_spec.name.lower().replace(' ', '_')}_monte_carlo.png"
    plt.savefig(output_dir / filename)
    print(f"Saved {filename} to {output_dir}")
    plt.close()


if __name__ == "__main__":
    from quicksig.rde_bench.rough_volatility import (
        make_black_scholes_model_spec,
        make_bergomi_model_spec,
        make_rough_bergomi_model_spec,
        get_bonesini_noise_drivers,
        solve_bonesini_rde_from_drivers,
    )

    noise_timesteps = 1000
    rde_timesteps = 5000
    num_paths = 3000

    output_dir = Path(__file__).resolve().parents[1] / "docs" / "rde_bench" / "rough_volatility"
    output_dir.mkdir(parents=True, exist_ok=True)

    # BLACK-SCHOLES
    print("Generating Black-Scholes Monte Carlo...")
    black_scholes_model_spec = make_black_scholes_model_spec(v_0=0.04)
    keys_bs = jax.random.split(jax.random.key(42), num_paths)
    y0_bs, X_bs, W_bs = jax.vmap(lambda key: get_bonesini_noise_drivers(key, noise_timesteps, black_scholes_model_spec, s_0=1.0))(keys_bs)
    solve_vmap_bs = jax.vmap(lambda y0, X, W: solve_bonesini_rde_from_drivers(y0, X, W, black_scholes_model_spec, noise_timesteps, rde_timesteps))
    solutions_bs = solve_vmap_bs(y0_bs, X_bs, W_bs)
    plot_bonesini_monte_carlo(solutions_bs, black_scholes_model_spec, output_dir=output_dir)

    # BERGOMI
    print("Generating Bergomi Monte Carlo...")
    bergomi_model_spec = make_bergomi_model_spec(v_0=0.0, rho=-0.848)
    keys_b = jax.random.split(jax.random.key(42), num_paths)
    y0_b, X_b, W_b = jax.vmap(lambda key: get_bonesini_noise_drivers(key, noise_timesteps, bergomi_model_spec, s_0=1.0))(keys_b)
    solve_vmap_b = jax.vmap(lambda y0, X, W: solve_bonesini_rde_from_drivers(y0, X, W, bergomi_model_spec, noise_timesteps, rde_timesteps))
    solutions_b = solve_vmap_b(y0_b, X_b, W_b)
    plot_bonesini_monte_carlo(solutions_b, bergomi_model_spec, output_dir=output_dir)

    # ROUGH BERGOMI
    print("Generating Rough Bergomi Monte Carlo...")
    rough_bergomi_model_spec = make_rough_bergomi_model_spec(v_0=0.04, nu=1.991, hurst=0.25, rho=-0.848)
    keys_rb = jax.random.split(jax.random.key(42), num_paths)
    y0_rb, X_rb, W_rb = jax.vmap(lambda key: get_bonesini_noise_drivers(key, noise_timesteps, rough_bergomi_model_spec, s_0=100.0))(keys_rb)
    solve_vmap_rb = jax.vmap(lambda y0, X, W: solve_bonesini_rde_from_drivers(y0, X, W, rough_bergomi_model_spec, noise_timesteps, rde_timesteps))
    solutions_rb = solve_vmap_rb(y0_rb, X_rb, W_rb)
    plot_bonesini_monte_carlo(solutions_rb, rough_bergomi_model_spec, use_log_price=True, output_dir=output_dir)

    print(f"\nAll plots saved to {output_dir}")
