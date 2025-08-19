import matplotlib.pyplot as plt
import diffrax as dfx
import jax
import jax.numpy as jnp
from quicksig.rde_bench.rough_volatility import BonesiniModelSpec

def plot_bonesini_rde(solution: dfx.Solution | list[dfx.Solution], model_spec: BonesiniModelSpec | list[BonesiniModelSpec]) -> None:
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
        plt.plot(ts, S, label=spec.name)
        
    plt.legend()
    plt.savefig("price_comparison.svg")

def plot_bonesini_monte_carlo(solution: dfx.Solution, model_spec: BonesiniModelSpec) -> None:
    fig, (ax_main, ax_marginal) = plt.subplots(1, 2, figsize=(12, 6), 
                                               gridspec_kw={'width_ratios': [3, 1]})
    
    ts_paths = jnp.asarray(solution.ts)
    ys_paths = jnp.asarray(solution.ys)

    if jnp.isnan(ys_paths).any():
        print(f"Warning: {model_spec.name} contains NaN")
    
    num_paths = ts_paths.shape[0]
    
    final_values = []
    initial_values = []
    for i in range(num_paths):
        ts = ts_paths[i]
        ys = ys_paths[i]
        S = ys if ys.ndim == 1 else ys[:, 0]
        ax_main.plot(ts, S, color='gray', alpha=0.6)
        final_values.append(S[-1])
        initial_values.append(S[0])
    
    # Calculate means
    mean_initial = jnp.mean(jnp.array(initial_values))
    mean_final = jnp.mean(jnp.array(final_values))
    
    # Plot mean lines
    x_min, x_max = ax_main.get_xlim()
    ax_main.axhline(y=mean_initial, color='red', linestyle='--', alpha=0.8, label=f't=0 Mean: {mean_initial:.4f}')
    ax_main.axhline(y=mean_final, color='blue', linestyle='--', alpha=0.8, label=f't=1 Mean: {mean_final:.4f}')
    ax_main.legend()
    
    ax_main.set_xlabel('Time')
    ax_main.set_ylabel('Price')
    ax_main.set_title(f'{model_spec.name} Monte Carlo')
    
    if final_values:
        ax_marginal.hist(final_values, bins=30, orientation='horizontal', color='gray', alpha=0.7)
        ax_marginal.set_xlabel('Frequency')
        ax_marginal.set_ylabel('Price')
        ax_marginal.set_title('t=1 Marginal')
        y_min, y_max = ax_main.get_ylim()
        ax_marginal.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(f"{model_spec.name.lower().replace(' ', '_')}_monte_carlo.png")