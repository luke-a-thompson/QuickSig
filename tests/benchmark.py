import time
from typing import Callable, Iterable, Any, TypedDict
import jax.numpy as jnp
import numpy as np
import argparse
import ast
import json
import os
from pathlib import Path
from tests.test_helpers import generate_scalar_path
from quicksig.path_signature import batch_signature_pure_jax
import jax

PRNG = jax.random.PRNGKey(0)
DEVICE = jax.devices("gpu")[0]  # fail fast if absent


class BenchmarkResult(TypedDict):
    mean_us: float
    std_us: float


class Baselines(TypedDict):
    baselines: dict[str, BenchmarkResult]


# Allow up to 1 standard deviation difference
def is_regression(current_mean: np.floating[Any], current_std: np.floating[Any], baseline_mean: float, baseline_std: float) -> bool:
    """Return True if current performance is significantly worse than baseline."""
    return float(current_mean) > baseline_mean + baseline_std


_DEFAULT_COMBINATIONS: list[tuple[int, int, int]] = [
    # (num_steps, channels, depth)
    (100, 1, 2),
    (100, 1, 3),
    (100, 2, 3),
    (250, 2, 3),
    (250, 2, 4),
    (250, 3, 3),
    (500, 3, 3),
    (500, 3, 4),
    (1000, 2, 3),
    (1000, 3, 4),
]


def _get_baseline_path() -> Path:
    """Return the path to the baseline JSON file."""
    return Path(__file__).parent / "benchmark_baseline.json"


def _load_baselines() -> Baselines:
    """Load baseline performance metrics."""
    baseline_path = _get_baseline_path()
    if not baseline_path.exists():
        return {"baselines": {}}
    with open(baseline_path) as f:
        return json.load(f)


def _save_baselines(baselines: dict) -> None:
    """Save baseline performance metrics."""
    with open(_get_baseline_path(), "w") as f:
        json.dump(baselines, f, indent=4)


def _prepare_path(num_timesteps: int, channels: int) -> jnp.ndarray:
    """Generate and prepare the path array once for reuse."""
    _, vals = generate_scalar_path(num_timesteps=num_timesteps, n_features=channels)
    return jnp.asarray(vals)[None, :, :]


def _time_once(path: jnp.ndarray, depth: int) -> float:
    """Return elapsed seconds for one forward pass."""
    # Warmup run
    _ = batch_signature_pure_jax(path, depth=depth).block_until_ready()

    # Actual measurement
    start = time.perf_counter()
    _ = batch_signature_pure_jax(path, depth=depth).block_until_ready()
    return time.perf_counter() - start


def benchmark_signature(
    combinations: Iterable[tuple[int, int, int]] = _DEFAULT_COMBINATIONS,
    n_runs: int = 10,
    printer: Callable[[str], None] = print,
    check_regression: bool = False,
    update_baseline: bool = False,
) -> bool:
    """
    Time each (steps, channels, depth) combination over `n_runs` evaluations and
    print the mean wall-clock latency in microseconds with standard deviation.

    Parameters
    ----------
    combinations : Iterable[tuple[int, int, int]]
        List of (num_steps, channels, depth) combinations to benchmark
    n_runs : int
        Number of runs per combination
    printer : Callable[[str], None]
        Function to print results
    check_regression : bool
        If True, check for performance regressions against baseline
    update_baseline : bool
        If True, update the baseline with current measurements

    Returns
    -------
    bool
        True if no regressions found or not checking for regressions, False otherwise
    """
    baselines = _load_baselines()
    has_regression = False

    header = f"{'steps':<8}{'channels':>10}{'depth':>8}{'mean μs':>12}{'std μs':>12}{'regression':>12}"
    printer(header)
    printer("-" * len(header))

    for _, (num_timesteps, channels, depth) in enumerate(combinations, 1):
        # Prepare path once for all runs
        path = _prepare_path(num_timesteps, channels)

        # Ensure JIT compilation is complete
        for _ in range(3):  # Multiple warmup runs for JIT
            _ = batch_signature_pure_jax(path, depth=depth).block_until_ready()

        # Run measurements
        times = [_time_once(path, depth) for _ in range(n_runs)]
        times_us = np.array(times) * 1e6
        mean_us = np.mean(times_us)
        std_us = np.std(times_us)

        # Check for regression
        key = f"{num_timesteps}_{channels}_{depth}"
        regression_info = ""
        if check_regression and key in baselines.get("baselines", {}):
            baseline = baselines["baselines"][key]
            baseline_mean = baseline["mean_us"]
            baseline_std = baseline["std_us"]
            if is_regression(mean_us, std_us, baseline_mean, baseline_std):
                regression_info = f"⚠️ {mean_us/(baseline_mean):.1f}x"
                has_regression = True
            else:
                regression_info = "✅"

        printer(f"{num_timesteps:<8}{channels:>10}{depth:>8}{mean_us:>12.2f}{std_us:>12.2f}{regression_info:>12}")

        # Update baseline if requested
        if update_baseline:
            if "baselines" not in baselines:
                baselines["baselines"] = {}
            baselines["baselines"][key] = {"mean_us": mean_us, "std_us": std_us}

    if update_baseline:
        _save_baselines(baselines)

    return not has_regression


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run signature benchmarks")
    parser.add_argument("--check-regression", action="store_true", help="Check for performance regressions")
    parser.add_argument("--update-baseline", action="store_true", help="Update baseline performance metrics")
    args = parser.parse_args()

    success = benchmark_signature(combinations=_DEFAULT_COMBINATIONS, n_runs=100, check_regression=args.check_regression, update_baseline=args.update_baseline)

    if not success:
        print("\n⚠️ Performance regression detected!")
        exit(1)
