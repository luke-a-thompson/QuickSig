import time
from typing import TypedDict
from collections.abc import Iterable
import jax.numpy as jnp
import numpy as np
import argparse
import json
from pathlib import Path
from tests.test_helpers import generate_scalar_path
from quicksig.get_signatures import get_signature
import jax

PRNG = jax.random.PRNGKey(0)
DEVICE = jax.devices("gpu")[0]  # fail fast if absent


class BenchmarkResult(TypedDict):
    mean_us: float
    std_us: float


class Baselines(TypedDict):
    baselines: dict[str, BenchmarkResult]


# Allow up to 1 standard deviation difference
def is_regression(current_mean: float, current_std: float, baseline_mean: float, baseline_std: float) -> bool:
    """Return True if current performance is significantly worse than baseline."""
    return current_mean > baseline_mean + baseline_std


_DEFAULT_COMBINATIONS: list[tuple[int, int, int]] = [
    # (num_steps, channels, depth)
    (1000, 2, 3),
    (1000, 3, 4),
    (10000, 3, 3),
    (10000, 4, 3),
    (10000, 4, 4),
    (50000, 4, 4),
    (50000, 5, 3),
    (100000, 5, 4),
    (100000, 5, 5),
    (200000, 4, 3),
    (200000, 5, 4),
]


def _get_baseline_path() -> Path:
    """Return the path to the baseline JSON file."""
    return Path(__file__).parent / "benchmark_baseline.json"


def _load_baselines() -> Baselines:
    """Load baseline performance metrics."""
    baseline_path = _get_baseline_path()
    if not baseline_path.exists():
        return Baselines(baselines={})
    with open(baseline_path) as f:
        data = json.load(f)
        return Baselines(baselines=data.get("baselines", {}))


def _save_baselines(baselines: Baselines) -> None:
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
    compiled = get_signature(path, depth=depth, stream=False).compile()

    # Actual measurement
    start = time.perf_counter()
    _ = compiled(path).block_until_ready()
    return time.perf_counter() - start


def benchmark_signature(
    combinations: Iterable[tuple[int, int, int]] = _DEFAULT_COMBINATIONS,
    n_runs: int = 100,
    check_regression: bool = False,
    update_baseline: bool = False,
) -> bool:
    """
    Time each (steps, channels, depth) combination over `n_runs` evaluations and
    print the mean wall-clock latency in microseconds with standard deviation.
    Outliers (bottom and top 5%) are removed before calculating statistics.

    Parameters
    ----------
    combinations : Iterable[tuple[int, int, int]]
        List of (num_steps, channels, depth) combinations to benchmark
    n_runs : int
        Number of runs per combination
    check_regression : bool
        If True, check for performance regressions against baseline
    update_baseline : bool
        If True, update the baseline with current measurements

    Returns
    -------
    bool
        True if no regressions found or not checking for regressions, False otherwise
    """
    assert n_runs > 20, "n_runs must be greater than 20"
    baselines = _load_baselines()
    has_regression = False

    header = f"{'steps':<8}{'channels':>10}{'depth':>8}{'mean μs':>12}{'std μs':>12}{'prev mean':>12}{'% diff':>12}{'regression':>12}"
    print(header)
    print("-" * len(header))

    for _, (num_timesteps, channels, depth) in enumerate(combinations, 1):
        # Prepare path once for all runs
        path = _prepare_path(num_timesteps, channels)
        compiled = batch_signature.lower(path, depth=depth).compile()
        _ = compiled(path).block_until_ready()

        # Run measurements
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = compiled(path).block_until_ready()
            times.append(time.perf_counter() - start)

        times_us = np.array(times) * 1e6

        # Remove outliers (bottom and top 5%)
        sorted_times = np.sort(times_us)
        n_outliers = int(len(sorted_times) * 0.05)
        filtered_times = sorted_times[n_outliers:-n_outliers]

        mean_us = float(np.mean(filtered_times))
        std_us = float(np.std(filtered_times))

        # Check for regression
        key = f"{num_timesteps}_{channels}_{depth}"
        regression_info = ""
        prev_mean = ""
        pct_diff = ""
        if check_regression and key in baselines["baselines"]:
            baseline = baselines["baselines"][key]
            baseline_mean = baseline["mean_us"]
            baseline_std = baseline["std_us"]
            prev_mean = f"{baseline_mean:>12.2f}"
            pct_diff = f"{((mean_us - baseline_mean) / baseline_mean * 100):>11.1f}%"
            if is_regression(mean_us, std_us, baseline_mean, baseline_std):
                regression_info = f"⚠️ {mean_us/(baseline_mean):.1f}x"
                has_regression = True
            else:
                regression_info = "✅"
        else:
            prev_mean = "N/A"
            pct_diff = "N/A"

        print(f"{num_timesteps:<8}{channels:>10}{depth:>8}{mean_us:>12.2f}{std_us:>12.2f}{prev_mean:>12}{pct_diff:>12}{regression_info:>12}")

        # Update baseline if requested
        if update_baseline:
            baselines["baselines"][key] = BenchmarkResult(mean_us=mean_us, std_us=std_us)

    if update_baseline:
        _save_baselines(baselines)

    return not has_regression


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run signature benchmarks")
    parser.add_argument("--check-regression", action="store_true", help="Check for performance regressions", default=True)
    parser.add_argument("--update-baseline", action="store_true", help="Update baseline performance metrics")
    args = parser.parse_args()

    benchmark_signature(combinations=_DEFAULT_COMBINATIONS, n_runs=100, check_regression=args.check_regression, update_baseline=args.update_baseline)
