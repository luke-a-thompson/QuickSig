import time
from typing import Literal, TypedDict
import numpy as np
import argparse
import json
import time
from pathlib import Path
from typing import TypedDict, Literal
from tests.test_helpers import generate_scalar_path
import jax
from rich.console import Console
from rich.table import Table
from rich.text import Text
from tqdm import tqdm

import quicksig
import signax

KEY = jax.random.PRNGKey(42)
console = Console()


class BenchmarkResult(TypedDict):
    median_us: float
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
    (1000, 2, 4),
    (1000, 3, 3),
    (10000, 2, 3),
    (10000, 3, 3),
    # (1000, 3, 4),
    # (10000, 4, 3),
    # (10000, 4, 4),
    # (50000, 4, 4),
    # (50000, 5, 3),
    # (100000, 5, 4),
    # (100000, 5, 5),
    # (200000, 4, 3),
    # (200000, 5, 4),
    # (200000, 5, 5),
    # (200000, 60, 5),
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


def benchmark_signature(
    jax_device: Literal["cpu", "gpu"],
    n_runs: int = 100,
    check_regression: bool = False,
    update_baseline: bool = False,
) -> bool:
    """
    Time each (steps, channels, depth) combination over `n_runs` evaluations and
    print the mean wall-clock latency in microseconds with standard deviation.
    Outliers (bottom and top 2.5%) are removed before calculating statistics.

    Parameters
    ----------
    jax_device : Literal["cpu", "gpu"]
        The JAX device to use for computations
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
    device = jax.devices(jax_device)[0]
    with jax.default_device(device):
        baselines = _load_baselines()
        has_regression = False

        table = Table(title=f"Signature Benchmark Results on {jax_device.upper()}.")
        table.add_column("Steps", justify="left")
        table.add_column("Channels", justify="left")
        table.add_column("Depth", justify="left")
        table.add_column("QuickSig (μs)", justify="left")
        table.add_column("Signax (μs)", justify="left")

        for _, (num_timesteps, channels, depth) in (
            pbar := tqdm(enumerate(_DEFAULT_COMBINATIONS), desc="Benchmarking combinations", position=0, leave=False, total=len(_DEFAULT_COMBINATIONS))
        ):
            pbar.set_description(f"Benchmarking (steps={num_timesteps}, channels={channels}, depth={depth})")

            # Clear JAX cache before each combination to ensure consistent memory usage
            jax.clear_caches()

            path = generate_scalar_path(KEY, num_timesteps, channels)

            # QuickSig benchmark
            compiled_quicksig = jax.jit(lambda x: quicksig.signatures.compute_path_signature(x, depth=depth, mode="full").flatten())
            _ = compiled_quicksig(path)

            # Signax benchmark
            compiled_signax = jax.jit(lambda x: signax.signature(x, depth=depth))
            _ = compiled_signax(path)

            # Run measurements
            quicksig_times = []
            signax_times = []

            for _ in tqdm(range(n_runs), desc="QuickSig", position=1, leave=False):
                start = time.perf_counter()
                compiled_quicksig(path).block_until_ready()
                quicksig_times.append(time.perf_counter() - start)

            for _ in tqdm(range(n_runs), desc="Signax", position=1, leave=False):
                start = time.perf_counter()
                compiled_signax(path).block_until_ready()
                signax_times.append(time.perf_counter() - start)

            # Process QuickSig times
            quicksig_times_us = np.array(quicksig_times) * 1e6
            sorted_quicksig = np.sort(quicksig_times_us)
            n_outliers = int(len(sorted_quicksig) * 0.025)
            filtered_quicksig = sorted_quicksig[n_outliers:-n_outliers]
            quicksig_median = float(np.median(filtered_quicksig))
            quicksig_std = float(np.std(filtered_quicksig))

            # Process Signax times
            signax_times_us = np.array(signax_times) * 1e6
            sorted_signax = np.sort(signax_times_us)
            filtered_signax = sorted_signax[n_outliers:-n_outliers]
            signax_median = float(np.median(filtered_signax))
            signax_std = float(np.std(filtered_signax))

            # Check for regression
            key = f"{num_timesteps}_{channels}_{depth}"
            is_regression_case = False
            if check_regression and key in baselines["baselines"]:
                baseline = baselines["baselines"][key]
                baseline_median = baseline["median_us"]
                baseline_std = baseline["std_us"]
                if is_regression(quicksig_median, quicksig_std, baseline_median, baseline_std):
                    is_regression_case = True
                    has_regression = True

            # Format the medians with color
            quicksig_text = f"{quicksig_median:.1f} ± {2*quicksig_std:.1f}"
            signax_text = f"{signax_median:.1f} ± {2*signax_std:.1f}"

            if check_regression:
                quicksig_text = Text(quicksig_text, style="red" if is_regression_case else "green")

            table.add_row(str(num_timesteps), str(channels), str(depth), quicksig_text, signax_text)

            # Update baseline if requested
            if update_baseline:
                baselines["baselines"][key] = BenchmarkResult(median_us=quicksig_median, std_us=quicksig_std)

        if update_baseline:
            _save_baselines(baselines)

        console.print(table)
        return not has_regression


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run signature benchmarks")
    parser.add_argument("--check-regression", action="store_true", help="Check for performance regressions", default=True)
    parser.add_argument("--update-baseline", action="store_true", help="Update baseline performance metrics")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu", help="JAX device to use for computations")
    args = parser.parse_args()

    # benchmark_signature(jax_device="cpu", n_runs=100, check_regression=args.check_regression, update_baseline=args.update_baseline)
    benchmark_signature(jax_device="gpu", n_runs=100, check_regression=args.check_regression, update_baseline=args.update_baseline)
