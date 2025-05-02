import time
from typing import Callable
from collections.abc import Iterable
import jax.numpy as jnp
import numpy as np
import argparse
import ast
from tests.test_helpers import generate_scalar_path
from quicksig.path_signature import batch_signature_pure_jax
import jax

PRNG = jax.random.PRNGKey(0)
DEVICE = jax.devices("gpu")[0]  # fail fast if absent

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
) -> None:
    """
    Time each (steps, channels, depth) combination over `n_runs` evaluations and
    print the mean wall-clock latency in microseconds with standard deviation.

    Example
    -------
    >>> benchmark_signature()
    steps  channels  depth   mean μs    std μs
    100    1   2     120.45    5.23
    ...
    """

    header = f"{'steps':<8}{'channels':>10}{'depth':>8}{'mean μs':>12}{'std μs':>12}"
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

        printer(f"{num_timesteps:<8}{channels:>10}{depth:>8}{mean_us:>12.2f}{std_us:>12.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run path signature benchmarks")
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of runs per combination (default: 10)",
    )
    parser.add_argument(
        "--combinations",
        type=str,
        default=None,
        help="Custom combinations as a Python list of tuples, e.g. '[(100,1,2), (200,2,3)]'",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    combinations = ast.literal_eval(args.combinations) if args.combinations else _DEFAULT_COMBINATIONS
    benchmark_signature(combinations=combinations, n_runs=args.n_runs)
