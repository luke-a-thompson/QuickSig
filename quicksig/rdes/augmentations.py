import functools
import jax
import jax.numpy as jnp
from typing import Any, Callable, Sequence

from quicksig.rdes.rde_types import Path, pathify


def augment_path(
    path: Path,
    augmentations: list[Callable[[Path], Any]],
) -> Any:
    """Augment the path with a list of augmentations.

    Path augmentations (`basepoint_augmentation`, `time_augmentation`) are
    applied first.

    Windowing augmentations (`non_overlapping_windower`, `dyadic_windower`) are
    applied after path augmentations.
    - If only one windower is provided, it is applied to the path.
    - If both `non_overlapping_windower` and `dyadic_windower` are provided,
      `non_overlapping_windower` is applied first, and then `dyadic_windower`
      is applied to each of the resulting sliding windows.

    Args:
        path: The input path as a `Path` object.
        augmentations: A list of callables to apply to the path. For augmentations
            that require parameters (e.g., `non_overlapping_windower`), use
            `functools.partial` to create the callable.

    Returns:
        The augmented path. The return type depends on the augmentations.
    """
    path_augmentations = []
    sliding_windower_aug = None
    dyadic_windower_aug = None

    for aug in augmentations:
        # Check if the augmentation is a windower, handling partials
        func_to_check = aug.func if isinstance(aug, functools.partial) else aug
        if func_to_check is non_overlapping_windower:
            sliding_windower_aug = aug
        elif func_to_check is dyadic_windower:
            dyadic_windower_aug = aug
        elif func_to_check in [basepoint_augmentation, time_augmentation, lead_lag_augmentation]:
            path_augmentations.append(aug)
        else:
            raise ValueError(f"Unknown augmentation: {func_to_check}")

    # Apply non-windower augmentations first
    current_path = path
    for aug in path_augmentations:
        current_path = aug(current_path)

    # Apply windowers
    if sliding_windower_aug and dyadic_windower_aug:
        sliding_windows = sliding_windower_aug(current_path)
        return [dyadic_windower_aug(window) for window in sliding_windows]
    elif sliding_windower_aug:
        return sliding_windower_aug(current_path)
    elif dyadic_windower_aug:
        return dyadic_windower_aug(current_path)

    return current_path


def basepoint_augmentation(path: Path) -> Path:
    r"""Augment the path with a basepoint at the origin.

    This function adds a single row of zeros at the beginning of the path,
    effectively ensuring the path starts at the origin.

    Args:
        path: Input path object.

    Returns:
        Augmented path object with a zero row prepended to the input path's data.
    """
    assert path.path.ndim == 2
    augmented_path_array = jnp.concatenate([jnp.zeros((1, path.ambient_dimension)), path.path], axis=0)

    return Path(
        path=augmented_path_array,
        interval=(0, augmented_path_array.shape[0]),
    )


def time_augmentation(path: Path) -> Path:
    r"""Augment the path with a time dimension.

    This function adds a time dimension to the path. The time dimension
    is a monotonically increasing series of values from 0 to 1, representing
    the progression along the path.

    Args:
        path: Input path object.

    Returns:
        Augmented path object with the time dimension prepended.
    """
    assert path.path.ndim == 2
    n_points = path.path.shape[0]
    time_points = jnp.linspace(0, 1, n_points).reshape(-1, 1)
    augmented_path_array = jnp.concatenate([time_points, path.path], axis=1)

    return Path(
        path=augmented_path_array,
        interval=path.interval,
    )


def non_overlapping_windower(path: Path, window_size: int) -> list[Path]:
    r"""Create non-overlapping windows over a path.

    This function creates non-overlapping windows of specified size over the input path.
    The last window may be smaller if the path length is not divisible by window_size.

    Args:
        path: Input path object.
        window_size: Size of each window.

    Returns:
        List of `Path` objects, each representing a window.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be greater than 0. Got {window_size}")
    assert path.path.ndim == 2

    n_points = path.path.shape[0]
    if n_points == 0:
        return []

    split_indices = list(range(window_size, n_points, window_size))
    return path.split_at_time(split_indices)


def dyadic_windower(path: Path, window_depth: int) -> list[list[Path]]:
    r"""Create dyadic windows over a path.

    This function divides the path into a series of dyadic windows.
    At depth `d`, the path is split into `2**d` windows.

    Args:
        path: Input path object.
        window_depth: The maximum depth of the dyadic windows.

    Returns:
        A list of lists of `Path` objects. For each depth `d`, the inner list
        contains the `Path` objects for that level.

    Raises:
        ValueError: If `window_depth` is so large that it may create windows
                    with fewer than two points.
    """
    assert path.path.ndim == 2
    assert window_depth >= 0

    seq_len = path.path.shape[0]

    if 2 ** (window_depth + 1) > seq_len:
        raise ValueError(f"window_depth {window_depth} is too large for path of length {seq_len}." f" This may result in windows with less than 2 points.")

    all_windows_info = []
    for d in range(window_depth + 1):
        num_windows = 2**d
        boundaries = jnp.floor(jnp.linspace(0, seq_len, num_windows + 1)).astype(jnp.int32)
        split_indices = boundaries[1:-1].tolist()
        path_windows_at_depth_d = path.split_at_time(split_indices)
        all_windows_info.append(path_windows_at_depth_d)

    return all_windows_info


def lead_lag_augmentation(leading_path: Path, lagging_path: Path) -> Path:
    r"""Augment the path with a lead-lag transformation.

    This function creates a lead-lag transformation by combining a leading
    path and a lagging path. The lagging path is shifted by one time step.

    The two paths must have the same number of time steps and the same interval.

    Args:
        leading_path: The path that leads.
        lagging_path: The path that lags.

    Returns:
        The augmented `Path` object.

    Raises:
        ValueError: If the number of time steps or intervals in the leading
                    and lagging paths are different.
    """
    if leading_path.path.shape[0] != lagging_path.path.shape[0]:
        raise ValueError("The number of time steps in the leading and lagging paths must be the same.")
    if leading_path.interval != lagging_path.interval:
        raise ValueError("The intervals of the leading and lagging paths must be the same.")

    # Lag the lagging path by one time step.
    lag = jnp.concatenate([lagging_path.path[:1], lagging_path.path[:-1]], axis=0)

    # Concatenate the leading path and the lagged path.
    lead_lag_path_array = jnp.concatenate([leading_path.path, lag], axis=1)

    return Path(
        path=lead_lag_path_array,
        interval=leading_path.interval,
    )


if __name__ == "__main__":
    path_array = jnp.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0],
            [17.0, 18.0],
            [19.0, 20.0],
            [21.0, 22.0],
            [23.0, 24.0],
            [25.0, 26.0],
            [27.0, 28.0],
            [29.0, 30.0],
            [31.0, 32.0],
        ]
    )
    path_obj = pathify(path_array)
    depth = 2
    print(path_obj)
    print(f"\nDyadic windows with depth {depth}:")

    windows_info = dyadic_windower(path_obj, depth)

    for d, windows_at_depth in enumerate(windows_info):
        print(f"\n--- Depth {d} ---")
        print(f"Number of windows: {len(windows_at_depth)}")
        for i, window in enumerate(windows_at_depth):
            print(f"  Window {i} (length {window.path.shape[0]}):")
            print(window)

    print(non_overlapping_windower(path_obj, 8))
