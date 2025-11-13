import functools
import jax.numpy as jnp
from typing import Callable

from stochastax.controls.paths_types import Path


def _get_underlying_function(func: Callable) -> Callable:
    """Extract the underlying function from a partial or regular function."""
    return func.func if isinstance(func, functools.partial) else func


def augment_path(
    path: Path,
    augmentations: list[Callable[[Path], Path]],
) -> Path | list[Path] | list[list[Path]] | list[list[list[Path]]]:
    """Augment the path with a list of augmentations.

    Augmentations are applied in order:
    1. Path transformations (basepoint, time, lead-lag) are applied sequentially
    2. Windowing operations (non-overlapping or dyadic) are applied last

    If both windowers are provided, non-overlapping windower is applied first,
    then dyadic windower is applied to each resulting window.

    Args:
        path: The path to augment
        augmentations: List of augmentation functions (may be partial functions)

    Returns:
        - Path: If no windowing augmentations are applied
        - list[Path]: If only non_overlapping_windower is applied
        - list[list[Path]]: If only dyadic_windower is applied
        - list[list[list[Path]]]: If both windowers are applied (non_overlapping then dyadic)

    Raises:
        ValueError: If an unknown augmentation function is provided
    """
    # Known path transformation augmentations
    PATH_AUGMENTATIONS = {
        basepoint_augmentation,
        time_augmentation,
        lead_lag_augmentation,
    }

    # Classify augmentations into path transformations and windowers
    path_transforms: list[Callable[[Path], Path]] = []
    non_overlapping_windower_aug: Callable[[Path], list[Path]] | None = None
    dyadic_windower_aug: Callable[[Path], list[list[Path]]] | None = None

    for aug in augmentations:
        underlying_func = _get_underlying_function(aug)

        if underlying_func is non_overlapping_windower:
            non_overlapping_windower_aug = aug  # type: ignore
        elif underlying_func is dyadic_windower:
            dyadic_windower_aug = aug  # type: ignore
        elif underlying_func in PATH_AUGMENTATIONS:
            path_transforms.append(aug)
        else:
            raise ValueError(f"Unknown augmentation: {underlying_func}")

    # Apply path transformations sequentially
    augmented_path = path
    for transform in path_transforms:
        augmented_path = transform(augmented_path)

    # Apply windowing operations (at most one of each type)
    if non_overlapping_windower_aug is not None and dyadic_windower_aug is not None:
        # Apply non-overlapping windower first, then dyadic to each window
        windows = non_overlapping_windower_aug(augmented_path)
        return [dyadic_windower_aug(window) for window in windows]

    if non_overlapping_windower_aug is not None:
        return non_overlapping_windower_aug(augmented_path)

    if dyadic_windower_aug is not None:
        return dyadic_windower_aug(augmented_path)

    return augmented_path


def basepoint_augmentation(path: Path) -> Path:
    assert path.path.ndim == 2
    augmented_path_array = jnp.concatenate(
        [jnp.zeros((1, path.ambient_dimension)), path.path], axis=0
    )

    return Path(
        path=augmented_path_array,
        interval=(0, augmented_path_array.shape[0]),
    )


def time_augmentation(path: Path) -> Path:
    assert path.path.ndim == 2
    n_points = path.path.shape[0]
    time_points = jnp.linspace(0, 1, n_points).reshape(-1, 1)
    augmented_path_array = jnp.concatenate([time_points, path.path], axis=1)

    return Path(
        path=augmented_path_array,
        interval=path.interval,
    )


def non_overlapping_windower(path: Path, window_size: int) -> list[Path]:
    if window_size < 1:
        raise ValueError(f"window_size must be greater than 0. Got {window_size}")
    assert path.path.ndim == 2

    n_points = path.path.shape[0]
    if n_points == 0:
        return []

    split_indices = list(range(window_size, n_points, window_size))
    return path.split_at_time(split_indices)


def dyadic_windower(path: Path, window_depth: int) -> list[list[Path]]:
    assert path.path.ndim == 2
    assert window_depth >= 0

    seq_len = path.path.shape[0]

    if 2 ** (window_depth + 1) > seq_len:
        raise ValueError(
            f"window_depth {window_depth} is too large for path of length {seq_len}."
            f" This may result in windows with less than 2 points."
        )

    all_windows_info = []
    for d in range(window_depth + 1):
        num_windows = 2**d
        boundaries = jnp.floor(jnp.linspace(0, seq_len, num_windows + 1)).astype(jnp.int32)
        split_indices = boundaries[1:-1].tolist()
        path_windows_at_depth_d = path.split_at_time(split_indices)
        all_windows_info.append(path_windows_at_depth_d)

    return all_windows_info


def lead_lag_augmentation(leading_path: Path, lagging_path: Path) -> Path:
    if leading_path.path.shape[0] != lagging_path.path.shape[0]:
        raise ValueError(
            "The number of time steps in the leading and lagging paths must be the same."
        )
    if leading_path.interval != lagging_path.interval:
        raise ValueError("The intervals of the leading and lagging paths must be the same.")

    lag = jnp.concatenate([lagging_path.path[:1], lagging_path.path[:-1]], axis=0)

    lead_lag_path_array = jnp.concatenate([leading_path.path, lag], axis=1)

    return Path(
        path=lead_lag_path_array,
        interval=leading_path.interval,
    )
