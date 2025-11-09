import functools
import jax.numpy as jnp
from typing import Callable

from quicksig.controls.paths_types import Path


def augment_path(
    path: Path,
    augmentations: list[Callable[[Path], Path]],
) -> Path:
    """Augment the path with a list of augmentations."""
    path_augmentations = []
    sliding_windower_aug = None
    dyadic_windower_aug = None

    for aug in augmentations:
        func_to_check = aug.func if isinstance(aug, functools.partial) else aug
        if func_to_check is non_overlapping_windower:
            sliding_windower_aug = aug
        elif func_to_check is dyadic_windower:
            dyadic_windower_aug = aug
        elif func_to_check in [
            basepoint_augmentation,
            time_augmentation,
            lead_lag_augmentation,
        ]:
            path_augmentations.append(aug)
        else:
            raise ValueError(f"Unknown augmentation: {func_to_check}")

    current_path = path
    for aug in path_augmentations:
        current_path = aug(current_path)

    if sliding_windower_aug and dyadic_windower_aug:
        sliding_windows = sliding_windower_aug(current_path)
        return [dyadic_windower_aug(window) for window in sliding_windows]
    elif sliding_windower_aug:
        return sliding_windower_aug(current_path)
    elif dyadic_windower_aug:
        return dyadic_windower_aug(current_path)

    return current_path


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



