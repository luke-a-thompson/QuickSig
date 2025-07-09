import jax
import jax.numpy as jnp
from typing import Callable


def augment_path(
    path: jax.Array,
    augmentations: list[Callable[[jax.Array], jax.Array]],
    window_depth: int | None = None,
    window_size: int | None = None,
) -> jax.Array | list[jax.Array]:
    """Augment the path with a list of augmentations."""
    if dyadic_windower in augmentations and window_depth is None:
        raise ValueError("window_depth must be provided if dyadic_windower is in augmentations")
    if dyadic_windower in augmentations and window_size is None:
        raise ValueError("window_size must be provided if dyadic_windower is in augmentations")
    windower_count = sum(1 for aug in augmentations if aug in [sliding_windower, dyadic_windower])
    if windower_count > 1:
        raise ValueError("Only one windower (sliding_windower or dyadic_windower) can be used at a time")

    if basepoint_augmentation in augmentations:
        path = basepoint_augmentation(path)

    if time_augmentation in augmentations:
        path = time_augmentation(path)

    if dyadic_windower in augmentations and window_depth is not None:
        path = dyadic_windower(path, window_depth, window_size)

    if sliding_windower in augmentations and window_size is not None:
        path = sliding_windower(path, window_size)

    return path


def basepoint_augmentation(path: jax.Array) -> jax.Array:
    r"""Augment the path with a basepoint at the origin.

    This function adds a single row of zeros at the beginning of the path,
    effectively ensuring the path starts at the origin.

    Args:
        path: Input path array of shape `(n_points, n_dims)`.

    Returns:
        Augmented path array of shape `(n_points + 1, n_dims)` with a zero row
        prepended to the input path.

    Examples:
        >>> import jax.numpy as jnp
        >>> from quicksig.augmentations import basepoint_augmentation
        >>> path = jnp.array([[1, 2], [3, 4]])
        >>> basepoint_augmentation(path)
        Array([[0., 0.],
               [1., 2.],
               [3., 4.]], dtype=float32)
    """
    assert path.ndim == 2
    augmented_path = jnp.concatenate([jnp.zeros((1, path.shape[1])), path], axis=0)

    return augmented_path


def time_augmentation(path: jax.Array) -> jax.Array:
    r"""Augment the path with a time dimension.

    This function adds a time dimension to the path. The time dimension
    is a monotonically increasing series of values from 0 to 1, representing
    the progression along the path.

    Args:
        path: Input path array of shape `(n_points, n_dims)`.

    Returns:
        Augmented path array of shape `(n_points, n_dims + 1)`, with the
        time dimension prepended to the original dimensions.

    Examples:
        >>> import jax.numpy as jnp
        >>> from quicksig.augmentations import time_augmentation
        >>> path = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> time_augmentation(path)
        Array([[0. , 1. , 2. ],
               [0.5, 3. , 4. ],
               [1. , 5. , 6. ]], dtype=float32)
    """
    assert path.ndim == 2
    n_points = path.shape[0]
    time_points = jnp.linspace(0, 1, n_points).reshape(-1, 1)
    augmented_path = jnp.concatenate([time_points, path], axis=1)

    return augmented_path


def sliding_windower(path: jax.Array, window_size: int) -> list[jax.Array]:
    r"""Create non-overlapping windows over a path using efficient reshape operations.

    This function creates non-overlapping windows of specified size over the input path.
    The last window may be smaller if the path length is not divisible by window_size.

    Args:
        path: Input path array of shape `(n_points, n_dims)`.
        window_size: Size of each window.

    Returns:
        List of window arrays. Each window has shape `(window_size, n_dims)` except
        the last window which may be smaller if n_points is not divisible by window_size.

    Examples:
        >>> import jax.numpy as jnp
        >>> from quicksig.augmentations import sliding_windower
        >>> path = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        >>> windows = sliding_windower(path, 2)
        >>> [w.shape for w in windows]
        [(2, 2), (2, 2), (1, 2)]
    """
    assert path.ndim == 2
    assert window_size > 0

    n_points, n_dims = path.shape
    n_full_windows = n_points // window_size
    remainder = n_points % window_size

    windows = []

    # Create full windows
    for i in range(n_full_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window = path[start_idx:end_idx]
        windows.append(window)

    # Add the last window if there's a remainder
    if remainder > 0:
        start_idx = n_full_windows * window_size
        last_window = path[start_idx:]
        windows.append(last_window)

    return windows


def dyadic_windower(path: jax.Array, window_depth: int, window_size: int | None = None) -> list[tuple[jax.Array, jax.Array]]:
    r"""Create dyadic windows over a path.

    This function divides the path into a series of dyadic windows.
    At depth `d`, the path is split into `2**d` windows.
    The implementation is designed to be JAX-friendly, returning padded
    arrays of windows and their true lengths.

    For each depth, a tuple is returned containing:
    1. A padded array of all windows at that depth.
       Shape: `(2**d, max_window_len_d, n_dims)`.
    2. An array with the true length of each window.
       Shape: `(2**d,)`.

    Args:
        path: Input path array of shape `(n_points, n_dims)`.
        window_depth: The maximum depth of the dyadic windows.

    Returns:
        A list of tuples. For each depth `d` from 0 to `window_depth`,
        the list contains `(padded_windows_d, window_lengths_d)`.

    Raises:
        ValueError: If `window_depth` is so large that it may create windows
                    with fewer than two points.
    """
    assert path.ndim == 2
    assert window_depth >= 0

    seq_len, n_dims = path.shape

    if 2 ** (window_depth + 1) > seq_len:
        raise ValueError(f"window_depth {window_depth} is too large for path of length {seq_len}." f" This may result in windows with less than 2 points.")

    all_windows_info = []
    for d in range(window_depth + 1):
        num_windows = 2**d

        boundaries = jnp.floor(jnp.linspace(0, seq_len, num_windows + 1)).astype(jnp.int32)
        starts = boundaries[:-1]
        ends = boundaries[1:]

        window_lengths = ends - starts
        max_window_length = jnp.max(window_lengths)

        def map_fn(i):
            start = starts[i]
            length = window_lengths[i]

            # Slice a fixed-size window of max_window_length. `max_window_length` is a
            # concrete value, which is required for slice sizes in `dynamic_slice`.
            window = jax.lax.dynamic_slice(path, (start, 0), (max_window_length, n_dims))

            # Create a mask to zero out elements beyond `length`, effectively padding.
            mask = jnp.arange(max_window_length) < length
            return window * mask[:, jnp.newaxis]

        # Using jax.lax.map to create the padded windows for the current depth
        padded_windows = jax.lax.map(map_fn, jnp.arange(num_windows))

        all_windows_info.append((padded_windows, window_lengths))

    return all_windows_info


if __name__ == "__main__":
    path = jnp.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0],
        ]
    )
    depth = 2
    print(f"Original path (length {path.shape[0]}):")
    print(path)
    print(f"\nDyadic windows with depth {depth}:")

    windows_info = dyadic_windower(path, depth)

    for d, (padded_windows, lengths) in enumerate(windows_info):
        print(f"\n--- Depth {d} ---")
        print(f"Padded windows tensor shape: {padded_windows.shape}")
        print(f"Window lengths: {lengths}")
        for i, (window, length) in enumerate(zip(padded_windows, lengths)):
            print(f"  Window {i} (length {length}):")
            print(window[:length])
