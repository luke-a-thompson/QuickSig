import pytest
import functools
import jax
import jax.numpy as jnp
from quicksig.rdes.augmentations import (
    augment_path,
    basepoint_augmentation,
    time_augmentation,
    lead_lag_augmentation,
    non_overlapping_windower,
    dyadic_windower,
)
from tests.test_helpers import generate_scalar_path

_test_key = jax.random.PRNGKey(42)


@pytest.mark.parametrize(
    "input_path, expected_shape, expected_first_row",
    [
        pytest.param(
            jnp.array([[1, 2], [3, 4]]),
            (3, 2),
            jnp.array([0.0, 0.0]),
            id="2x2_path",
        ),
        pytest.param(
            jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            (4, 3),
            jnp.array([0.0, 0.0, 0.0]),
            id="3x3_path",
        ),
        pytest.param(
            jnp.array([[1.5, 2.5]]),
            (2, 2),
            jnp.array([0.0, 0.0]),
            id="1x2_path",
        ),
    ],
)
def test_basepoint_augmentation_valid_inputs(input_path, expected_shape, expected_first_row):
    """Test basepoint_augmentation with valid inputs."""
    result = basepoint_augmentation(input_path)

    assert result.shape == expected_shape
    assert jnp.array_equal(result[0], expected_first_row)
    assert jnp.array_equal(result[1:], input_path)


@pytest.mark.parametrize(
    "invalid_path",
    [
        pytest.param(jnp.array([1, 2, 3]), id="1D_array"),
        pytest.param(jnp.array([[[1, 2], [3, 4]]]), id="3D_array"),
        pytest.param(jnp.array([]), id="empty_array"),
    ],
)
def test_basepoint_augmentation_invalid_inputs(invalid_path):
    """Test basepoint_augmentation with invalid inputs."""
    with pytest.raises(AssertionError):
        basepoint_augmentation(invalid_path)


@pytest.mark.parametrize(
    "input_path, expected_shape, expected_time_values",
    [
        pytest.param(
            jnp.array([[1, 2], [3, 4]]),
            (2, 3),
            jnp.array([0.0, 1.0]),
            id="2x2_path",
        ),
        pytest.param(
            jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            (3, 4),
            jnp.array([0.0, 0.5, 1.0]),
            id="3x3_path",
        ),
        pytest.param(
            jnp.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5], [7.5, 8.5]]),
            (4, 3),
            jnp.array([0.0, 0.33333333, 0.66666667, 1.0]),
            id="4x2_path",
        ),
    ],
)
def test_time_augmentation_valid_inputs(input_path, expected_shape, expected_time_values):
    """Test time_augmentation with valid inputs."""
    result = time_augmentation(input_path)

    assert result.shape == expected_shape
    assert jnp.allclose(result[:, 0], expected_time_values)
    assert jnp.array_equal(result[:, 1:], input_path)


@pytest.mark.parametrize(
    "invalid_path",
    [
        pytest.param(jnp.array([1, 2, 3]), id="1D_array"),
        pytest.param(jnp.array([[[1, 2], [3, 4]]]), id="3D_array"),
        pytest.param(jnp.array([]), id="empty_array"),
    ],
)
def test_time_augmentation_invalid_inputs(invalid_path):
    """Test time_augmentation with invalid inputs."""
    with pytest.raises(AssertionError):
        time_augmentation(invalid_path)


@pytest.mark.parametrize(
    "leading_path, lagging_path, expected_output",
    [
        pytest.param(
            jnp.array([[1.0], [2.0], [3.0]]),
            jnp.array([[0.1], [0.2], [0.3]]),
            jnp.array([[1.0, 0.1], [2.0, 0.1], [3.0, 0.2]]),
            id="1D_paths",
        ),
        pytest.param(
            jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            jnp.array([[0.1, 0.2], [0.3, 0.4]]),
            jnp.array([[1.0, 2.0, 0.1, 0.2], [3.0, 4.0, 0.1, 0.2]]),
            id="2D_paths",
        ),
        pytest.param(
            jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            jnp.array([[0.1], [0.3], [0.5]]),
            jnp.array([[1.0, 2.0, 0.1], [3.0, 4.0, 0.1], [5.0, 6.0, 0.3]]),
            id="2D_leading_1D_lagging",
        ),
    ],
)
def test_lead_lag_augmentation_valid_inputs(leading_path, lagging_path, expected_output):
    """Test lead_lag_augmentation with valid inputs."""
    result = lead_lag_augmentation(leading_path, lagging_path)
    assert result.shape == (
        leading_path.shape[0],
        leading_path.shape[1] + lagging_path.shape[1],
    )
    assert jnp.allclose(result, expected_output)


@pytest.mark.parametrize(
    "leading_path, lagging_path",
    [
        pytest.param(
            jnp.array([[1.0], [2.0], [3.0]]),
            jnp.array([[0.1], [0.2]]),
            id="leading_longer",
        ),
        pytest.param(
            jnp.array([[1.0], [2.0]]),
            jnp.array([[0.1], [0.2], [0.3]]),
            id="lagging_longer",
        ),
    ],
)
def test_lead_lag_augmentation_mismatched_lengths(leading_path, lagging_path):
    """Test lead_lag_augmentation raises ValueError for mismatched lengths."""
    with pytest.raises(ValueError):
        lead_lag_augmentation(leading_path, lagging_path)


@pytest.mark.parametrize(
    "input_path, window_size, expected_num_windows, expected_shapes",
    [
        pytest.param(
            jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
            2,
            2,
            [(2, 2), (2, 2)],
            id="4x2_path_window_2",
        ),
        pytest.param(
            jnp.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]),
            3,
            2,
            [(3, 2), (3, 2)],
            id="6x2_path_window_3",
        ),
        pytest.param(
            jnp.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
            2,
            3,
            [(2, 2), (2, 2), (1, 2)],
            id="5x2_path_window_2_remainder",
        ),
        pytest.param(
            jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            4,
            1,
            [(4, 3)],
            id="4x3_path_window_4",
        ),
    ],
)
def test_sliding_windower_valid_inputs(input_path, window_size, expected_num_windows, expected_shapes):
    """Test sliding_windower with valid inputs."""
    result = non_overlapping_windower(input_path, window_size)

    assert len(result) == expected_num_windows
    for i, (window, expected_shape) in enumerate(zip(result, expected_shapes)):
        assert window.shape == expected_shape


@pytest.mark.parametrize(
    "input_path, window_size, expected_windows",
    [
        pytest.param(
            jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
            2,
            [
                jnp.array([[1, 2], [3, 4]]),
                jnp.array([[5, 6], [7, 8]]),
            ],
            id="4x2_path_window_2",
        ),
        pytest.param(
            jnp.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
            2,
            [
                jnp.array([[1, 2], [3, 4]]),
                jnp.array([[5, 6], [7, 8]]),
                jnp.array([[9, 10]]),
            ],
            id="5x2_path_window_2_remainder",
        ),
    ],
)
def test_sliding_windower_content(input_path, window_size, expected_windows):
    """Test sliding_windower content correctness."""
    result = non_overlapping_windower(input_path, window_size)

    assert len(result) == len(expected_windows)
    for actual, expected in zip(result, expected_windows):
        assert jnp.array_equal(actual, expected)


@pytest.mark.parametrize(
    "invalid_path, window_size",
    [
        pytest.param(jnp.array([1, 2, 3]), 2, id="1D_array"),
        pytest.param(jnp.array([[[1, 2], [3, 4]]]), 2, id="3D_array"),
        pytest.param(jnp.array([]), 2, id="empty_array"),
    ],
)
def test_sliding_windower_invalid_inputs(invalid_path, window_size):
    """Test sliding_windower with invalid inputs."""
    with pytest.raises(AssertionError):
        non_overlapping_windower(invalid_path, window_size)


@pytest.mark.parametrize(
    "input_path, window_depth, expected_num_depths, expected_window_counts",
    [
        pytest.param(
            jnp.array(
                [
                    [1, 2],
                    [3, 4],
                    [5, 6],
                    [7, 8],
                    [9, 10],
                    [11, 12],
                    [13, 14],
                    [15, 16],
                ]
            ),
            2,
            3,
            [1, 2, 4],
            id="8_points_depth_2",
        ),
        pytest.param(
            jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
            1,
            2,
            [1, 2],
            id="4_points_depth_1",
        ),
        pytest.param(
            jnp.array(
                [
                    [1, 2],
                    [3, 4],
                    [5, 6],
                    [7, 8],
                    [9, 10],
                    [11, 12],
                    [13, 14],
                    [15, 16],
                ]
            ),
            0,
            1,
            [1],
            id="8_points_depth_0",
        ),
    ],
)
def test_dyadic_windower_valid_inputs(input_path, window_depth, expected_num_depths, expected_window_counts):
    """Test dyadic_windower with valid inputs."""
    result = dyadic_windower(input_path, window_depth)

    assert len(result) == expected_num_depths
    for i, (windows_info, expected_count) in enumerate(zip(result, expected_window_counts)):
        padded_windows, window_lengths = windows_info
        assert padded_windows.shape[0] == expected_count
        assert window_lengths.shape[0] == expected_count


@pytest.mark.parametrize(
    "input_path, window_depth",
    [
        pytest.param(jnp.array([[1, 2], [3, 4]]), 2, id="depth_too_large_for_small_path"),
        pytest.param(
            jnp.array([[1, 2], [3, 4], [5, 6]]),
            3,
            id="depth_too_large_for_medium_path",
        ),
    ],
)
def test_dyadic_windower_invalid_depth(input_path, window_depth):
    """Test dyadic_windower with invalid depth."""
    with pytest.raises(ValueError):
        dyadic_windower(input_path, window_depth)


@pytest.mark.parametrize(
    "invalid_path, window_depth",
    [
        pytest.param(jnp.array([1, 2, 3]), 1, id="1D_array"),
        pytest.param(jnp.array([[[1, 2], [3, 4]]]), 1, id="3D_array"),
        pytest.param(jnp.array([]), 1, id="empty_array"),
    ],
)
def test_dyadic_windower_invalid_inputs(invalid_path, window_depth):
    """Test dyadic_windower with invalid inputs."""
    with pytest.raises(AssertionError):
        dyadic_windower(invalid_path, window_depth)


@pytest.mark.parametrize(
    "input_path, window_depth",
    [
        pytest.param(jnp.array([[1, 2], [3, 4]]), -1, id="negative_depth"),
    ],
)
def test_dyadic_windower_negative_depth(input_path, window_depth):
    """Test dyadic_windower with negative depth."""
    with pytest.raises(AssertionError):
        dyadic_windower(input_path, window_depth)


@pytest.mark.parametrize(
    "input_path, augmentations, expected_shape",
    [
        pytest.param(
            jnp.array([[1, 2], [3, 4]]),
            [basepoint_augmentation],
            (3, 2),
            id="basepoint_augmentation",
        ),
        pytest.param(
            jnp.array([[1, 2], [3, 4]]),
            [time_augmentation],
            (2, 3),
            id="time_augmentation",
        ),
        pytest.param(
            jnp.array([[1, 2], [3, 4]]),
            [basepoint_augmentation, time_augmentation],
            (3, 3),
            id="basepoint_and_time_augmentation",
        ),
        pytest.param(jnp.array([[1, 2], [3, 4]]), [], (2, 2), id="no_augmentation"),
    ],
)
def test_augment_path_valid_inputs(input_path, augmentations, expected_shape):
    """Test augment_path with valid inputs."""
    result = augment_path(input_path, augmentations)
    assert result.shape == expected_shape


@pytest.mark.parametrize(
    "input_path, augmentations, expected_num_windows, expected_shapes",
    [
        # Just sliding windower
        pytest.param(
            jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
            [functools.partial(non_overlapping_windower, window_size=2)],
            2,
            [(2, 2), (2, 2)],
            id="sliding_only",
        ),
        # Sliding windower with remainder
        pytest.param(
            jnp.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
            [functools.partial(non_overlapping_windower, window_size=2)],
            3,
            [(2, 2), (2, 2), (1, 2)],
            id="sliding_only_remainder",
        ),
        # Sliding windower with basepoint augmentation
        pytest.param(
            jnp.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),  # 5 points
            [
                basepoint_augmentation,
                functools.partial(non_overlapping_windower, window_size=2),
            ],
            3,  # 6 points with basepoint, so 3 windows of size 2
            [(2, 2), (2, 2), (2, 2)],
            id="sliding_with_basepoint",
        ),
        # Sliding windower with time augmentation
        pytest.param(
            jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]]),  # 4x2
            [time_augmentation, functools.partial(non_overlapping_windower, window_size=2)],
            2,  # 4 points, 2 windows of size 2
            [(2, 3), (2, 3)],  # shape should have time dim
            id="sliding_with_time",
        ),
        # Sliding windower with basepoint and time augmentation
        pytest.param(
            jnp.array([[1, 2], [3, 4], [5, 6]]),  # 3 points
            [
                basepoint_augmentation,
                time_augmentation,
                functools.partial(non_overlapping_windower, window_size=2),
            ],
            2,  # 4 points with basepoint, 2 windows
            [(2, 3), (2, 3)],  # shape (4,3) -> windows of (2,3)
            id="sliding_with_basepoint_and_time",
        ),
    ],
)
def test_augment_path_with_sliding_windower(input_path, augmentations, expected_num_windows, expected_shapes):
    """Test augment_path with sliding_windower and other augmentations."""
    result = augment_path(input_path, augmentations)

    assert isinstance(result, list)
    assert len(result) == expected_num_windows
    for window, shape in zip(result, expected_shapes):
        assert window.shape == shape

    # Check content for basepoint augmentation case
    if basepoint_augmentation in augmentations:
        assert jnp.all(result[0][0] == 0)


@pytest.mark.parametrize(
    "input_path, augmentations, expected_num_depths, expected_window_counts",
    [
        # Just dyadic windower
        pytest.param(
            jnp.arange(16).reshape(8, 2),  # 8 points
            [functools.partial(dyadic_windower, window_depth=2)],
            3,  # depths 0, 1, 2
            [1, 2, 4],
            id="dyadic_only",
        ),
        # Dyadic windower with basepoint augmentation
        pytest.param(
            jnp.arange(14).reshape(7, 2),  # 7 points
            [
                basepoint_augmentation,
                functools.partial(dyadic_windower, window_depth=2),
            ],
            3,  # depths 0, 1, 2. Path becomes 8 points long.
            [1, 2, 4],
            id="dyadic_with_basepoint",
        ),
        # Dyadic windower with time augmentation
        pytest.param(
            jnp.arange(16).reshape(8, 2),  # 8x2
            [time_augmentation, functools.partial(dyadic_windower, window_depth=1)],
            2,  # depths 0, 1. Path is 8x3
            [1, 2],
            id="dyadic_with_time",
        ),
    ],
)
def test_augment_path_with_dyadic_windower(
    input_path,
    augmentations,
    expected_num_depths,
    expected_window_counts,
):
    """Test augment_path with dyadic_windower and other augmentations."""
    result = augment_path(input_path, augmentations)

    assert isinstance(result, list)
    assert len(result) == expected_num_depths
    for windows_info, expected_count in zip(result, expected_window_counts):
        padded_windows, window_lengths = windows_info
        assert padded_windows.shape[0] == expected_count
        assert window_lengths.shape[0] == expected_count
        # If time augmentation was used, check shape
        if time_augmentation in augmentations:
            assert padded_windows.shape[-1] == input_path.shape[1] + 1
        elif basepoint_augmentation not in augmentations:
            assert padded_windows.shape[-1] == input_path.shape[1]


def test_augment_path_sliding_and_dyadic_windower():
    """Test using both sliding and dyadic windowers with augment_path."""
    input_path = jnp.arange(64).reshape(32, 2)
    window_size = 16
    window_depth = 2

    augmentations = [
        functools.partial(non_overlapping_windower, window_size=window_size),
        functools.partial(dyadic_windower, window_depth=window_depth),
    ]

    result = augment_path(input_path, augmentations)

    # We should get 2 sliding windows of size 16.
    assert isinstance(result, list)
    assert len(result) == 2  # 32 / 16

    # Each element of result should be the output of dyadic_windower
    for dyadic_result in result:
        assert isinstance(dyadic_result, list)
        assert len(dyadic_result) == window_depth + 1
        expected_window_counts = [1, 2, 4]  # for depth 2
        for windows_info, expected_count in zip(dyadic_result, expected_window_counts):
            padded_windows, window_lengths = windows_info
            assert padded_windows.shape[0] == expected_count
            assert window_lengths.shape[0] == expected_count


def test_augment_path_sliding_and_dyadic_with_path_augmentations():
    """Test sliding and dyadic windowers with path augmentations."""
    input_path = jnp.arange(60).reshape(30, 2)  # 30 points
    window_size = 16
    window_depth = 2

    augmentations = [
        basepoint_augmentation,  # Path becomes 31 points
        time_augmentation,  # Path becomes 31x3
        functools.partial(non_overlapping_windower, window_size=window_size),
        functools.partial(dyadic_windower, window_depth=window_depth),
    ]

    result = augment_path(input_path, augmentations)

    # 31 points / 16 = 1 full window, 1 remainder window of 15
    assert isinstance(result, list)
    assert len(result) == 2

    # Check first dyadic result (from window of 16)
    dyadic_result_1 = result[0]
    assert isinstance(dyadic_result_1, list)
    assert len(dyadic_result_1) == window_depth + 1
    expected_window_counts_1 = [1, 2, 4]
    for windows_info, expected_count in zip(dyadic_result_1, expected_window_counts_1):
        padded_windows, _ = windows_info
        assert padded_windows.shape[0] == expected_count
        assert padded_windows.shape[-1] == 3  # time dimension

    # Check second dyadic result (from window of 15)
    dyadic_result_2 = result[1]
    assert isinstance(dyadic_result_2, list)
    assert len(dyadic_result_2) == window_depth + 1
    expected_window_counts_2 = [1, 2, 4]
    for windows_info, expected_count in zip(dyadic_result_2, expected_window_counts_2):
        padded_windows, _ = windows_info
        assert padded_windows.shape[0] == expected_count
        assert padded_windows.shape[-1] == 3


def test_multiple_augmentations_combined():
    """Test combining multiple augmentations."""
    input_path = jnp.array([[1, 2], [3, 4], [5, 6]])
    augmentations = [basepoint_augmentation, time_augmentation]

    result = augment_path(input_path, augmentations)

    # Should have basepoint (n_points + 1) and time dimension (n_dims + 1)
    expected_shape = (input_path.shape[0] + 1, input_path.shape[1] + 1)
    assert result.shape == expected_shape

    # Check basepoint is zero
    assert jnp.array_equal(result[0, 1:], jnp.zeros(input_path.shape[1]))

    # Check time dimension
    expected_times = jnp.linspace(0, 1, input_path.shape[0] + 1)
    assert jnp.allclose(result[:, 0], expected_times)


def test_sliding_windower_with_augmented_path():
    """Test sliding_windower with an augmented path."""
    input_path = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    augmented_path = basepoint_augmentation(input_path)

    windows = non_overlapping_windower(augmented_path, 2)

    assert len(windows) == 3  # 6 points with window_size=2
    assert windows[0].shape == (2, 2)
    assert windows[1].shape == (2, 2)
    assert windows[2].shape == (2, 2)

    # Check first window contains basepoint
    assert jnp.array_equal(windows[0][0], jnp.array([0.0, 0.0]))


@pytest.mark.parametrize("n_features", [1, 2, 3])
@pytest.mark.parametrize("num_steps", [10, 20])
def test_basepoint_augmentation_with_gbm_path(n_features: int, num_steps: int):
    """Test basepoint_augmentation with GBM-generated paths."""
    global _test_key
    _test_key, subkey = jax.random.split(_test_key)

    path = generate_scalar_path(subkey, num_timesteps=num_steps, n_features=n_features)
    result = basepoint_augmentation(path)

    assert result.shape == (num_steps + 1, n_features)
    assert jnp.array_equal(result[0], jnp.zeros(n_features))
    assert jnp.array_equal(result[1:], path)


@pytest.mark.parametrize("n_features", [1, 2, 3])
@pytest.mark.parametrize("num_steps", [10, 20])
def test_time_augmentation_with_gbm_path(n_features: int, num_steps: int):
    """Test time_augmentation with GBM-generated paths."""
    global _test_key
    _test_key, subkey = jax.random.split(_test_key)

    path = generate_scalar_path(subkey, num_timesteps=num_steps, n_features=n_features)
    result = time_augmentation(path)

    assert result.shape == (num_steps, n_features + 1)
    expected_times = jnp.linspace(0, 1, num_steps)
    assert jnp.allclose(result[:, 0], expected_times)
    assert jnp.array_equal(result[:, 1:], path)


@pytest.mark.parametrize("n_features", [1, 2, 3])
@pytest.mark.parametrize("window_size", [2, 3, 4])
def test_sliding_windower_with_gbm_path(n_features: int, window_size: int):
    """Test sliding_windower with GBM-generated paths."""
    global _test_key
    _test_key, subkey = jax.random.split(_test_key)

    path = generate_scalar_path(subkey, num_timesteps=20, n_features=n_features)
    windows = non_overlapping_windower(path, window_size)

    n_full_windows = 20 // window_size
    remainder = 20 % window_size

    expected_num_windows = n_full_windows + (1 if remainder > 0 else 0)
    assert len(windows) == expected_num_windows

    # Check full windows
    for i in range(n_full_windows):
        assert windows[i].shape == (window_size, n_features)

    # Check last window if there's a remainder
    if remainder > 0:
        assert windows[-1].shape == (remainder, n_features)


@pytest.mark.parametrize("n_features", [1, 2, 3])
@pytest.mark.parametrize("window_depth", [1, 2])
def test_dyadic_windower_with_gbm_path(n_features: int, window_depth: int):
    """Test dyadic_windower with GBM-generated paths."""
    global _test_key
    _test_key, subkey = jax.random.split(_test_key)

    path = generate_scalar_path(subkey, num_timesteps=16, n_features=n_features)
    result = dyadic_windower(path, window_depth)

    assert len(result) == window_depth + 1

    for d, (padded_windows, window_lengths) in enumerate(result):
        expected_num_windows = 2**d
        assert padded_windows.shape[0] == expected_num_windows
        assert window_lengths.shape[0] == expected_num_windows


@pytest.mark.parametrize("n_features", [1, 2, 3])
def test_augment_path_with_gbm_path(n_features: int):
    """Test augment_path with GBM-generated paths."""
    global _test_key
    _test_key, subkey = jax.random.split(_test_key)

    path = generate_scalar_path(subkey, num_timesteps=10, n_features=n_features)
    augmentations = [basepoint_augmentation, time_augmentation]

    result = augment_path(path, augmentations)

    # Should have basepoint (n_points + 1) and time dimension (n_dims + 1)
    expected_shape = (path.shape[0] + 1, path.shape[1] + 1)
    assert result.shape == expected_shape

    # Check basepoint is zero
    assert jnp.array_equal(result[0, 1:], jnp.zeros(n_features))

    # Check time dimension
    expected_times = jnp.linspace(0, 1, path.shape[0] + 1)
    assert jnp.allclose(result[:, 0], expected_times)


def test_augment_path_unknown_augmentation():
    """Test that augment_path raises an error for an unknown augmentation."""
    input_path = jnp.array([[1, 2]])

    def unknown_aug(p):
        return p

    with pytest.raises(ValueError, match="Unknown augmentation"):
        augment_path(input_path, [unknown_aug])
