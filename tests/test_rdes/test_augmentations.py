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
from quicksig.rdes.rde_types import pathify, Path
from tests.test_helpers import generate_scalar_path

_test_key = jax.random.PRNGKey(42)


@pytest.mark.parametrize(
    "input_path_array, expected_shape, expected_first_row",
    [
        pytest.param(jnp.array([[1, 2], [3, 4]]), (3, 2), jnp.array([0.0, 0.0]), id="2x2_path"),
        pytest.param(jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), (4, 3), jnp.array([0.0, 0.0, 0.0]), id="3x3_path"),
        pytest.param(jnp.array([[1.5, 2.5]]), (2, 2), jnp.array([0.0, 0.0]), id="1x2_path"),
    ],
)
def test_basepoint_augmentation_valid_inputs(input_path_array, expected_shape, expected_first_row):
    """Test basepoint_augmentation with valid inputs."""
    input_path = pathify(input_path_array)
    result = basepoint_augmentation(input_path)

    assert isinstance(result, Path)
    assert result.path.shape == expected_shape
    assert jnp.array_equal(result.path[0], expected_first_row)
    assert jnp.array_equal(result.path[1:], input_path.path)


@pytest.mark.parametrize(
    "input_path_array, expected_shape, expected_time_values",
    [
        pytest.param(jnp.array([[1, 2], [3, 4]]), (2, 3), jnp.array([0.0, 1.0]), id="2x2_path"),
        pytest.param(jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), (3, 4), jnp.array([0.0, 0.5, 1.0]), id="3x3_path"),
        pytest.param(jnp.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5], [7.5, 8.5]]), (4, 3), jnp.array([0.0, 1.0/3.0, 2.0/3.0, 1.0]), id="4x2_path"),
    ],
)
def test_time_augmentation_valid_inputs(input_path_array, expected_shape, expected_time_values):
    """Test time_augmentation with valid inputs."""
    input_path = pathify(input_path_array)
    result = time_augmentation(input_path)

    assert isinstance(result, Path)
    assert result.path.shape == expected_shape
    assert jnp.allclose(result.path[:, 0], expected_time_values)
    assert jnp.array_equal(result.path[:, 1:], input_path.path)
    assert result.ambient_dimension == input_path.ambient_dimension + 1
    assert result.interval == input_path.interval


@pytest.mark.parametrize(
    "leading_path_array, lagging_path_array, expected_output",
    [
        pytest.param(jnp.array([[1.0], [2.0], [3.0]]), jnp.array([[0.1], [0.2], [0.3]]), jnp.array([[1.0, 0.1], [2.0, 0.1], [3.0, 0.2]]), id="1D_paths"),
        pytest.param(jnp.array([[1.0, 2.0], [3.0, 4.0]]), jnp.array([[0.1, 0.2], [0.3, 0.4]]), jnp.array([[1.0, 2.0, 0.1, 0.2], [3.0, 4.0, 0.1, 0.2]]), id="2D_paths"),
        pytest.param(jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), jnp.array([[0.1], [0.3], [0.5]]), jnp.array([[1.0, 2.0, 0.1], [3.0, 4.0, 0.1], [5.0, 6.0, 0.3]]), id="2D_leading_1D_lagging"),
    ],
)
def test_lead_lag_augmentation_valid_inputs(leading_path_array, lagging_path_array, expected_output):
    """Test lead_lag_augmentation with valid inputs."""
    leading_path = pathify(leading_path_array)
    lagging_path = pathify(lagging_path_array)
    result = lead_lag_augmentation(leading_path, lagging_path)

    assert isinstance(result, Path)
    assert result.path.shape == (leading_path.path.shape[0], leading_path.path.shape[1] + lagging_path.path.shape[1])
    assert jnp.allclose(result.path, expected_output)
    assert result.ambient_dimension == leading_path.ambient_dimension + lagging_path.ambient_dimension
    assert result.interval == leading_path.interval


@pytest.mark.parametrize(
    "leading_path_array, lagging_path_array",
    [
        pytest.param(jnp.array([[1.0], [2.0], [3.0]]), jnp.array([[0.1], [0.2]]), id="leading_longer"),
        pytest.param(jnp.array([[1.0], [2.0]]), jnp.array([[0.1], [0.2], [0.3]]), id="lagging_longer"),
    ],
)
def test_lead_lag_augmentation_mismatched_lengths(leading_path_array, lagging_path_array):
    """Test lead_lag_augmentation raises ValueError for mismatched lengths."""
    leading_path = pathify(leading_path_array)
    lagging_path = pathify(lagging_path_array)
    with pytest.raises(ValueError):
        lead_lag_augmentation(leading_path, lagging_path)


@pytest.mark.parametrize(
    "input_path_array, window_size, expected_num_windows, expected_shapes",
    [
        pytest.param(jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]]), 2, 2, [(2, 2), (2, 2)], id="4x2_path_window_2"),
        pytest.param(jnp.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]), 3, 2, [(3, 2), (3, 2)], id="6x2_path_window_3"),
        pytest.param(jnp.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), 2, 3, [(2, 2), (2, 2), (1, 2)], id="5x2_path_window_2_remainder"),
        pytest.param(jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]), 4, 1, [(4, 3)], id="4x3_path_window_4"),
    ],
)
def test_non_overlapping_windower_valid_inputs(input_path_array, window_size, expected_num_windows, expected_shapes):
    """Test non_overlapping_windower with valid inputs."""
    input_path = pathify(input_path_array)
    result = non_overlapping_windower(input_path, window_size)

    assert len(result) == expected_num_windows
    for i, (window, expected_shape) in enumerate(zip(result, expected_shapes)):
        assert isinstance(window, Path)
        assert window.path.shape == expected_shape


@pytest.mark.parametrize(
    "input_path_array, window_size, expected_windows_arrays",
    [
        pytest.param(jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]]), 2, [jnp.array([[1, 2], [3, 4]]), jnp.array([[5, 6], [7, 8]])], id="4x2_path_window_2"),
        pytest.param(jnp.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), 2, [jnp.array([[1, 2], [3, 4]]), jnp.array([[5, 6], [7, 8]]), jnp.array([[9, 10]])], id="5x2_path_window_2_remainder"),
    ],
)
def test_non_overlapping_windower_content(input_path_array, window_size, expected_windows_arrays):
    """Test non_overlapping_windower content correctness."""
    input_path = pathify(input_path_array)
    result = non_overlapping_windower(input_path, window_size)

    assert len(result) == len(expected_windows_arrays)
    for actual, expected in zip(result, expected_windows_arrays):
        assert jnp.array_equal(actual.path, expected)


@pytest.mark.parametrize(
    "input_path_array, window_depth, expected_num_depths, expected_window_counts",
    [
        pytest.param(jnp.arange(16).reshape(8, 2), 2, 3, [1, 2, 4], id="8_points_depth_2"),
        pytest.param(jnp.arange(8).reshape(4, 2), 1, 2, [1, 2], id="4_points_depth_1"),
        pytest.param(jnp.arange(16).reshape(8, 2), 0, 1, [1], id="8_points_depth_0"),
    ],
)
def test_dyadic_windower_valid_inputs(input_path_array, window_depth, expected_num_depths, expected_window_counts):
    """Test dyadic_windower with valid inputs."""
    input_path = pathify(input_path_array)
    result = dyadic_windower(input_path, window_depth)

    assert len(result) == expected_num_depths
    for i, (windows_at_depth, expected_count) in enumerate(zip(result, expected_window_counts)):
        assert len(windows_at_depth) == expected_count
        for window in windows_at_depth:
            assert isinstance(window, Path)


@pytest.mark.parametrize(
    "input_path_array, window_depth",
    [
        pytest.param(jnp.array([[1, 2], [3, 4]]), 2, id="depth_too_large_for_small_path"),
        pytest.param(jnp.array([[1, 2], [3, 4], [5, 6]]), 3, id="depth_too_large_for_medium_path"),
    ],
)
def test_dyadic_windower_invalid_depth(input_path_array, window_depth):
    """Test dyadic_windower with invalid depth."""
    input_path = pathify(input_path_array)
    with pytest.raises(ValueError):
        dyadic_windower(input_path, window_depth)


@pytest.mark.parametrize(
    "input_path_array, augmentations, expected_shape",
    [
        pytest.param(jnp.array([[1, 2], [3, 4]]), [basepoint_augmentation], (3, 2), id="basepoint_augmentation"),
        pytest.param(jnp.array([[1, 2], [3, 4]]), [time_augmentation], (2, 3), id="time_augmentation"),
        pytest.param(jnp.array([[1, 2], [3, 4]]), [basepoint_augmentation, time_augmentation], (3, 3), id="basepoint_and_time_augmentation"),
        pytest.param(jnp.array([[1, 2], [3, 4]]), [], (2, 2), id="no_augmentation"),
    ],
)
def test_augment_path_valid_inputs(input_path_array, augmentations, expected_shape):
    """Test augment_path with valid inputs."""
    input_path = pathify(input_path_array)
    result = augment_path(input_path, augmentations)
    assert result.path.shape == expected_shape


def test_augment_path_with_non_zero_start_interval():
    """Test that augment_path correctly handles non-default start intervals."""
    # Create a path with a non-zero start interval, e.g., (10, 20)
    input_path_array = jnp.array([[1, 2], [3, 4], [5, 6]])
    input_path = Path(path=input_path_array, interval=(10, 13))

    # Perform a basepoint augmentation
    result = augment_path(input_path, [basepoint_augmentation])

    # The resulting path should have its interval updated to reflect the new length
    # but starting from the original start time.
    assert result.interval == (0, 4), f"Expected interval (0, 4), but got {result.interval}"

    # Now, test with a windowing augmentation
    windows = augment_path(input_path, [functools.partial(non_overlapping_windower, window_size=2)])

    # The sub-paths should have intervals relative to the original path's interval
    assert len(windows) == 2
    assert windows[0].interval == (10, 12), f"Expected interval (10, 12), but got {windows[0].interval}"
    assert windows[1].interval == (12, 13), f"Expected interval (12, 13), but got {windows[1].interval}"


def test_augment_path_unknown_augmentation():
    """Test that augment_path raises an error for an unknown augmentation."""
    input_path = pathify(jnp.array([[1, 2]]))

    def unknown_aug(p):
        return p

    with pytest.raises(ValueError, match="Unknown augmentation"):
        augment_path(input_path, [unknown_aug])
