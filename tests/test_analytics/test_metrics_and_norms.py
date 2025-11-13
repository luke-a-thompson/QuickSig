import pytest
from stochastax.analytics.metrics_and_norms import (
    hurst_to_holder_a,
    hurst_to_minimal_signature_depth,
)


def test_get_holder_alpha_valid_cases():
    """Test get_holder_alpha with valid inputs."""
    # Case: H > 0.5 should return 1 (Young integration suffices)
    assert hurst_to_holder_a(0.6) == 1
    assert hurst_to_holder_a(0.75, epsilon=0.01) == 1

    # Case: H < 0.5 should return H - epsilon
    assert hurst_to_holder_a(0.4, epsilon=0.01) == pytest.approx(0.39)
    assert hurst_to_holder_a(0.3, epsilon=0.05) == pytest.approx(0.25)

    # Case: H exactly 0.5
    assert hurst_to_holder_a(0.5) == pytest.approx(0.49)


def test_get_holder_alpha_error_cases():
    """Test get_holder_alpha error handling."""
    # H must be positive
    with pytest.raises(ValueError, match="H must be positive"):
        hurst_to_holder_a(0.0)

    with pytest.raises(ValueError, match="H must be positive"):
        hurst_to_holder_a(-0.1)

    # H - epsilon must be positive
    with pytest.raises(ValueError, match="Hölder exponent .* must be positive"):
        hurst_to_holder_a(0.005, epsilon=0.01)


def test_get_minimal_signature_depth_valid_cases():
    """Test get_minimal_signature_depth with valid inputs."""
    # H > 0.5: alpha = 1, so depth = floor(1/1) = 1
    assert hurst_to_minimal_signature_depth(0.6) == 1
    assert hurst_to_minimal_signature_depth(0.75) == 1

    # H = 0.4, epsilon=0.01: alpha = 0.39, depth = floor(1/0.39) = floor(2.564...) = 2
    assert hurst_to_minimal_signature_depth(0.4, epsilon=0.01) == 2

    # H = 0.3, epsilon=0.05: alpha = 0.25, depth = floor(1/0.25) = 4
    assert hurst_to_minimal_signature_depth(0.3, epsilon=0.05) == 4

    # H = 0.2, epsilon=0.01: alpha = 0.19, depth = floor(1/0.19) = floor(5.26...) = 5
    assert hurst_to_minimal_signature_depth(0.2, epsilon=0.01) == 5


def test_get_minimal_signature_depth_error_cases():
    """Test get_minimal_signature_depth error handling (inherits from get_holder_alpha)."""
    # Invalid H values should propagate errors from get_holder_alpha
    with pytest.raises(ValueError, match="H must be positive"):
        hurst_to_minimal_signature_depth(0.0)

    with pytest.raises(ValueError, match="H must be positive"):
        hurst_to_minimal_signature_depth(-0.5)

    with pytest.raises(ValueError, match="Hölder exponent .* must be positive"):
        hurst_to_minimal_signature_depth(0.005, epsilon=0.01)
