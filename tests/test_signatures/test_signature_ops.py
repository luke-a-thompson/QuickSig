import jax
import jax.numpy as jnp
import pytest
from quicksig.signatures.compute_path_signature import compute_path_signature
from quicksig.signatures.signature_types import Signature, LogSignature, _chen_identity
from tests.test_helpers import scalar_path_fixture


@pytest.mark.parametrize("scalar_path_fixture", [(1, 20), (2, 30)], indirect=True)
@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_chen_identity(scalar_path_fixture: jax.Array, depth: int):
    path = scalar_path_fixture
    midpoint_idx = len(path) // 2

    # Split the path into two overlapping segments
    path_1 = path[: midpoint_idx + 1]
    path_2 = path[midpoint_idx:]

    # Compute signatures for each sub-path.
    sig_1_computed = compute_path_signature(path_1, depth=depth, mode="full")
    sig_2_computed = compute_path_signature(path_2, depth=depth, mode="full")

    # Re-create the signatures with intervals that are contiguous and reflect their position in the original path.
    sig_1 = Signature(
        signature=sig_1_computed.signature,
        interval=(0, midpoint_idx),
        ambient_dimension=sig_1_computed.ambient_dimension,
        depth=sig_1_computed.depth,
        basis_name=sig_1_computed.basis_name,
    )
    sig_2 = Signature(
        signature=sig_2_computed.signature,
        interval=(midpoint_idx, len(path) - 1),
        ambient_dimension=sig_2_computed.ambient_dimension,
        depth=sig_2_computed.depth,
        basis_name=sig_2_computed.basis_name,
    )

    # Combine the signatures using Chen's identity
    combined_sig = sig_1 @ sig_2

    # Compute the signature over the whole path for comparison
    whole_sig = compute_path_signature(path, depth=depth, mode="full")

    # The flattened signature values should be the same
    assert jnp.allclose(combined_sig.flatten(), whole_sig.flatten(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("scalar_path_fixture", [(1, 30), (2, 30)], indirect=True)
@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_chen_identity_three_signatures(scalar_path_fixture: jax.Array, depth: int):
    path = scalar_path_fixture
    third_point_idx = len(path) // 3
    two_thirds_point_idx = 2 * len(path) // 3

    # Split the path into three overlapping segments
    path_1 = path[: third_point_idx + 1]
    path_2 = path[third_point_idx : two_thirds_point_idx + 1]
    path_3 = path[two_thirds_point_idx:]

    # Compute signatures for each sub-path
    sig_1_computed = compute_path_signature(path_1, depth=depth, mode="full")
    sig_2_computed = compute_path_signature(path_2, depth=depth, mode="full")
    sig_3_computed = compute_path_signature(path_3, depth=depth, mode="full")

    # Re-create the signatures with contiguous intervals
    sig_1 = Signature(
        signature=sig_1_computed.signature,
        interval=(0, third_point_idx),
        ambient_dimension=sig_1_computed.ambient_dimension,
        depth=sig_1_computed.depth,
        basis_name=sig_1_computed.basis_name,
    )
    sig_2 = Signature(
        signature=sig_2_computed.signature,
        interval=(third_point_idx, two_thirds_point_idx),
        ambient_dimension=sig_2_computed.ambient_dimension,
        depth=sig_2_computed.depth,
        basis_name=sig_2_computed.basis_name,
    )
    sig_3 = Signature(
        signature=sig_3_computed.signature,
        interval=(two_thirds_point_idx, len(path) - 1),
        ambient_dimension=sig_3_computed.ambient_dimension,
        depth=sig_3_computed.depth,
        basis_name=sig_3_computed.basis_name,
    )

    # The @ operator is the Chen identity for signatures.
    combined_sig = sig_1 @ sig_2 @ sig_3

    # Compute the signature over the whole path for comparison
    whole_sig = compute_path_signature(path, depth=depth, mode="full")

    # The flattened signature values should be the same
    assert jnp.allclose(combined_sig.flatten(), whole_sig.flatten(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("scalar_path_fixture", [(1, 20)], indirect=True)
@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_chen_identity_non_consecutive_intervals(scalar_path_fixture: jax.Array, depth: int):
    path = scalar_path_fixture
    midpoint_idx = len(path) // 2

    # Split the path into two overlapping segments
    path_1 = path[: midpoint_idx + 1]
    path_2 = path[midpoint_idx:]

    # Compute signatures for each sub-path.
    sig_1_computed = compute_path_signature(path_1, depth=depth, mode="full")
    sig_2_computed = compute_path_signature(path_2, depth=depth, mode="full")

    # Re-create the signatures with intervals that are contiguous and reflect their position in the original path.
    sig_1 = Signature(
        signature=sig_1_computed.signature,
        interval=(0, midpoint_idx),
        ambient_dimension=sig_1_computed.ambient_dimension,
        depth=sig_1_computed.depth,
        basis_name=sig_1_computed.basis_name,
    )
    # create a gap between intervals
    sig_2_non_consecutive = Signature(
        signature=sig_2_computed.signature,
        interval=(midpoint_idx + 1, len(path) - 1),
        ambient_dimension=sig_2_computed.ambient_dimension,
        depth=sig_2_computed.depth,
        basis_name=sig_2_computed.basis_name,
    )

    # Check that combining signatures with non-consecutive intervals raises a ValueError
    with pytest.raises(ValueError, match="The intervals of the signatures must be contiguous."):
        sig_1 @ sig_2_non_consecutive

    with pytest.raises(ValueError, match="The intervals of the signatures must be contiguous."):
        _chen_identity(sig_1, sig_2_non_consecutive)


@pytest.mark.parametrize("scalar_path_fixture", [(1, 20)], indirect=True)
@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_chen_identity_mismatched_ambient_dimension(scalar_path_fixture: jax.Array, depth: int):
    path = scalar_path_fixture
    midpoint_idx = len(path) // 2

    # Split the path into two overlapping segments
    path_1 = path[: midpoint_idx + 1]
    path_2 = path[midpoint_idx:]

    # Compute signatures for each sub-path.
    sig_1_computed = compute_path_signature(path_1, depth=depth, mode="full")
    sig_2_computed = compute_path_signature(path_2, depth=depth, mode="full")

    # Create signatures with different ambient dimensions
    sig_1 = Signature(
        signature=sig_1_computed.signature,
        interval=(0, midpoint_idx),
        ambient_dimension=sig_1_computed.ambient_dimension,
        depth=sig_1_computed.depth,
        basis_name=sig_1_computed.basis_name,
    )
    sig_2 = Signature(
        signature=sig_2_computed.signature,
        interval=(midpoint_idx, len(path) - 1),
        ambient_dimension=sig_2_computed.ambient_dimension + 1,  # Different ambient dimension
        depth=sig_2_computed.depth,
        basis_name=sig_2_computed.basis_name,
    )

    # Check that combining signatures with different ambient dimensions raises a ValueError
    with pytest.raises(ValueError, match="Signatures must have the same ambient_dimension."):
        sig_1 @ sig_2

    with pytest.raises(ValueError, match="Signatures must have the same ambient_dimension."):
        _chen_identity(sig_1, sig_2)


@pytest.mark.parametrize("scalar_path_fixture", [(1, 20)], indirect=True)
@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_chen_identity_mismatched_depth(scalar_path_fixture: jax.Array, depth: int):
    path = scalar_path_fixture
    midpoint_idx = len(path) // 2

    # Split the path into two overlapping segments
    path_1 = path[: midpoint_idx + 1]
    path_2 = path[midpoint_idx:]

    # Compute signatures for each sub-path.
    sig_1_computed = compute_path_signature(path_1, depth=depth, mode="full")
    sig_2_computed = compute_path_signature(path_2, depth=depth + 1, mode="full")  # Different depth

    # Create signatures with different depths
    sig_1 = Signature(
        signature=sig_1_computed.signature,
        interval=(0, midpoint_idx),
        ambient_dimension=sig_1_computed.ambient_dimension,
        depth=sig_1_computed.depth,
        basis_name=sig_1_computed.basis_name,
    )
    sig_2 = Signature(
        signature=sig_2_computed.signature,
        interval=(midpoint_idx, len(path) - 1),
        ambient_dimension=sig_2_computed.ambient_dimension,
        depth=sig_2_computed.depth,
        basis_name=sig_2_computed.basis_name,
    )

    # Check that combining signatures with different depths raises a ValueError
    with pytest.raises(ValueError, match="Signatures must have the same depth."):
        sig_1 @ sig_2

    with pytest.raises(ValueError, match="Signatures must have the same depth."):
        _chen_identity(sig_1, sig_2)


def test_signature_str_representation():
    """Test the __str__ method of Signature class."""
    # Create a simple signature for testing
    signature_terms = [jnp.array([[1.0, 2.0], [3.0, 4.0]]), jnp.array([[[5.0, 6.0], [7.0, 8.0]]])]
    sig = Signature(
        signature=signature_terms,
        interval=(0.0, 1.0),
        ambient_dimension=2,
        depth=2,
        basis_name="Tensor words",
    )

    str_repr = str(sig)

    # Check that the string representation contains expected information
    assert "depth=2" in str_repr
    assert "ambient_dimension=2" in str_repr
    assert "interval=(0.0, 1.0)" in str_repr
    assert "signature_shapes=" in str_repr


def test_log_signature_matmul_not_implemented():
    """Test that LogSignature.__matmul__ raises NotImplementedError."""
    # Create two LogSignature objects
    signature_terms = [jnp.array([[1.0, 2.0], [3.0, 4.0]])]
    log_sig_1 = LogSignature(
        signature=signature_terms,
        interval=(0.0, 1.0),
        ambient_dimension=2,
        depth=1,
        basis_name="Tensor words",
    )
    log_sig_2 = LogSignature(
        signature=signature_terms,
        interval=(1.0, 2.0),
        ambient_dimension=2,
        depth=1,
        basis_name="Tensor words",
    )

    # Check that combining LogSignatures raises NotImplementedError
    with pytest.raises(NotImplementedError, match="Product of log signatures is not defined."):
        log_sig_1 @ log_sig_2
