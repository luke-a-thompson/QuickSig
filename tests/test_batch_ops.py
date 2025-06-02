import jax
import jax.numpy as jnp
import pytest
from quicksig.tensor_ops import restricted_tensor_exp, tensor_log, tensor_product, seq_tensor_product, cauchy_prod


def test_batch_log_inverse_of_exp() -> None:
    """
    Tests if tensor_log correctly inverts restricted_tensor_exp
    for the first-order term when batching is handled by vmap.
    """
    B: int = 2  # Batch size
    n_features: int = 3  # Dimensionality of the initial tensor
    depth: int = 4  # Truncation depth for exp and log

    key = jax.random.PRNGKey(42)

    # 1. Create an initial random tensor X (batched)
    # restricted_tensor_exp expects unbatched input, vmap handles batching.
    initial_x_batched: jax.Array = jax.random.normal(key, shape=(B, n_features))

    # 2. Compute its tensor exponential: exp(X) = sum X^k/k!
    # restricted_tensor_exp returns a list of tensors. vmap maps over the first arg (initial_x_batched)
    # and treats depth as static. Output is a list of batched tensors.
    exp_terms_list = jax.vmap(restricted_tensor_exp, in_axes=(0, None))(initial_x_batched, depth)

    # For tensor_log, we need to flatten and concatenate these terms (already batched)
    exp_terms_flat_list = [term.reshape(B, -1) for term in exp_terms_list]
    exp_concatenated = jnp.concatenate(exp_terms_flat_list, axis=-1)

    expected_total_dim: int = sum(n_features**k for k in range(1, depth + 1))
    assert exp_concatenated.shape == (B, expected_total_dim), f"Shape mismatch for exp_concatenated: expected {(B, expected_total_dim)}, got {exp_concatenated.shape}"

    # 3. Compute the tensor logarithm of the exponentiated terms
    # tensor_log expects a list of unbatched tensors. vmap will pass a list of unbatched tensors
    # (sliced from exp_terms_list) to each call of tensor_log.
    # in_axes=(0, None) means:
    #   - 0 for exp_terms_list: vmap iterates through the batch dim of each tensor in the list.
    #   - None for n_features: it's a static argument.
    log_of_exp_terms = jax.vmap(tensor_log, in_axes=(0, None))(exp_terms_list, n_features)


    # Flatten the log terms (already batched)
    log_terms_flat_list = [term.reshape(B, -1) for term in log_of_exp_terms]
    log_concatenated = jnp.concatenate(log_terms_flat_list, axis=-1)


    assert log_concatenated.shape == (
        B,
        expected_total_dim,
    ), f"Shape mismatch for log_of_exp_concatenated: expected {(B, expected_total_dim)}, got {log_concatenated.shape}"

    # 4. Extract the first term from the log series. This should be our original X.
    # The log series is L = L1 + L2 + ...
    # L1 is the part we want, and it has n_features dimensions.
    # log_of_exp_terms is a list of batched tensors. log_of_exp_terms[0] is the first batched term (L1).
    recovered_x_flat: jax.Array = log_of_exp_terms[0].reshape(B, -1)[:, :n_features]
    recovered_x: jax.Array = recovered_x_flat.reshape(initial_x_batched.shape)

    # 5. Verify correctness
    # Note: Due to floating point precision and truncation depth,
    # it might not be exactly identical but should be very close.
    assert jnp.allclose(initial_x_batched, recovered_x, atol=1e-5), f"Initial X is not close to recovered X. Max diff: {jnp.max(jnp.abs(initial_x_batched - recovered_x))}"


# --- New Parameterized Tests ---


@pytest.mark.parametrize(
    "x_shape_suffix, y_shape_suffix, expected_out_suffix",
    [
        ((2,), (3,), (2, 3)),
        ((2, 4), (3,), (2, 4, 3)),  # x is (B, M, K), y is (B, N)
        ((2,), (3, 5), (2, 3, 5)),  # x is (B, M), y is (B, N, L)
        ((1,), (3,), (1, 3)),  # Edge case: M=1
        ((2,), (1,), (2, 1)),  # Edge case: N=1
        ((1,), (1,), (1, 1)),  # Edge case: M=1, N=1
    ],
)
def test_batch_tensor_product_shapes(x_shape_suffix: tuple[int, ...], y_shape_suffix: tuple[int, ...], expected_out_suffix: tuple[int, ...]) -> None:
    B: int = 4  # Batch size
    key = jax.random.PRNGKey(0)

    x_shape = (B,) + x_shape_suffix
    y_shape = (B,) + y_shape_suffix
    expected_shape = (B,) + expected_out_suffix

    x = jax.random.normal(key, x_shape)
    y = jax.random.normal(key, y_shape)

    result = jax.vmap(tensor_product)(x, y)
    assert result.shape == expected_shape, f"Input x:{x.shape}, y:{y.shape}. Expected {expected_shape}, got {result.shape}"


# Minimal value test for batch_tensor_product
def test_batch_tensor_product_values() -> None:
    B: int = 1
    # These are already batched with B=1
    x_b = jnp.array([[1.0, 2.0]])  # Shape (1, 2)
    y_b = jnp.array([[3.0, 4.0, 5.0]])  # Shape (1, 3)

    # Expected: (1, 2, 3)
    # x[b, i] * y[b, j]
    # result[0,0,0] = x[0,0]*y[0,0] = 1*3 = 3
    # result[0,0,1] = x[0,0]*y[0,1] = 1*4 = 4
    # result[0,0,2] = x[0,0]*y[0,2] = 1*5 = 5
    # result[0,1,0] = x[0,1]*y[0,0] = 2*3 = 6
    # result[0,1,1] = x[0,1]*y[0,1] = 2*4 = 8
    # result[0,1,2] = x[0,1]*y[0,2] = 2*5 = 10
    expected_result = jnp.array([[[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]]])

    result = jax.vmap(tensor_product)(x_b, y_b)
    assert result.shape == (B, 2, 3)
    assert jnp.allclose(result, expected_result)

    # Test with more feature dimensions
    # These are already batched with B=1
    x2_b = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
    y2_b = jnp.array([[5.0, 6.0]])  # (1, 2)
    # Expected: (1, 2, 2, 2)
    # result[0,i,k,j] = x2[0,i,k] * y2[0,j]
    # result[0,0,0,0] = x2[0,0,0]*y2[0,0] = 1*5=5
    # result[0,0,0,1] = x2[0,0,0]*y2[0,1] = 1*6=6
    expected_result2 = jnp.array([[[[5.0, 6.0], [10.0, 12.0]], [[15.0, 18.0], [20.0, 24.0]]]])
    result2 = jax.vmap(tensor_product)(x2_b, y2_b)
    assert result2.shape == (B, 2, 2, 2)
    assert jnp.allclose(result2, expected_result2)


@pytest.mark.parametrize(
    "x_shape_suffix, y_shape_suffix, expected_out_suffix",
    [
        ((2,), (3,), (2, 3)),  # x:(B,S,M), y:(B,S,N) -> (B,S,M,N)
        ((2, 4), (3,), (2, 4, 3)),  # x:(B,S,M,K), y:(B,S,N) -> (B,S,M,K,N)
        ((2,), (3, 5), (2, 3, 5)),  # x:(B,S,M), y:(B,S,N,L) -> (B,S,M,N,L)
        ((1,), (3,), (1, 3)),  # Edge: M=1
        ((2,), (1,), (2, 1)),  # Edge: N=1
        ((1,), (1,), (1, 1)),  # Edge: M=1, N=1
    ],
)
def test_batch_seq_tensor_product_shapes(x_shape_suffix: tuple[int, ...], y_shape_suffix: tuple[int, ...], expected_out_suffix: tuple[int, ...]) -> None:
    B: int = 3  # Batch size
    S: int = 5  # Sequence length
    key = jax.random.PRNGKey(1)

    x_shape = (B, S) + x_shape_suffix
    y_shape = (B, S) + y_shape_suffix
    expected_shape = (B, S) + expected_out_suffix

    x = jax.random.normal(key, x_shape)
    y = jax.random.normal(key, y_shape)

    result = jax.vmap(seq_tensor_product)(x, y)
    assert result.shape == expected_shape, f"Input x:{x.shape}, y:{y.shape}. Expected {expected_shape}, got {result.shape}"


def test_batch_seq_tensor_product_values() -> None:
    B: int = 1
    S: int = 1
    # x_b_s: (B, S, M) = (1,1,2)
    x_b_s = jnp.array([[[1.0, 2.0]]])
    # y_b_s: (B, S, N) = (1,1,3)
    y_b_s = jnp.array([[[3.0, 4.0, 5.0]]])

    # Expected: (B, S, M, N) = (1,1,2,3)
    expected_result = jnp.array([[[[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]]]])

    result = jax.vmap(seq_tensor_product)(x_b_s, y_b_s)
    assert result.shape == (B, S, 2, 3)
    assert jnp.allclose(result, expected_result)

    # Test with more feature dimensions and S > 1
    B2: int = 1
    S2: int = 2
    # x2_b_s: (B2, S2, M, K) = (1, 2, 2, 2)
    x2_b_s = jnp.array([[[[1.0, 2.0], [3.0, 4.0]], [[1.1, 2.1], [3.1, 4.1]]]])
    # y2_b_s: (B2, S2, N) = (1, 2, 2)
    y2_b_s = jnp.array([[[5.0, 6.0], [5.1, 6.1]]])


    # Manually construct expected output by applying vmap(tensor_product) per sequence element
    # and then stacking. This also indirectly tests tensor_product logic.
    # x2_b_s[:, 0, :, :] has shape (B2, M, K)
    # y2_b_s[:, 0, :] has shape (B2, N)
    expected_s0 = jax.vmap(tensor_product)(x2_b_s[:, 0, :, :], y2_b_s[:, 0, :])  # Shape (B2, M, K, N)
    expected_s1 = jax.vmap(tensor_product)(x2_b_s[:, 1, :, :], y2_b_s[:, 1, :])  # Shape (B2, M, K, N)
    expected_result2 = jnp.stack([expected_s0, expected_s1], axis=1)  # Shape (B2, S2, M, K, N)

    result2 = jax.vmap(seq_tensor_product)(x2_b_s, y2_b_s)
    assert result2.shape == (B2, S2, 2, 2, 2), f"Expected {(B2, S2, 2, 2, 2)}, got {result2.shape}"
    assert jnp.allclose(result2, expected_result2)


@pytest.mark.parametrize(
    "depth, n_features",
    [
        (1, 3),
        (2, 2),
        (3, 2),  # A common case
        (5, 1),  # Edge case: n_features = 1
    ],
)
def test_batch_restricted_tensor_exp_output_structure(depth: int, n_features: int) -> None:
    B: int = 2
    key = jax.random.PRNGKey(depth + n_features)
    # x_b is batched input for vmap
    x_b: jax.Array = jax.random.normal(key, shape=(B, n_features))

    # restricted_tensor_exp takes unbatched x, depth is static
    # Output is a list of batched tensors
    result_list = jax.vmap(restricted_tensor_exp, in_axes=(0, None))(x_b, depth)

    assert len(result_list) == depth, f"Expected tuple of length {depth}, got {len(result_list)}"

    for k_idx, term in enumerate(result_list):
        order = k_idx + 1
        expected_term_shape = (B,) + (n_features,) * order
        assert term.shape == expected_term_shape, f"Term {k_idx} (order {order}) has shape {term.shape}, expected {expected_term_shape}"


def test_batch_restricted_tensor_exp_values() -> None:
    B: int = 1
    n_features: int = 2
    # x_val_b is batched input (B=1)
    x_val_b = jnp.array([[1.0, 2.0]])  # Shape (B, n_features)

    # Depth 1
    # restricted_tensor_exp is vmapped over x_val_b, depth is static
    result_depth1 = jax.vmap(restricted_tensor_exp, in_axes=(0, None))(x_val_b, 1)
    assert len(result_depth1) == 1
    assert jnp.allclose(result_depth1[0], x_val_b)

    # Depth 2: (x, x^2/2!)
    result_depth2 = jax.vmap(restricted_tensor_exp, in_axes=(0, None))(x_val_b, 2)
    assert len(result_depth2) == 2
    assert jnp.allclose(result_depth2[0], x_val_b)

    # x_val_b is (B, n_features)
    # tensor_product is vmapped over (batched x, batched x/2)
    term2_expected = jax.vmap(tensor_product)(x_val_b, x_val_b / 2.0)
    assert jnp.allclose(result_depth2[1], term2_expected)

    # Depth 3: (x, x^2/2!, x^3/3!)
    result_depth3 = jax.vmap(restricted_tensor_exp, in_axes=(0, None))(x_val_b, 3)
    assert len(result_depth3) == 3
    assert jnp.allclose(result_depth3[0], x_val_b)
    assert jnp.allclose(result_depth3[1], term2_expected)  # From depth 2 test

    # result_depth3[1] is batched (output from vmapped restricted_tensor_exp)
    # tensor_product is vmapped over (batched term, batched x/3)
    term3_expected = jax.vmap(tensor_product)(result_depth3[1], x_val_b / 3.0)
    assert jnp.allclose(result_depth3[2], term3_expected)


@pytest.mark.parametrize(
    "x_terms_defs, y_terms_defs, depth, n_features",
    [
        # Case 1: Basic - X=(X1), Y=(Y1), depth=2. Expect Z=(0, X1⊗Y1)
        ([(1,)], [(1,)], 2, 2),  # x_terms: one term of order 1  # y_terms: one term of order 1  # depth, n_features
        # Case 2: X=(X1,X2), Y=(Y1), depth=3. Expect Z=(0, X1⊗Y1, X2⊗Y1)
        ([(1,), (2,)], [(1,)], 3, 2),  # x_terms: order 1, order 2  # y_terms: order 1
        # Case 3: X=(X1), Y=(Y1,Y2), depth=3. Expect Z=(0, X1⊗Y1, X1⊗Y2)
        ([(1,)], [(1,), (2,)], 3, 2),
        # Case 4: X=(X1,X2), Y=(Y1,Y2), depth=3. Expect Z=(0, X1⊗Y1, X1⊗Y2 + X2⊗Y1)
        ([(1,), (2,)], [(1,), (2,)], 3, 2),
        # Case 5: Deeper - X=(X1,X2), Y=(Y1,Y2), depth=4
        # Z1=0, Z2=X1Y1, Z3=X1Y2+X2Y1, Z4=X2Y2
        ([(1,), (2,)], [(1,), (2,)], 4, 2),
        # Case 6: Truncation by depth - X=(X1,X2,X3), Y=(Y1), depth=2. Expect Z=(0, X1Y1) (X2Y1, X3Y1 ignored)
        ([(1,), (2,), (3,)], [(1,)], 2, 2),
        # Case 7: n_features = 1
        ([(1,)], [(1,)], 2, 1),
    ],
)
def test_batch_cauchy_prod_logic(x_terms_defs: list[tuple[int, ...]], y_terms_defs: list[tuple[int, ...]], depth: int, n_features: int) -> None:
    B: int = 2
    key = jax.random.PRNGKey(sum(d[0] for d in x_terms_defs) + sum(d[0] for d in y_terms_defs) + depth + n_features)

    def _create_terms(term_defs: list[tuple[int, ...]], current_key: jax.Array, batch_size: int) -> list[jax.Array]:
        terms = []
        for i, order_def in enumerate(term_defs):
            order = order_def[0]  # Assuming the first element in tuple is the order indicator
            current_key, subkey = jax.random.split(current_key)
            # Create batched terms directly
            term_shape = (batch_size,) + (n_features,) * order
            # Small integer values for easier debugging if needed, scaled by order
            terms.append(jax.random.uniform(subkey, term_shape, minval=1, maxval=3) * order)
        return terms

    key, x_key, y_key, s_key = jax.random.split(key, 4)
    # x_terms and y_terms are lists of batched tensors
    x_terms: list[jax.Array] = _create_terms(x_terms_defs, x_key, B)
    y_terms: list[jax.Array] = _create_terms(y_terms_defs, y_key, B)

    # S_levels_shapes is also a list of batched tensors (shapes)
    S_levels_shapes: list[jax.Array] = []
    for i in range(1, depth + 1):
        s_key, subkey = jax.random.split(s_key)
        S_levels_shapes.append(jax.random.normal(subkey, (B,) + (n_features,) * i)) # dummy content, shape is key

    # Calculate expected output manually (already batched)
    expected_out_terms: list[jax.Array] = [jnp.zeros_like(S_levels_shapes[k]) for k in range(depth)]

    # Loop over output orders (k_out for Z^{(k_out+1)})
    # Z order k_out+1. This means k_out index in expected_out_terms.
    for k_out in range(depth): # k_out from 0 to depth-1
        target_order_Z = k_out + 1 # order of Z term we are computing
        current_sum_for_order_Zk = jnp.zeros_like(S_levels_shapes[k_out])

        # Sum over X_i Y_j where order(X_i) + order(Y_j) = target_order_Z
        for i_x_term_idx in range(len(x_terms)):
            x_term_order = x_terms_defs[i_x_term_idx][0] # Actual order of X term, e.g. 1, 2...
            
            for j_y_term_idx in range(len(y_terms)):
                y_term_order = y_terms_defs[j_y_term_idx][0] # Actual order of Y term

                if x_term_order + y_term_order == target_order_Z:
                    # x_terms[i_x_term_idx] is (B, feat_x)
                    # y_terms[j_y_term_idx] is (B, feat_y)
                    # vmap tensor_product over these batched tensors
                    prod = jax.vmap(tensor_product)(x_terms[i_x_term_idx], y_terms[j_y_term_idx])
                    current_sum_for_order_Zk += prod
        if target_order_Z > 0 : # Z_0 is always zero in this context (not computed by cauchy_prod)
             expected_out_terms[k_out] = current_sum_for_order_Zk


    # cauchy_prod expects lists of unbatched tensors.
    # vmap handles the batching. in_axes=(0,0,None,0) means:
    #   - x_terms: each tensor in list is unstacked at axis 0
    #   - y_terms: each tensor in list is unstacked at axis 0
    #   - depth: static
    #   - S_levels_shapes: each tensor in list is unstacked at axis 0
    result_terms = jax.vmap(cauchy_prod, in_axes=(0, 0, None, 0))(x_terms, y_terms, depth, S_levels_shapes)

    assert len(result_terms) == depth
    for i in range(depth):
        assert result_terms[i].shape == S_levels_shapes[i].shape, f"Term {i} shape mismatch. Expected {S_levels_shapes[i].shape}, got {result_terms[i].shape}"
        # Check if expected_out_terms[i] is non-zero or if result_terms[i] is also zero
        # This handles cases where an order might not be produced (e.g. Z1 for X=(X2), Y=(Y2))
        if jnp.any(expected_out_terms[i] != 0) or jnp.any(result_terms[i] != 0):
             assert jnp.allclose(result_terms[i], expected_out_terms[i], atol=1e-5), f"Term {i} (order {i+1}) mismatch. Got:\n{result_terms[i]}\nExpected:\n{expected_out_terms[i]}"
        else: # Both are zero, which is fine
            pass


@pytest.mark.parametrize("depth, n_features", [(1, 3), (2, 2), (3, 2), (1, 1)])  # Edge case depth 1, n_features 1
def test_batch_tensor_log_specific_depths(depth: int, n_features: int) -> None:
    B: int = 2  # Batch size
    key = jax.random.PRNGKey(depth * 10 + n_features)

    # initial_x_batched is the L1 term we expect to recover, already batched.
    initial_x_batched: jax.Array = jax.random.normal(key, shape=(B, n_features))

    # This list will contain batched tensors representing the signature S
    sig_terms_list_batched: list[jax.Array]

    if depth == 1:
        # For depth 1, sig S is just X (our L1). log(S) should be X.
        # tensor_log expects unbatched terms, so vmap handles the list of batched terms.
        sig_terms_list_batched = [initial_x_batched.reshape(B, -1)]
        expected_log_terms_batched = [initial_x_batched.reshape(B, -1)]
    else:
        # For depth > 1, construct S = exp(L) where L has only L1 = initial_x_batched and Lk=0 for k>1.
        # restricted_tensor_exp is vmapped. initial_x_batched is (B, n_features).
        # Output exp_terms_of_x is list[batched_tensor].
        exp_terms_of_x: list[jax.Array] = jax.vmap(restricted_tensor_exp, in_axes=(0, None))(initial_x_batched, depth)

        # These terms form the signature S (list of batched tensors)
        sig_terms_list_batched = [term.reshape(B, -1) for term in exp_terms_of_x]

        # The expected log output should be (X, 0, 0, ..., 0) when batched.
        # The first term (L1) is X, subsequent terms (L2, L3, ...) should be close to zero.
        expected_l1_batched_flat = initial_x_batched.reshape(B, -1)
        expected_log_terms_batched: list[jax.Array] = [expected_l1_batched_flat]
        for i in range(2, depth + 1):
            order_i_dim = n_features**i
            expected_log_terms_batched.append(jnp.zeros((B, order_i_dim)))

    # Compute the tensor logarithm using vmap
    # tensor_log takes (list_of_unbatched_tensors, n_features)
    # vmap handles sig_terms_list_batched (which is list[batched_tensor])
    # and provides list[unbatched_tensor_slice] to each call of tensor_log.
    log_output_terms_batched = jax.vmap(tensor_log, in_axes=(0, None))(sig_terms_list_batched, n_features)

    # Verify shapes and values for each term (all terms are batched)
    assert len(log_output_terms_batched) == depth, f"Expected {depth} terms, got {len(log_output_terms_batched)}"
    for i, (output_term, expected_term) in enumerate(zip(log_output_terms_batched, expected_log_terms_batched)):
        assert output_term.shape == expected_term.shape, f"Term {i} shape mismatch: got {output_term.shape}, expected {expected_term.shape}"
        assert jnp.allclose(output_term, expected_term, atol=1e-5), f"Term {i} value mismatch for depth={depth}, n_features={n_features}.\nGot:\n{output_term}\nExpected:\n{expected_term}"
