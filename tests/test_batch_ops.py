import jax
import jax.numpy as jnp
import pytest  # Add pytest import for parameterization
from quicksig.batch_ops import batch_restricted_exp, batch_tensor_log, batch_tensor_product, batch_seq_tensor_product, batch_cauchy_prod


def test_batch_log_inverse_of_exp() -> None:
    """
    Tests if batch_tensor_log correctly inverts batch_restricted_exp
    for the first-order term.
    """
    B: int = 2  # Batch size
    n_features: int = 3  # Dimensionality of the initial tensor
    depth: int = 4  # Truncation depth for exp and log

    key = jax.random.PRNGKey(42)

    # 1. Create an initial random tensor X
    initial_x: jax.Array = jax.random.normal(key, shape=(B, n_features))

    # 2. Compute its tensor exponential: exp(X) = sum X^k/k!
    exp_terms_list = batch_restricted_exp(initial_x, depth)

    # For batch_tensor_log, we need to flatten and concatenate these terms
    exp_terms_flat_list = [term.reshape(B, -1) for term in exp_terms_list]
    exp_concatenated = jnp.concatenate(exp_terms_flat_list, axis=-1)

    expected_total_dim: int = sum(n_features**k for k in range(1, depth + 1))
    assert exp_concatenated.shape == (B, expected_total_dim), f"Shape mismatch for exp_concatenated: expected {(B, expected_total_dim)}, got {exp_concatenated.shape}"

    # 3. Compute the tensor logarithm of the exponentiated terms
    log_of_exp_concatenated = batch_tensor_log(exp_terms_list, n_features)

    exp_terms_flat_list = [term.reshape(B, -1) for term in log_of_exp_concatenated]
    exp_concatenated = jnp.concatenate(exp_terms_flat_list, axis=-1)

    assert exp_concatenated.shape == (
        B,
        expected_total_dim,
    ), f"Shape mismatch for log_of_exp_concatenated: expected {(B, expected_total_dim)}, got {log_of_exp_concatenated[0].shape}"

    # 4. Extract the first term from the log series. This should be our original X.
    # The log series is L = L1 + L2 + ...
    # L1 is the part we want, and it has n_features dimensions.
    recovered_x_flat: jax.Array = log_of_exp_concatenated[0][:, :n_features]
    recovered_x: jax.Array = recovered_x_flat.reshape(initial_x.shape)

    # 5. Verify correctness
    # Note: Due to floating point precision and truncation depth,
    # it might not be exactly identical but should be very close.
    assert jnp.allclose(initial_x, recovered_x, atol=1e-5), f"Initial X is not close to recovered X. Max diff: {jnp.max(jnp.abs(initial_x - recovered_x))}"


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

    result = batch_tensor_product(x, y)
    assert result.shape == expected_shape, f"Input x:{x.shape}, y:{y.shape}. Expected {expected_shape}, got {result.shape}"


# Minimal value test for batch_tensor_product
def test_batch_tensor_product_values() -> None:
    B: int = 1
    x = jnp.array([[1.0, 2.0]])  # Shape (1, 2)
    y = jnp.array([[3.0, 4.0, 5.0]])  # Shape (1, 3)

    # Expected: (1, 2, 3)
    # x[b, i] * y[b, j]
    # result[0,0,0] = x[0,0]*y[0,0] = 1*3 = 3
    # result[0,0,1] = x[0,0]*y[0,1] = 1*4 = 4
    # result[0,0,2] = x[0,0]*y[0,2] = 1*5 = 5
    # result[0,1,0] = x[0,1]*y[0,0] = 2*3 = 6
    # result[0,1,1] = x[0,1]*y[0,1] = 2*4 = 8
    # result[0,1,2] = x[0,1]*y[0,2] = 2*5 = 10
    expected_result = jnp.array([[[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]]])

    result = batch_tensor_product(x, y)
    assert result.shape == (B, 2, 3)
    assert jnp.allclose(result, expected_result)

    # Test with more feature dimensions
    x2 = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
    y2 = jnp.array([[5.0, 6.0]])  # (1, 2)
    # Expected: (1, 2, 2, 2)
    # result[0,i,k,j] = x2[0,i,k] * y2[0,j]
    # result[0,0,0,0] = x2[0,0,0]*y2[0,0] = 1*5=5
    # result[0,0,0,1] = x2[0,0,0]*y2[0,1] = 1*6=6
    expected_result2 = jnp.array([[[[5.0, 6.0], [10.0, 12.0]], [[15.0, 18.0], [20.0, 24.0]]]])
    result2 = batch_tensor_product(x2, y2)
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

    result = batch_seq_tensor_product(x, y)
    assert result.shape == expected_shape, f"Input x:{x.shape}, y:{y.shape}. Expected {expected_shape}, got {result.shape}"


def test_batch_seq_tensor_product_values() -> None:
    B: int = 1
    S: int = 1
    # x: (B, S, M) = (1,1,2)
    x = jnp.array([[[1.0, 2.0]]])
    # y: (B, S, N) = (1,1,3)
    y = jnp.array([[[3.0, 4.0, 5.0]]])

    # Expected: (B, S, M, N) = (1,1,2,3)
    expected_result = jnp.array([[[[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]]]])

    result = batch_seq_tensor_product(x, y)
    assert result.shape == (B, S, 2, 3)
    assert jnp.allclose(result, expected_result)

    # Test with more feature dimensions and S > 1
    B2: int = 1
    S2: int = 2
    # x: (1, 2, 2, 2)
    x2 = jnp.array([[[[1.0, 2.0], [3.0, 4.0]], [[1.1, 2.1], [3.1, 4.1]]]])
    # y: (1, 2, 2)
    y2 = jnp.array([[[5.0, 6.0], [5.1, 6.1]]])

    # For B=0, S=0:
    # x_val = x2[0,0,:,:] = [[1,2],[3,4]]
    # y_val = y2[0,0,:] = [5,6]
    # out_val[0,0,m,k,n] = x_val[m,k] * y_val[n]
    # out_val[0,0,0,0,0] = 1*5 = 5
    # out_val[0,0,0,0,1] = 1*6 = 6
    # ...etc
    term1_s0 = batch_tensor_product(x2[:, 0, :, :], y2[:, 0, :])  # (B, M, K, N)
    term1_s1 = batch_tensor_product(x2[:, 1, :, :], y2[:, 1, :])  # (B, M, K, N)

    # Manually construct expected output by applying batch_tensor_product per sequence element
    # and then stacking. This also indirectly tests batch_tensor_product logic.
    expected_s0 = batch_tensor_product(x2[:, 0, :, :], y2[:, 0, :])  # Shape (B2, 2, 2, 2)
    expected_s1 = batch_tensor_product(x2[:, 1, :, :], y2[:, 1, :])  # Shape (B2, 2, 2, 2)
    expected_result2 = jnp.stack([expected_s0, expected_s1], axis=1)  # Shape (B2, S2, 2, 2, 2)

    result2 = batch_seq_tensor_product(x2, y2)
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
def test_batch_restricted_exp_output_structure(depth: int, n_features: int) -> None:
    B: int = 2
    key = jax.random.PRNGKey(depth + n_features)
    x: jax.Array = jax.random.normal(key, shape=(B, n_features))

    result_list = batch_restricted_exp(x, depth)

    assert len(result_list) == depth, f"Expected tuple of length {depth}, got {len(result_list)}"

    for k_idx, term in enumerate(result_list):
        order = k_idx + 1
        expected_term_shape = (B,) + (n_features,) * order
        assert term.shape == expected_term_shape, f"Term {k_idx} (order {order}) has shape {term.shape}, expected {expected_term_shape}"


def test_batch_restricted_exp_values() -> None:
    B: int = 1
    n_features: int = 2
    x_val = jnp.array([[1.0, 2.0]])  # Shape (B, n_features)

    # Depth 1
    result_depth1 = batch_restricted_exp(x_val, 1)
    assert len(result_depth1) == 1
    assert jnp.allclose(result_depth1[0], x_val)

    # Depth 2: (x, x^2/2!)
    # x^1/1! = [[1, 2]]
    # x^2/2! = batch_tensor_product([[1,2]], [[1,2]]/2)
    #          = batch_tensor_product([[1,2]], [[0.5,1]])
    #          = [[[1*0.5, 1*1], [2*0.5, 2*1]]] = [[[0.5, 1], [1, 2]]]
    result_depth2 = batch_restricted_exp(x_val, 2)
    assert len(result_depth2) == 2
    assert jnp.allclose(result_depth2[0], x_val)

    term2_expected = batch_tensor_product(x_val, x_val / 2.0)
    assert jnp.allclose(result_depth2[1], term2_expected)

    # Depth 3: (x, x^2/2!, x^3/3!)
    # x^3/3! = batch_tensor_product(x^2/2!, x/3)
    result_depth3 = batch_restricted_exp(x_val, 3)
    assert len(result_depth3) == 3
    assert jnp.allclose(result_depth3[0], x_val)
    assert jnp.allclose(result_depth3[1], term2_expected)  # From depth 2 test

    term3_expected = batch_tensor_product(result_depth3[1], x_val / 3.0)
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

    def _create_terms(term_defs: list[tuple[int, ...]], current_key: jax.Array) -> list[jax.Array]:
        terms = []
        for i, order_def in enumerate(term_defs):
            order = order_def[0]  # Assuming the first element in tuple is the order indicator
            current_key, subkey = jax.random.split(current_key)
            term_shape = (B,) + (n_features,) * order
            # Small integer values for easier debugging if needed, scaled by order
            terms.append(jax.random.uniform(subkey, term_shape, minval=1, maxval=3) * order)
        return terms

    key, x_key, y_key, s_key = jax.random.split(key, 4)
    x_terms: list[jax.Array] = _create_terms(x_terms_defs, x_key)
    y_terms: list[jax.Array] = _create_terms(y_terms_defs, y_key)

    S_levels_shapes: list[jax.Array] = []
    for i in range(1, depth + 1):
        s_key, subkey = jax.random.split(s_key)
        # These are just for shape, content doesn't matter for jnp.zeros_like
        S_levels_shapes.append(jax.random.normal(subkey, (B,) + (n_features,) * i))

    # Calculate expected output manually
    # expected_out[i] holds Z^{(i+1)}
    expected_out_terms: list[jax.Array] = [jnp.zeros_like(S_levels_shapes[k]) for k in range(depth)]
    for i_out_idx in range(1, depth):  # output index for Z, computes Z^{(i_out_idx+1)}
        # We want Z^{(i_out_idx+1)} = sum_{ (j+1)+(k+1) = i_out_idx+1 } X^{(j+1)} Y^{(k+1)}
        # (j from 0 to i_out_idx-1 for X terms, k from 0 for Y terms)
        # j_term_idx ranges from 0 to len(x_terms)-1
        # k_term_idx ranges from 0 to len(y_terms)-1
        current_sum_for_order = jnp.zeros_like(expected_out_terms[i_out_idx])
        for j_term_idx in range(len(x_terms)):
            x_term_order = x_terms_defs[j_term_idx][0]
            # k_term_order needed = (i_out_idx + 1) - x_term_order
            # k_term_idx corresponds to k_term_order - 1
            k_term_order_needed = (i_out_idx + 1) - x_term_order
            k_term_idx_needed = k_term_order_needed - 1

            if k_term_idx_needed >= 0 and k_term_idx_needed < len(y_terms) and y_terms_defs[k_term_idx_needed][0] == k_term_order_needed:
                prod = batch_tensor_product(x_terms[j_term_idx], y_terms[k_term_idx_needed])
                current_sum_for_order += prod
        expected_out_terms[i_out_idx] = current_sum_for_order

    result_terms = batch_cauchy_prod(x_terms, y_terms, depth, S_levels_shapes)

    assert len(result_terms) == depth
    for i in range(depth):
        assert result_terms[i].shape == S_levels_shapes[i].shape, f"Term {i} shape mismatch"
        assert jnp.allclose(result_terms[i], expected_out_terms[i], atol=1e-5), f"Term {i} (order {i+1}) mismatch. Got:\n{result_terms[i]}\nExpected:\n{expected_out_terms[i]}"


@pytest.mark.parametrize("depth, n_features", [(1, 3), (2, 2), (3, 2), (1, 1)])  # Edge case depth 1, n_features 1
def test_batch_tensor_log_specific_depths(depth: int, n_features: int) -> None:
    B: int = 2  # Batch size
    key = jax.random.PRNGKey(depth * 10 + n_features)

    # 1. Create an initial tensor X (this will be our L1, the first log term)
    initial_x: jax.Array = jax.random.normal(key, shape=(B, n_features))

    if depth == 1:
        # For depth 1, sig_flat is just X. log(X) should be X.
        sig_flat_input_list = [initial_x.reshape(B, -1)]
        expected_log_terms = [initial_x.reshape(B, -1)]
    else:
        # For depth > 1, we construct a signature S = exp(L) where L has only L1 = X and Lk=0 for k>1.
        # Then log(S) should give back L (i.e., only L1=X should be non-zero).
        # The `batch_restricted_exp` computes terms of X, X^2/2!, ..., X^depth/depth!
        # These terms form the signature S if L = (X, 0, 0, ...)
        exp_terms_of_x: list[jax.Array] = batch_restricted_exp(initial_x, depth)

        # Concatenate these terms to form sig_flat_input for batch_tensor_log
        sig_flat_input_list: list[jax.Array] = [term.reshape(B, -1) for term in exp_terms_of_x]

        # The expected log output should be (X, 0, 0, ..., 0) when flattened.
        # The first term (L1) is X, subsequent terms (L2, L3, ...) should be close to zero.
        expected_l1_flat = initial_x.reshape(B, -1)
        expected_log_terms: list[jax.Array] = [expected_l1_flat]
        for i in range(2, depth + 1):
            order_i_dim = n_features**i
            expected_log_terms.append(jnp.zeros((B, order_i_dim)))

    # Compute the tensor logarithm
    log_output_terms = batch_tensor_log(sig_flat_input_list, n_features)

    # Verify shapes and values for each term
    assert len(log_output_terms) == depth, f"Expected {depth} terms, got {len(log_output_terms)}"
    for i, (output_term, expected_term) in enumerate(zip(log_output_terms, expected_log_terms)):
        assert output_term.shape == expected_term.shape, f"Term {i} shape mismatch: got {output_term.shape}, expected {expected_term.shape}"
        assert jnp.allclose(output_term, expected_term, atol=1e-5), f"Term {i} value mismatch for depth={depth}, n_features={n_features}.\nGot:\n{output_term}\nExpected:\n{expected_term}"
