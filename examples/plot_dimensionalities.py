import jax.numpy as jnp
from stochastax.analytics.signature_sizes import (
    get_signature_dim,
    get_log_signature_dim,
)

# Global flag to control log signature plotting
PLOT_LOG_SIGNATURE = True

# Generate dimensions from 1 to 30
dims = jnp.arange(1, 31)
# For depth=3, use the get_signature_dim function
depth = 3
sig_dims = [get_signature_dim(depth, dim) for dim in dims]

# Calculate log signature dimensions if flag is enabled
if PLOT_LOG_SIGNATURE:
    log_sig_dims = [get_log_signature_dim(depth, dim) for dim in dims]

import matplotlib.pyplot as plt

# USYD ochre color hex: #e64626
usyd_ochre = "#e64626"

# Set up the figure with a clean, professional style
plt.figure(figsize=(10, 7))
plt.style.use("seaborn-v0_8-whitegrid")

# Create the main plot with enhanced styling as a pure line (no markers)
plt.plot(dims, sig_dims, color=usyd_ochre, linewidth=3, label="Signature Dimension")

# Add log signature line if flag is enabled
if PLOT_LOG_SIGNATURE:
    plt.plot(dims, log_sig_dims, color="blue", linewidth=3, label="Log-Signature Dimension")

# Enhance the plot appearance
plt.xlabel("Path Dimension", fontsize=14, fontweight="bold")
plt.ylabel("Signature Dimension (depth=3)", fontsize=14, fontweight="bold")
plt.title(
    "Signature Dimension Growth with Path Dimension",
    fontsize=16,
    fontweight="bold",
    pad=20,
)

# Customize grid and axes
plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
plt.gca().set_facecolor("#fafafa")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_linewidth(1.5)
plt.gca().spines["bottom"].set_linewidth(1.5)

# Add labels at x = 1, 3, 5, 10, 15, then every 5 up to 32
label_xs = [1, 3, 5, 10, 15] + list(range(20, 31, 5))
label_xs = [x for x in label_xs if x <= 30]
annotate_indices = [int(jnp.argmin(jnp.abs(dims - x))) for x in label_xs]
for idx in annotate_indices:
    x = float(dims[idx])
    y = float(sig_dims[idx])
    plt.annotate(
        f"{int(round(y))}",
        (x, y),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
    )

# Set axis limits with some padding
plt.xlim(jnp.min(dims) - 1, jnp.max(dims) + 1)
plt.ylim(jnp.min(sig_dims) * 0.8, jnp.max(sig_dims) * 1.2)

# Add subtle legend
plt.legend(frameon=True, fancybox=True, shadow=True, loc="upper left", fontsize=12)

# Tight layout for better spacing
plt.tight_layout()

# Save with high quality
plt.savefig(
    "/home/luke/QuickSig/examples/signature_dimensionalities.pdf",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
