# Continuity of the Lyons map vs. discontinuity of the Itô map
#
# Figure: Driver error (||X^(n)-X||_inf) vs Solution error (||Y^(n)-Z||_inf)
# - Y^(n) solves dY = Y dX^(n) where X^(n) is the piecewise-linear
#   approximation of a fixed Brownian path X on a coarse mesh.
# - By Wong–Zakai, Y^(n) -> Y (Stratonovich/RDE) with Y = Y0 * exp(X).
# - If we (incorrectly) target the Itô solution \tilde{Y} = Y0 * exp(X - t/2),
#   the error \|Y^(n) - \tilde{Y}\|_\infty does NOT vanish -> "error floor".
#
# Implementation notes:
# - We generate one fine Brownian path X on a fine grid (N_fine).
# - For each coarse n (number of intervals), we downsample to coarse knots
#   and linearly interpolate back to the fine grid to obtain X^(n).
# - IMPORTANT (Wong–Zakai nuance): We evaluate Y^(n) on a *finer* grid
#   than the noise generation grid for X^(n) (here, the entire fine grid)
#   to emulate solving the RDE/ODE with substeps within each coarse interval.
#
# Output:
# - A Matplotlib plot with two series:
#     1) Continuity (RDE/Stratonovich target): ||Y^(n) - Y||_inf vs ||X^(n) - X||_inf
#     2) Discontinuity (Itô target):           ||Y^(n) - \tilde{Y}||_inf vs ||X^(n) - X||_inf
# - A PNG is saved to /mnt/data and a small printed table of the errors.

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import jax
import jax.numpy as jnp
from quicksig.signatures.compute_path_signature import compute_path_signature

# Reproducibility
rng = np.random.default_rng(12345)

T = 1.0
N_fine = 2**14  # fine evaluation grid (16384)
dt_fine = T / N_fine
t_fine = np.linspace(0.0, T, N_fine + 1)

# Build a single Brownian path on the fine grid
dW = rng.normal(loc=0.0, scale=np.sqrt(dt_fine), size=N_fine)
X_fine = np.concatenate([[0.0], np.cumsum(dW)])  # standard Brownian motion
assert X_fine.shape == (N_fine + 1,)

Y0 = 1.0

# "True" Stratonovich/RDE solution for dY = Y ∘ dX with Brownian driver
Y_strat = Y0 * np.exp(X_fine)

# "True" Itô solution: dY = Y dX (Itô), closed form Y = Y0 * exp(X - 0.5 t)
Y_ito = Y0 * np.exp(X_fine - 0.5 * t_fine)

# Choose coarse driver meshes (must divide N_fine to align knots)
coarse_ns = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
coarse_ns = [n for n in coarse_ns if N_fine % n == 0 and n <= N_fine]

driver_errs = []
rde_sol_errs = []
ito_sol_errs = []

for n in coarse_ns:
    # Coarse knot times and indices
    dt_coarse = T / n
    t_coarse = np.linspace(0.0, T, n + 1)
    idx_coarse = (t_coarse * N_fine / T).astype(int)

    # Sample the fine Brownian path at coarse knots
    X_coarse = X_fine[idx_coarse]

    # Piecewise-linear interpolation back to the fine grid to form X^(n)
    X_n = np.interp(t_fine, t_coarse, X_coarse)

    # RDE/Young solution for smooth driver X^(n) to dY = Y dX^(n) is explicit:
    # Y^(n)(t) = Y0 * exp( X^(n)(t) ), since dY/Y = dX^(n).
    Y_n = Y0 * np.exp(X_n)

    # Errors (sup over the fine grid)
    driver_err = np.max(np.abs(X_n - X_fine))
    rde_err = np.max(np.abs(Y_n - Y_strat))     # should -> 0 (Lyons continuity)
    ito_err = np.max(np.abs(Y_n - Y_ito))       # should -> const > 0 (Itô discontinuity)

    driver_errs.append(driver_err)
    rde_sol_errs.append(rde_err)
    ito_sol_errs.append(ito_err)

# Convert to arrays for plotting / table
driver_errs = np.array(driver_errs)
rde_sol_errs = np.array(rde_sol_errs)
ito_sol_errs = np.array(ito_sol_errs)

# --- Plot ---
plt.figure(figsize=(10, 7))
plt.style.use('seaborn-v0_8-whitegrid')
usyd_ochre = "#e64626"

# avoid zero driver errors (log scale)
_valid = driver_errs > 0

plt.loglog(driver_errs[_valid], rde_sol_errs[_valid],
         color=usyd_ochre,
         marker="o",
         markersize=8,
         linewidth=3,
         markerfacecolor=usyd_ochre,
         markeredgecolor='white',
         markeredgewidth=2,
         label='RDE/Stratonovich target')

plt.loglog(driver_errs[_valid], ito_sol_errs[_valid],
         color="#1f77b4",
         marker="^",
         markersize=8,
         linewidth=3,
         markerfacecolor="#1f77b4",
         markeredgecolor='white',
         markeredgewidth=2,
         label='Itô target')

plt.xlabel(r"Driver error $\|X^{(n)}-X\|_\infty$", fontsize=14, fontweight='bold')
plt.ylabel(r"Solution error $\|Y^{(n)}-Z\|_\infty$", fontsize=14, fontweight='bold')
plt.title("Driver error vs solution error (fixed Brownian path)", fontsize=16, fontweight='bold', pad=20)

plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax = plt.gca()
ax.set_facecolor('#fafafa')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

plt.legend(frameon=True, fancybox=True, shadow=True, loc='upper left', fontsize=12)
plt.tight_layout()
plt.savefig("/home/luke/QuickSig/examples/driver_error.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
