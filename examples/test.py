import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from quicksig.rdes.drivers import bm_driver
from quicksig.rdes.augmentations import non_overlapping_windower
from quicksig.signatures.compute_path_signature import compute_path_signature

brownian_motion = bm_driver(jax.random.key(0), 32, 1)
windows = non_overlapping_windower(brownian_motion, 8)

# Plot each segment with different colors
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#2C3E50']  # Added dark blue for whole path
for i, segment in enumerate(windows):
    # Create indices for this specific segment
    start_idx = int(segment.interval[0])
    end_idx = int(segment.interval[1])
    indices = range(start_idx, end_idx)
    # Plot the segment
    plt.plot(indices, segment.path[:, 0], color=colors[i], linewidth=2, 
             label=f'Segment {i+1}', alpha=0.8)
    
    # Mark segment boundaries
    if i < len(windows) - 1:
        plt.axvline(x=int(windows[i+1].interval[0]), color='red', linestyle='--', alpha=0.6)
    
    # Add segment annotation
    mid_idx = (start_idx + end_idx) // 2
    plt.annotate(f'Seg {i+1}', xy=(mid_idx, float(segment.path[mid_idx - start_idx, 0])), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7),
                fontsize=10, ha='center')

# Overlay whole path with distinctive style
plt.plot(range(brownian_motion.num_timesteps), brownian_motion.path[:, 0], color=colors[4], linewidth=3, 
         alpha=0.5, linestyle=':', label='Whole Path')

plt.title('Path Segments for Chen Identity Verification', fontsize=14, fontweight='bold')
plt.xlabel('Time Index', fontsize=12)
plt.ylabel('X₁ Component', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
print(windows)

concat_path = windows[0] + windows[1] + windows[2] + windows[3]
# print(bool(jnp.allclose(brownian_motion.path, concat_path.path)))
print(windows[0])
print(windows[1])
print(windows[2])
print(windows[3])


# print(brownian_motion == concat_path)
# print("Success. The concatenated path is equal to the original path.")