# %%
import os
import sys
sys.path.append('/app/Craftax/craftax')
sys.path.append('/app/Craftax')
os.environ["CUDA_VISIBLE_DEVICES"] = "7,"

import pickle

with open("/app/Craftax/train/saved_1agents_rollout.pkl", "rb") as f:
    saved_1agents_rollout = pickle.load(f)


# %%
import numpy as np
import matplotlib.pyplot as plt

# Initialize a 48x48 grid to store the frequency of visits
heatmap = np.zeros((48, 48), dtype=int)

# Loop through the rollout and populate the heatmap
for i, step in enumerate(saved_1agents_rollout):
    state = step[0]
    position = state.player_position[0][0]  # Assuming position is [x, y]
    x, y = position
    if 0 <= x < 48 and 0 <= y < 48:
        heatmap[y, x] += 1  # y as row, x as column

# Plot the heatmap
plt.figure(figsize=(8, 8))
plt.imshow(heatmap, cmap='hot', interpolation='nearest', origin='lower')
plt.colorbar(label='Visit Frequency')
plt.title("Player Position Heatmap")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(False)
plt.show()



# %%
