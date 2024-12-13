# %%
from craftax_marl.constants import *
import numpy as np
import jax.numpy as jnp
from PIL import Image

# %%
player_count = 4
color_palette = (jnp.array(husl_palette(player_count, h=0.5, l=0.5)) * 255).astype(jnp.uint32)
color_palette = jnp.concatenate([color_palette, jnp.ones((player_count, 1))], axis=-1)
base_player_textures = TEXTURES[64]["player_textures"]

# %%
multiplayer_textures = base_player_textures[None, :].repeat(player_count, 0)
mask = (multiplayer_textures == jnp.array([0,0,0,1])).all(axis=-1)

# %%
# mask is of size (player_count, 6, 64, 64)
# create a separate array of size mask.shape which stores the value of the player index
player_indices = jnp.zeros_like(mask, dtype=jnp.int32)
player_indices = player_indices.at[jnp.arange(player_count), :, :, :].set(jnp.arange(player_count)[:, None, None, None])

# %%
multiplayer_textures = multiplayer_textures.at[mask].set(color_palette[player_indices[mask]])


# %%
for i in range(player_count):
    image = Image.fromarray(np.array(multiplayer_textures[i, 1].at[:, :, 3].mul(255), dtype=np.uint8))
    image.save(f'/home/balomari/Craftax/craftax/texture_{i}.png')

# %%

