# %%
from craftax_marl.constants import *
import jax.numpy as jnp
import jax
import numpy as np
from PIL import Image
from craftax_marl.envs.craftax_pixels_env import CraftaxMARLPixelsEnv as CraftaxEnv
from craftax_marl.world_gen.world_gen import generate_world

# %%
rng = jax.random.PRNGKey(0)
env = CraftaxEnv(CraftaxEnv.default_static_params())
state = generate_world(rng, env.default_params, env.static_env_params)
obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)
static_params = env.default_static_params()

# %%
chest_map_view = jnp.zeros((
    static_params.player_count,
    static_params.player_count,
    *OBS_DIM
))

player_level = 1

# %%
def _create_chest_map(player_index):
        local_position = (
            state.chest_positions[player_level]
            - state.player_position[player_index, None]
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        in_bounds = jnp.logical_and(local_position >= 0, local_position < obs_dim_array).all(axis=-1)
        is_still_chest = state.map[
            player_level, 
            state.chest_positions[player_level, :, :, 0], 
            state.chest_positions[player_level, :, :, 1]
        ] == BlockType.CHEST.value
        placed_chest = jnp.logical_and(in_bounds, is_still_chest)
        row_indices = jnp.arange(placed_chest.shape[0]).reshape(-1, 1)
        row_indices_array = jnp.tile(row_indices, (1, placed_chest.shape[1]))

        player_chest_map = jnp.zeros((static_params.player_count, *OBS_DIM))
        player_chest_map = player_chest_map.at[
            row_indices_array, 
            local_position[..., 0], 
            local_position[..., 1]
        ].max(placed_chest)
        return player_chest_map

chest_map_view = jax.vmap(_create_chest_map)(
    jnp.arange(static_params.player_count)
)

