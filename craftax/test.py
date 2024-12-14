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

# %%
player_level = 1
player_index = 0
local_position = (
    state.chest_positions[state.player_level, player_index]
    - state.player_position[:, None]
    + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
)


def _add_chests(chest_map_view, player_index):
    """Add players chest locations to other player maps"""
    local_position = (
        state.chest_positions[state.player_level, player_index]
        - state.player_position[:, None]
        + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
    )
    placed_block = jnp.logical_and(
        state.map[
            state.chest_positions[state.player_level, player_index, :, 0],
            state.chest_positions[state.player_level, player_index, :, 1],
        ] == BlockType.CHEST.value,
        in_obs_bounds(state.chest_positions[state.player_level, player_index, :])
    )
    return chest_map_view, None

chest_map_view = jnp.zeros((
    static_params.player_count,
    static_params.player_count,
    *OBS_DIM
))
chest_map_view, _ = jax.lax.scan(
    _add_chests,
    chest_map_view,
    jnp.arange(static_params.player_count)
)
