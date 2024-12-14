# %%
import jax.numpy as jnp
from craftax_marl.constants import *


# %%
player_position = jnp.array([
    [13, 15],
    [16, 18],
    [22, 30],
    [0, 0],
])
player_alive = jnp.array([True, False, True, True])
player_count = 4
player_index = 0

teammate_map = jnp.zeros(
    (player_count, *OBS_DIM, 2), dtype=jnp.int32
)
local_position = (
    -1 * player_position[player_index]
    + player_position
    + jnp.array([OBS_DIM[0], OBS_DIM[1]]) // 2
)

on_screen = jnp.logical_and(
    local_position >= 0, local_position < jnp.array([OBS_DIM[0], OBS_DIM[1]])
).all(axis=-1)


# %%
teammate_direction = jnp.zeros(
    (player_count, 8, 2)
)
direction_index_2d = jnp.where(
    local_position < 0, 1,
    jnp.where(local_position > jnp.array([OBS_DIM[0], OBS_DIM[1]]), 2, 0)
)
direction_index = direction_index_2d[:, :, 0]*3 + direction_index_2d[:, :, 1] - 1
teammate_direction = teammate_direction.at[
    jnp.arange(player_count)[:, None], direction_index, 0
].max(
    jnp.logical_and(
        jnp.logical_not(on_screen),
        player_alive
    )
)
teammate_direction = teammate_direction.at[
    jnp.arange(player_count)[:, None], direction_index, 1
].max(
    jnp.logical_and(
        jnp.logical_not(on_screen),
        jnp.logical_not(player_alive)
    )
)


# %%
import jax.numpy as jnp

x = jnp.array([
    [0,0,0],
    [0,0,0],
    [0,1,0],
])

# %%
