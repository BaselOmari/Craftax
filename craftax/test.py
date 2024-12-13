# %%
import jax
import jax.numpy as jnp

rng = jax.random.PRNGKey(0)

# %%
from craftax_marl.util.game_logic_utils import *
from craftax_marl.envs.craftax_pixels_env import CraftaxMARLPixelsEnv as CraftaxEnv

static_params = CraftaxEnv.default_static_params()
env = CraftaxEnv(static_params)
obs, state = env.reset(rng, env.default_params)


# %%
jitted_reset = jax.jit(env.reset)


# %%
obs, state = jitted_reset(rng, env.default_params)

# %%
from craftax_marl.util.maths_utils import get_all_players_distance_map
def spawn_mobs(state, rng, params, static_params):
    player_distance_map = get_all_players_distance_map(
        state.player_position, state.player_alive, static_params
    )
    grave_map = jnp.logical_or(
        state.map[state.player_level] == BlockType.GRAVE.value,
        jnp.logical_or(
            state.map[state.player_level] == BlockType.GRAVE2.value,
            state.map[state.player_level] == BlockType.GRAVE3.value,
        ),
    )

    floor_mob_spawn_chance = FLOOR_MOB_SPAWN_CHANCE * static_params.player_count
    monster_spawn_coeff = (
        1
        + (state.monsters_killed[state.player_level] < MONSTERS_KILLED_TO_CLEAR_LEVEL)
        * 2
    )  # Triple spawn rate if we are on an uncleared level

    monster_spawn_coeff *= jax.lax.select(
        is_fighting_boss(state, static_params),
        is_boss_spawn_wave(state, static_params) * 1000,
        1,
    )

    # Passive mobs
    can_spawn_passive_mob = (
        state.passive_mobs.mask[state.player_level].sum()
        < static_params.max_passive_mobs
    )

    rng, _rng = jax.random.split(rng)
    can_spawn_passive_mob = jnp.logical_and(
        can_spawn_passive_mob,
        jax.random.uniform(_rng) < floor_mob_spawn_chance[state.player_level, 0],
    )

    can_spawn_passive_mob = jnp.logical_and(
        can_spawn_passive_mob, jnp.logical_not(is_fighting_boss(state, static_params))
    )

    all_valid_blocks_map = jnp.logical_or(
        state.map[state.player_level] == BlockType.GRASS.value,
        jnp.logical_or(
            state.map[state.player_level] == BlockType.PATH.value,
            jnp.logical_or(
                state.map[state.player_level] == BlockType.FIRE_GRASS.value,
                state.map[state.player_level] == BlockType.ICE_GRASS.value,
            ),
        ),
    )
    new_passive_mob_type = FLOOR_MOB_MAPPING[state.player_level, MobType.PASSIVE.value]

    passive_mobs_can_spawn_map = all_valid_blocks_map

    passive_mobs_can_spawn_map = jnp.logical_and(
        passive_mobs_can_spawn_map, player_distance_map > 3
    )
    passive_mobs_can_spawn_map = jnp.logical_and(
        passive_mobs_can_spawn_map, player_distance_map < params.mob_despawn_distance
    )
    passive_mobs_can_spawn_map = jnp.logical_and(
        passive_mobs_can_spawn_map, jnp.logical_not(state.mob_map[state.player_level])
    )

    # To avoid spawning mobs ontop of dead players
    passive_mobs_can_spawn_map = passive_mobs_can_spawn_map.at[
        state.player_position[:, 0], state.player_position[:, 1]
    ].set(False)

    can_spawn_passive_mob = jnp.logical_and(
        can_spawn_passive_mob, passive_mobs_can_spawn_map.sum() > 0
    )

    rng, _rng = jax.random.split(rng)
    passive_mob_position = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        shape=(1,),
        p=jnp.reshape(passive_mobs_can_spawn_map, -1)
        / jnp.sum(passive_mobs_can_spawn_map),
    )
    passive_mob_position = jnp.array(
        [
            passive_mob_position // static_params.map_size[0],
            passive_mob_position % static_params.map_size[1],
        ]
    ).T.astype(jnp.int32)[0]

    new_passive_mob_index = jnp.argmax(
        jnp.logical_not(state.passive_mobs.mask[state.player_level])
    )

    new_passive_mob_position = jax.lax.select(
        can_spawn_passive_mob,
        passive_mob_position,
        state.passive_mobs.position[state.player_level, new_passive_mob_index],
    )

    new_passive_mob_health = jax.lax.select(
        can_spawn_passive_mob,
        MOB_TYPE_HEALTH_MAPPING[new_passive_mob_type, MobType.PASSIVE.value],
        state.passive_mobs.health[state.player_level, new_passive_mob_index],
    )

    new_passive_mob_mask = jax.lax.select(
        can_spawn_passive_mob,
        True,
        state.passive_mobs.mask[state.player_level, new_passive_mob_index],
    )

    passive_mobs = Mobs(
        position=state.passive_mobs.position.at[
            state.player_level, new_passive_mob_index
        ].set(new_passive_mob_position),
        health=state.passive_mobs.health.at[
            state.player_level, new_passive_mob_index
        ].set(new_passive_mob_health),
        mask=state.passive_mobs.mask.at[state.player_level, new_passive_mob_index].set(
            new_passive_mob_mask
        ),
        attack_cooldown=state.passive_mobs.attack_cooldown,
        type_id=state.passive_mobs.type_id.at[
            state.player_level, new_passive_mob_index
        ].set(new_passive_mob_type),
    )

    state = state.replace(
        passive_mobs=passive_mobs,
        mob_map=state.mob_map.at[
            state.player_level, new_passive_mob_position[0], new_passive_mob_position[1]
        ].set(
            jnp.logical_or(
                state.mob_map[
                    state.player_level,
                    new_passive_mob_position[0],
                    new_passive_mob_position[1],
                ],
                new_passive_mob_mask,
            )
        ),
    )

    # Monsters
    monsters_can_spawn_player_range_map = player_distance_map > 9
    monsters_can_spawn_player_range_map_boss = player_distance_map <= 6

    monsters_can_spawn_player_range_map = jax.lax.select(
        is_fighting_boss(state, static_params),
        monsters_can_spawn_player_range_map_boss,
        monsters_can_spawn_player_range_map,
    )

    # Melee mobs
    can_spawn_melee_mob = (
        state.melee_mobs.mask[state.player_level].sum() < static_params.max_melee_mobs * static_params.player_count
    )

    new_melee_mob_type = FLOOR_MOB_MAPPING[state.player_level, MobType.MELEE.value]
    new_melee_mob_type_boss = FLOOR_MOB_MAPPING[
        state.boss_progress, MobType.MELEE.value
    ]

    new_melee_mob_type = jax.lax.select(
        is_fighting_boss(state, static_params),
        new_melee_mob_type_boss,
        new_melee_mob_type,
    )

    rng, _rng = jax.random.split(rng)
    melee_mob_spawn_chance = floor_mob_spawn_chance[
        state.player_level, 1
    ] + floor_mob_spawn_chance[state.player_level, 3] * jnp.square(
        1 - state.light_level
    )
    can_spawn_melee_mob = jnp.logical_and(
        can_spawn_melee_mob,
        jax.random.uniform(_rng) < melee_mob_spawn_chance * monster_spawn_coeff,
    )

    melee_mobs_can_spawn_map = jax.lax.select(
        is_fighting_boss(state, static_params), grave_map, all_valid_blocks_map
    )

    melee_mobs_can_spawn_map = jnp.logical_and(
        melee_mobs_can_spawn_map, monsters_can_spawn_player_range_map
    )
    melee_mobs_can_spawn_map = jnp.logical_and(
        melee_mobs_can_spawn_map, player_distance_map < params.mob_despawn_distance
    )
    melee_mobs_can_spawn_map = jnp.logical_and(
        melee_mobs_can_spawn_map, jnp.logical_not(state.mob_map[state.player_level])
    )
    melee_mobs_can_spawn_map = melee_mobs_can_spawn_map.at[
        state.player_position[:, 0], state.player_position[:, 1]
    ].set(False)

    can_spawn_melee_mob = jnp.logical_and(
        can_spawn_melee_mob, melee_mobs_can_spawn_map.sum() > 0
    )

    rng, _rng = jax.random.split(rng)
    melee_mob_position = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        shape=(1,),
        p=jnp.reshape(melee_mobs_can_spawn_map, -1) / jnp.sum(melee_mobs_can_spawn_map),
    )
    melee_mob_position = jnp.array(
        [
            melee_mob_position // static_params.map_size[0],
            melee_mob_position % static_params.map_size[1],
        ]
    ).T.astype(jnp.int32)[0]

    new_melee_mob_index = jnp.argmax(
        jnp.logical_not(state.melee_mobs.mask[state.player_level])
    )

    new_melee_mob_position = jax.lax.select(
        can_spawn_melee_mob,
        melee_mob_position,
        state.melee_mobs.position[state.player_level, new_melee_mob_index],
    )

    new_melee_mob_health = jax.lax.select(
        can_spawn_melee_mob,
        MOB_TYPE_HEALTH_MAPPING[new_melee_mob_type, MobType.MELEE.value],
        state.melee_mobs.health[state.player_level, new_melee_mob_index],
    )

    new_melee_mob_mask = jax.lax.select(
        can_spawn_melee_mob,
        True,
        state.melee_mobs.mask[state.player_level, new_melee_mob_index],
    )

    melee_mobs = Mobs(
        position=state.melee_mobs.position.at[
            state.player_level, new_melee_mob_index
        ].set(new_melee_mob_position),
        health=state.melee_mobs.health.at[state.player_level, new_melee_mob_index].set(
            new_melee_mob_health
        ),
        mask=state.melee_mobs.mask.at[state.player_level, new_melee_mob_index].set(
            new_melee_mob_mask
        ),
        attack_cooldown=state.melee_mobs.attack_cooldown,
        type_id=state.melee_mobs.type_id.at[
            state.player_level, new_melee_mob_index
        ].set(new_melee_mob_type),
    )

    state = state.replace(
        melee_mobs=melee_mobs,
        mob_map=state.mob_map.at[
            state.player_level, new_melee_mob_position[0], new_melee_mob_position[1]
        ].set(
            jnp.logical_or(
                state.mob_map[
                    state.player_level,
                    new_melee_mob_position[0],
                    new_melee_mob_position[1],
                ],
                new_melee_mob_mask,
            )
        ),
    )

    # Ranged mobs
    can_spawn_ranged_mob = (
        state.ranged_mobs.mask[state.player_level].sum() < static_params.max_ranged_mobs
    )

    new_ranged_mob_type = FLOOR_MOB_MAPPING[state.player_level, MobType.RANGED.value]
    new_ranged_mob_type_boss = FLOOR_MOB_MAPPING[
        state.boss_progress, MobType.RANGED.value
    ]

    new_ranged_mob_type = jax.lax.select(
        is_fighting_boss(state, static_params),
        new_ranged_mob_type_boss,
        new_ranged_mob_type,
    )

    rng, _rng = jax.random.split(rng)
    can_spawn_ranged_mob = jnp.logical_and(
        can_spawn_ranged_mob,
        jax.random.uniform(_rng)
        < floor_mob_spawn_chance[state.player_level, 2] * monster_spawn_coeff,
    )

    # Hack for deep thing
    ranged_mobs_can_spawn_map = jax.lax.select(
        new_ranged_mob_type == 5,
        state.map[state.player_level] == BlockType.WATER.value,
        all_valid_blocks_map,
    )
    ranged_mobs_can_spawn_map = jax.lax.select(
        is_fighting_boss(state, static_params), grave_map, ranged_mobs_can_spawn_map
    )

    ranged_mobs_can_spawn_map = jnp.logical_and(
        ranged_mobs_can_spawn_map, monsters_can_spawn_player_range_map
    )
    ranged_mobs_can_spawn_map = jnp.logical_and(
        ranged_mobs_can_spawn_map, player_distance_map < params.mob_despawn_distance
    )
    ranged_mobs_can_spawn_map = jnp.logical_and(
        ranged_mobs_can_spawn_map, jnp.logical_not(state.mob_map[state.player_level])
    )
    ranged_mobs_can_spawn_map = ranged_mobs_can_spawn_map.at[
        state.player_position[:, 0], state.player_position[:, 1]
    ].set(False)

    can_spawn_ranged_mob = jnp.logical_and(
        can_spawn_ranged_mob, ranged_mobs_can_spawn_map.sum() > 0
    )

    rng, _rng = jax.random.split(rng)
    ranged_mob_position = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        shape=(1,),
        p=jnp.reshape(ranged_mobs_can_spawn_map, -1)
        / jnp.sum(ranged_mobs_can_spawn_map),
    )
    ranged_mob_position = jnp.array(
        [
            ranged_mob_position // static_params.map_size[0],
            ranged_mob_position % static_params.map_size[1],
        ]
    ).T.astype(jnp.int32)[0]

    new_ranged_mob_index = jnp.argmax(
        jnp.logical_not(state.ranged_mobs.mask[state.player_level])
    )

    new_ranged_mob_position = jax.lax.select(
        can_spawn_ranged_mob,
        ranged_mob_position,
        state.ranged_mobs.position[state.player_level, new_ranged_mob_index],
    )

    new_ranged_mob_health = jax.lax.select(
        can_spawn_ranged_mob,
        MOB_TYPE_HEALTH_MAPPING[new_ranged_mob_type, MobType.RANGED.value],
        state.ranged_mobs.health[state.player_level, new_ranged_mob_index],
    )

    new_ranged_mob_mask = jax.lax.select(
        can_spawn_ranged_mob,
        True,
        state.ranged_mobs.mask[state.player_level, new_ranged_mob_index],
    )

    ranged_mobs = Mobs(
        position=state.ranged_mobs.position.at[
            state.player_level, new_ranged_mob_index
        ].set(new_ranged_mob_position),
        health=state.ranged_mobs.health.at[
            state.player_level, new_ranged_mob_index
        ].set(new_ranged_mob_health),
        mask=state.ranged_mobs.mask.at[state.player_level, new_ranged_mob_index].set(
            new_ranged_mob_mask
        ),
        attack_cooldown=state.ranged_mobs.attack_cooldown,
        type_id=state.ranged_mobs.type_id.at[
            state.player_level, new_ranged_mob_index
        ].set(new_ranged_mob_type),
    )

    state = state.replace(
        ranged_mobs=ranged_mobs,
        mob_map=state.mob_map.at[
            state.player_level, new_ranged_mob_position[0], new_ranged_mob_position[1]
        ].set(
            jnp.logical_or(
                state.mob_map[
                    state.player_level,
                    new_ranged_mob_position[0],
                    new_ranged_mob_position[1],
                ],
                new_ranged_mob_mask,
            )
        ),
    )

    return state

state = spawn_mobs(state, rng, env.default_params, static_params)

# %%
jitted_spawn_mobs = jax.jit(spawn_mobs, static_argnums=(3,))

# %%
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    for _ in range(5):
        state = jitted_spawn_mobs(
            state,
            jnp.array([
                0,
                Action.SHOOT_ARROW.value,
                Action.SHOOT_ARROW.value,
                Action.SLEEP.value
            ]),
            env.static_env_params
        )
        print('hello')

# %%
import timeit
def run_jitted_spawn_mobs():
    jitted_spawn_mobs(
        state, rng, env.default_params, static_params
    )

# Time the execution over 1000 runs
execution_time = timeit.timeit(run_jitted_spawn_mobs, number=1000)
print(f"Execution time over 1000 runs: {execution_time:.4f} seconds")
print(f"Average time per run: {execution_time / 1000:.6f} seconds")


# %%
projectile_index = 0
projectiles = state.player_projectiles
projectile_owner = state.player_projectile_owners[
    state.player_level, projectile_index
]
projectile_type = state.player_projectiles.type_id[
    state.player_level, projectile_index
]
projectile_damage_vector = (
    MOB_TYPE_DAMAGE_MAPPING[projectile_type, MobType.PROJECTILE.value]
    * projectiles.mask[state.player_level, projectile_index]
)
is_arrow = jnp.logical_or(
    projectile_type == ProjectileType.ARROW.value,
    projectile_type == ProjectileType.ARROW2.value,
)
# Bow enchantment
arrow_damage_add = jnp.zeros(3, dtype=jnp.float32)
arrow_damage_add = arrow_damage_add.at[state.bow_enchantment[projectile_owner]].set(
    projectile_damage_vector[0] / 2
)
arrow_damage_add = arrow_damage_add.at[0].set(0)
projectile_damage_vector += jax.lax.select(
    is_arrow,
    arrow_damage_add,
    jnp.zeros(3, dtype=jnp.float32),
)
# Apply attribute scaling
arrow_damage_coeff = 1 + 0.2 * (state.player_dexterity[projectile_owner] - 1)
magic_damage_coeff = 1 + 0.5 * (state.player_intelligence[projectile_owner] - 1)
projectile_damage_vector *= jax.lax.select(
    is_arrow,
    arrow_damage_coeff,
    1.0,
)
projectile_damage_vector *= jax.lax.select(
    jnp.logical_or(
        projectile_type == ProjectileType.FIREBALL.value,
        projectile_type == ProjectileType.ICEBALL.value,
    ),
    magic_damage_coeff,
    1.0,
)
proposed_position = (
    projectiles.position[state.player_level, projectile_index]
    + state.player_projectile_directions[state.player_level, projectile_index]
)
        
proposed_position_in_bounds = in_bounds(proposed_position[None, :], static_params).item()
in_wall = is_in_solid_block(state.map[state.player_level], proposed_position[None, :]).item()
in_wall = jnp.logical_and(
    in_wall,
    jnp.logical_not(
        state.map[state.player_level][
            proposed_position[0], proposed_position[1]
        ]
        == BlockType.WATER.value
    ),
)  # Arrows can go over water
state, did_attack_mob0, did_kill_mob0 = attack_mob(
    state,
    jnp.array([True]),
    projectiles.position[None, state.player_level, projectile_index],
    projectile_damage_vector[None, :],
    jnp.array([False]),
)
did_attack_mob0 = did_attack_mob0.item()
projectile_damage_vector = projectile_damage_vector * (1 - did_attack_mob0)
state, did_attack_mob1, did_kill_mob1 = attack_mob(
    state,
    jnp.array([True]),
    proposed_position[None, :],
    projectile_damage_vector[None, :],
    jnp.array([False])
)
did_attack_mob1 = did_attack_mob1.item()
did_attack_mob = jnp.logical_or(did_attack_mob0, did_attack_mob1)
continue_move = jnp.logical_and(
    proposed_position_in_bounds, jnp.logical_not(in_wall)
)
continue_move = jnp.logical_and(continue_move, jnp.logical_not(did_attack_mob))
position = proposed_position
# Clear our old entry if we are alive
new_mask = jnp.logical_and(
    continue_move, projectiles.mask[state.player_level, projectile_index]
)
state = state.replace(
    player_projectiles=state.player_projectiles.replace(
        position=state.player_projectiles.position.at[
            state.player_level, projectile_index
        ].set(position),
        mask=state.player_projectiles.mask.at[
            state.player_level, projectile_index
        ].set(new_mask),
    ),
)

# %%
