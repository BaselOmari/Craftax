# %%
import jax
import jax.numpy as jnp

rng = jax.random.PRNGKey(0)

# %%
from craftax_marl.envs.craftax_pixels_env import StaticEnvParams
from craftax_marl.envs.craftax_pixels_env import CraftaxMARLPixelsEnv as CraftaxEnv

# %%
# Generate World
count_by_player_count = []
for player_count in range(1, 32+1):
    from craftax_marl.world_gen.world_gen import generate_world
    env = CraftaxEnv(StaticEnvParams(player_count=player_count))

    world_gen_jitted = jax.jit(generate_world, static_argnums=(2, ), backend='gpu')

    for _ in range(100):
        state = world_gen_jitted(rng, env.default_params, env.static_env_params)

    import time
    times = []
    for _ in range(1000):
        start_time = time.time()
        state = world_gen_jitted(rng, env.default_params, env.static_env_params)
        end_time = time.time()
        times.append(end_time - start_time)

    mean_time = jnp.mean(jnp.array(times))
    median_time = jnp.median(jnp.array(times))

    print(f"Player count: {player_count}")
    print(f"Mean time per run: {mean_time:.6f} seconds")
    print(f"Median time per run: {median_time:.6f} seconds")
    print()
    count_by_player_count.append((player_count, mean_time, median_time))

# %%
# Generate World
from craftax_marl.world_gen.world_gen import generate_world
from craftax_marl.game_logic import craftax_step
world_gen_jitted = jax.jit(generate_world, static_argnums=(2, ), backend='gpu')
step_jitted = jax.jit(craftax_step, static_argnums=(4, ))

step_time_by_player_count = []
for player_count in range(1, 32+1):
    env = CraftaxEnv(StaticEnvParams(player_count=player_count))
    state = world_gen_jitted(rng, env.default_params, env.static_env_params)
    actions = jnp.array([0] * player_count)

    for _ in range(100):
        state, _ = step_jitted(rng, state, actions, env.default_params, env.static_env_params)

    import time
    times = []
    for _ in range(1000):
        rng, _rng = jax.random.split(rng)
        start_time = time.time()
        state, _ = step_jitted(_rng, state, actions, env.default_params, env.static_env_params)
        end_time = time.time()
        times.append(end_time - start_time)

    mean_time = jnp.mean(jnp.array(times))
    median_time = jnp.median(jnp.array(times))

    print(f"Player count: {player_count}")
    print(f"Mean time per run: {mean_time:.6f} seconds")
    print(f"Median time per run: {median_time:.6f} seconds")
    print()
    step_time_by_player_count.append((player_count, mean_time, median_time))


# %%
# Generate World
from craftax.world_gen.world_gen import generate_world as generate_world_single
from craftax.game_logic import craftax_step as craftax_step_single
from craftax.envs.craftax_pixels_env import CraftaxPixelsEnv as CraftaxSingleEnv
world_gen_single_jitted = jax.jit(generate_world_single, static_argnums=(2, ))
step_single_jitted = jax.jit(craftax_step_single, static_argnums=(4, ))

env = CraftaxSingleEnv()
state = world_gen_single_jitted(rng, env.default_params, env.static_env_params)
action = 0

for _ in range(100):
    state, _ = step_single_jitted(rng, state, action, env.default_params, env.static_env_params)

import time
times = []
for _ in range(1000):
    rng, _rng = jax.random.split(rng)
    start_time = time.time()
    state, _ = step_single_jitted(_rng, state, action, env.default_params, env.static_env_params)
    end_time = time.time()
    times.append(end_time - start_time)

mean_time = jnp.mean(jnp.array(times))
median_time = jnp.median(jnp.array(times))

print(f"Mean time per run: {mean_time:.6f} seconds")
print(f"Median time per run: {median_time:.6f} seconds")
print()

