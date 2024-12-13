# %%
import jax
import jax.numpy as jnp

rng = jax.random.PRNGKey(0)

# %%
from craftax_marl.envs.craftax_pixels_env import StaticEnvParams
from craftax_marl.envs.craftax_pixels_env import CraftaxMARLPixelsEnv as CraftaxEnv

# Step
from craftax_marl.world_gen.world_gen import generate_world
from craftax_marl.game_logic import craftax_step
world_gen_jitted = jax.jit(generate_world, static_argnums=(2, ), backend='gpu')
step_jitted = jax.jit(craftax_step, static_argnums=(4, ))


player_count = 4
env = CraftaxEnv(StaticEnvParams(player_count=player_count))
actions = jnp.array([0] * player_count)

state = world_gen_jitted(rng, env.default_params, env.static_env_params)
for _ in range(100):
    state, _ = step_jitted(rng, state, actions, env.default_params, env.static_env_params)


# %%
import time
runs = 1000
start = time.time()
jax.lax.fori_loop(0, runs, lambda i, state: step_jitted(rng, state, actions, env.default_params, env.static_env_params)[0], state)
end = time.time()
print("Time per run:", (end - start) / runs, "seconds")

# %%
