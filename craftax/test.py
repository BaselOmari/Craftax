# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,4,6"

import jax
import jax.numpy as jnp

rng = jax.random.PRNGKey(0)

# %%
from craftax_marl.envs.craftax_symbolic_env import CraftaxMARLSymbolicEnv as CraftaxEnv
import os

env = CraftaxEnv(CraftaxEnv.default_static_params())

# %%
obs, state = env.reset(rng)

# %%
actions = {name: i for i, name in enumerate(env.agents)}
obs, state, _, _, _, = env.step(rng, state, actions)
