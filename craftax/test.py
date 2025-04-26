# %%
import os
import sys
sys.path.append('/app/Craftax/craftax')
os.environ["CUDA_VISIBLE_DEVICES"] = "6,"

import jax
import jax.numpy as jnp

from jaxmarl.wrappers.baselines import (
    LogWrapper,
)

rng = jax.random.PRNGKey(0)

# %%
from craftax_marl.envs.craftax_symbolic_env import CraftaxMARLSymbolicEnv as CraftaxEnv
import os

env = CraftaxEnv()

# %%

# %%
from jaxmarl.wrappers.baselines import (
    CTRolloutManager,
)
env_num = 32
wrapped_env = CTRolloutManager(env, batch_size=env_num, preprocess_obs=True)

# %%
obs, env_state = wrapped_env.batch_reset(rng)

_rngs = jax.random.split(rng, env.num_agents)
new_action = {
    agent: wrapped_env.batch_sample(_rngs[i], agent)
    for i, agent in enumerate(env.agents)
}
new_obs, new_env_state, reward, new_done, info = wrapped_env.batch_step(
    rng, env_state, new_action
)

# %%
avail_actions = jax.vmap(wrapped_env.get_avail_actions)(env_state)


# %%