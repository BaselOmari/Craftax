# %%
import os
import sys
sys.path.append('/app/Craftax/craftax')
sys.path.append('/app/Craftax')
os.environ["CUDA_VISIBLE_DEVICES"] = "7,"

import jax
import jax.numpy as jnp

from craftax_marl_basic.envs.craftax_symbolic_env import CraftaxMARLSymbolicEnv as CraftaxEnv 
from craftax_marl_basic.renderer.renderer_pixels import render_craftax_pixels
from craftax_marl_basic.constants import *

rng = jax.random.PRNGKey(0)

from train.ippo_rnn_basic import ActorCriticRNN, ScannedRNN, batchify, unbatchify
import pickle

save_dir = "/app/Craftax/train/saved_states/IPPO - Basic - Individual Rewards - 4 Agents"
state_name = "actor_state_960000000"
with open(f"{save_dir}/config.pkl", "rb") as f:
    config = pickle.load(f)
    config["NUM_ENVS"] = 2
    config["NUM_ACTORS"] = config["NUM_AGENTS"] * config["NUM_ENVS"]

env = CraftaxEnv(num_agents=config["NUM_AGENTS"],)

player_specific_textures = load_player_specific_textures(
    TEXTURES[BLOCK_PIXEL_SIZE_HUMAN],
    env.static_env_params.player_count
)

actor_network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
import os
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import flax.serialization


ac_init_x = (
    jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
    jnp.zeros((1, config["NUM_ENVS"])),
)
ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
params = actor_network.init(rng, ac_init_hstate, ac_init_x)

def linear_schedule(count):
    frac = (
        1.0
        - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
        / config["NUM_UPDATES"]
    )
    return config["LR"] * frac
actor_tx = optax.chain(
    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
    optax.adam(learning_rate=linear_schedule, eps=1e-5),
)
state = train_state.TrainState.create(apply_fn=actor_network.apply, params=params, tx=actor_tx)

with open(f"{save_dir}/{state_name}", "rb") as f:
    bytes_input = f.read()

# Use original structure as template
restored_state = flax.serialization.from_bytes(state, bytes_input)


# %%
# INITIALIZATION
rng, _rng = jax.random.split(rng)
reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
init_obsv, init_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

rng, _rng = jax.random.split(rng)
init_ac_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

init_done = jnp.zeros((config["NUM_ACTORS"]), dtype=bool)

# %%
def _step(runner_state, unused):
    model_state, env_state, last_obs, last_done, hstate, rng = runner_state
    obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
    ac_in = (
        obs_batch[None, :],
        last_done[None, :],
    )

    rng, _rng = jax.random.split(rng)
    ac_hstate, pi, value = actor_network.apply(model_state.params, hstate, ac_in)
    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)
    env_act = unbatchify(
        action, env.agents, config["NUM_ENVS"], env.num_agents
    )
    env_act = {k: v.squeeze() for k, v in env_act.items()}

    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state, reward, done, info = jax.vmap(
        env.step, in_axes=(0, 0, 0)
    )(rng_step, env_state, env_act)

    done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()

    runner_state = (model_state, env_state, obsv, done_batch, ac_hstate, rng)
    return runner_state, (env_state, done, reward, log_prob, info)

jitted_test_step = jax.jit(_step)


# %%
init_runner_state = (restored_state, init_env_state, init_obsv, init_done, init_ac_hstate, rng)

saved = []
runner_state = init_runner_state
num_steps = 500
for _ in range(num_steps):
    runner_state, res = jitted_test_step(runner_state, None)
    saved.append(res)

# %%
with open("saved_4agents_rollout.pkl", "wb") as f:
    pickle.dump(saved, f)
