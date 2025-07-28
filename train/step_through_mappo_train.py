# %%
import os
import sys
sys.path.append('/app/Craftax/craftax')
sys.path.append('/app/Craftax')
os.environ["CUDA_VISIBLE_DEVICES"] = "5,"

import jax
import jax.numpy as jnp

from craftax_marl_basic.envs.craftax_symbolic_env import CraftaxMARLSymbolicEnv as CraftaxEnv
from craftax_marl_basic.renderer.renderer_pixels import render_craftax_pixels
from craftax_marl_basic.constants import *


rng = jax.random.PRNGKey(0)
env = CraftaxEnv()

# %%
from train.mappo_rnn_basic import ActorRNN, ScannedRNN, batchify, unbatchify_actions
import pickle

save_dir = "/app/Craftax/train/saved_states/IPPO - Basic - Individual Rewards - 2 Agents"
state_name = "actor_state_960000000"
with open(f"{save_dir}/config.pkl", "rb") as f:
    config = pickle.load(f)
    config["NUM_ENVS"] = 26
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]

actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)


# %%
import os
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.training.train_state import TrainState
import flax.serialization

rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
ac_init_x = (
    jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
    jnp.zeros((1, config["NUM_ENVS"])),
)
ac_init_hstate = jax.vmap(
    lambda _: ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
)(jnp.arange(env.num_agents))
actor_network_params = jax.vmap(
    lambda rng, hstate: actor_network.init(rng, hstate, ac_init_x),
    in_axes=(0, 0)
)(
    jax.random.split(_rng_actor, env.num_agents), 
    ac_init_hstate
)

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

# %%
actor_train_state = jax.vmap(
    lambda params: TrainState.create(
        apply_fn=actor_network.apply,
        params=params,
        tx=actor_tx,
    )
)(actor_network_params)

# %%
rng, _rng = jax.random.split(rng)
reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]).reshape(env.num_agents, config["NUM_ENVS"], -1)
cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]).reshape(env.num_agents, config["NUM_ENVS"], -1)

hstates = (ac_init_hstate, cr_init_hstate)

# %%
import numpy as np
def batchify_per_agent(x: dict, agent_list):
    x = jnp.stack([x[a] for a in agent_list])
    return x


# %%
train_states = (actor_train_state, None)
env_state = env_state
last_obs = obsv
last_done = jnp.zeros((config["NUM_ACTORS"],))
hstates = (ac_init_hstate, cr_init_hstate)
rng = jax.random.PRNGKey(0)


# %%
rng, _rng = jax.random.split(rng)
_rngs = jax.random.split(_rng, env.num_agents)
obs_batch = batchify_per_agent(
    x=obsv,
    agent_list=env.agents,
) # (num_agents*num_envs, obs_dim) -> (num_agents, num_envs, obs_dim)

done_agent_batch = last_done.reshape(
    (env.num_agents, config["NUM_ENVS"])
) # (num_agents*num_envs,) -> (num_agents, num_envs)

ac_hstates = hstates[0].reshape(
    (env.num_agents, config["NUM_ENVS"], -1)
)  # (num_agents*num_envs, gru_hidden_dim) -> (num_agents, num_envs, gru_hidden_dim)

# %%
def apply_action_per_agent(rng, train_state, obs, done, ac_hstate):
    ac_in = (
        obs[np.newaxis, :],
        done[np.newaxis, :],
    )
    ac_hstate, pi = actor_network.apply(
            train_state.params, 
            ac_hstate, 
            ac_in
    )
    action = pi.sample(seed=rng)
    log_prob = pi.log_prob(action)
    return (action.squeeze(), log_prob.squeeze(), ac_hstate) 

actions_per_agent, log_probs_per_agent, ac_hstates_per_agent = jax.vmap(
    apply_action_per_agent,
    in_axes=(0, 0, 0, 0, 0),
    out_axes=0,
)(
    _rngs, 
    train_states[0], 
    obs_batch, 
    done_agent_batch, 
    ac_hstates,
)

env_actions = {agent: actions_per_agent[i] for i, agent in enumerate(env.agents)}

# %%
