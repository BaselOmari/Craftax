# %%
import os
import sys
sys.path.append('/app/Craftax/craftax')
sys.path.append('/app/Craftax')
os.environ["CUDA_VISIBLE_DEVICES"] = "3,"

import jax
import jax.numpy as jnp

from craftax_marl.envs.craftax_symbolic_env import CraftaxMARLSymbolicEnv as CraftaxEnv
from craftax_marl.renderer.renderer_pixels import render_craftax_pixels
from craftax_marl.constants import *


rng = jax.random.PRNGKey(0)
env = CraftaxEnv()

player_specific_textures = load_player_specific_textures(
    TEXTURES[BLOCK_PIXEL_SIZE_HUMAN],
    env.static_env_params.player_count
)


# %%
from train.mappo_rnn import ActorRNN, ScannedRNN, batchify, unbatchify, unbatchify_actions
import pickle

save_dir = "/app/Craftax/train/saved_states/mappo-revive-chest_removed-recovery++"
with open(f"{save_dir}/config.pkl", "rb") as f:
    config = pickle.load(f)
    config["NUM_ENVS"] = 4
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]


actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)


# %%
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

with open(f"{save_dir}/actor_state_20000", "rb") as f:
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
    ac_hstate, pi = actor_network.apply(model_state.params, hstate, ac_in)
    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)
    env_act = unbatchify_actions(
        action, env.agents, config["NUM_ENVS"], env.num_agents
    )

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
num_steps = 1000
for _ in range(num_steps):
    runner_state, res = jitted_test_step(runner_state, None)
    saved.append(res)

# %%
class CraftaxRenderer:
    def __init__(self, env: CraftaxEnv, env_params, pixel_render_size=64//BLOCK_PIXEL_SIZE_HUMAN):
        self.env = env
        self.env_params = env_params
        self.pixel_render_size = pixel_render_size
        self.pygame_events = []

        self.screen_size = (
            OBS_DIM[1] * BLOCK_PIXEL_SIZE_HUMAN * pixel_render_size,
            (2 + OBS_DIM[0] + INVENTORY_OBS_HEIGHT)
            * BLOCK_PIXEL_SIZE_HUMAN
            * pixel_render_size,
        )

        # Init rendering
        pygame.init()
        pygame.key.set_repeat(250, 75)

        self.screen_surface = pygame.display.set_mode(self.screen_size)

        self._render = render_craftax_pixels

    def update(self):
        # Update pygame events
        self.pygame_events = list(pygame.event.get())

        # Update screen
        pygame.display.flip()

    def is_quit_requested(self):
        for event in self.pygame_events:
            if event.type == pygame.QUIT:
                return True
        return False
pixels_obsv = jax.vmap(render_craftax_pixels, in_axes=(0, None, None, None))(
    env_state,
    BLOCK_PIXEL_SIZE_HUMAN,
    env.static_env_params,
    player_specific_textures
)/255.0
