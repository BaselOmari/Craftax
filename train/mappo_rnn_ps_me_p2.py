"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.
"""

import os
import sys
sys.path.append('/app/Craftax/craftax')
os.environ["CUDA_VISIBLE_DEVICES"] = "5,"

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict
import chex

import flax
from flax.training.train_state import TrainState
import distrax
from functools import partial
import jaxmarl
from jaxmarl.wrappers.baselines import JaxMARLWrapper

import wandb
import functools

from jaxmarl.wrappers.baselines import (
    LogWrapper,
)
from craftax_marl.envs.craftax_symbolic_env import CraftaxMARLSymbolicEnv as CraftaxEnv
import pickle

    
class WorldStateWrapper(JaxMARLWrapper):
    
    @partial(jax.jit, static_argnums=0)
    def reset(self,
              key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state
    
    @partial(jax.jit, static_argnums=0)
    def step(self,
             key,
             state,
             action):
        obs, env_state, reward, done, info = self._env.step(
            key, state, action
        )
        obs["world_state"] = self.world_state(obs)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def world_state(self, obs):
        """ 
        For each agent: [agent obs, all other agent obs]
        """
        
        @partial(jax.vmap, in_axes=(0, None))
        def _roll_obs(aidx, all_obs):
            robs = jnp.roll(all_obs, -aidx, axis=0)
            robs = robs.flatten()
            return robs
            
        all_obs = jnp.array([obs[agent] for agent in self._env.agents]).flatten()
        all_obs = jnp.expand_dims(all_obs, axis=0).repeat(self._env.num_agents, axis=0)
        return all_obs
    
    def world_state_size(self):
        spaces = [self._env.observation_space(agent) for agent in self._env.agents]
        return sum([space.shape[-1] for space in spaces])

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi


class CriticRNN(nn.Module):
    config: Dict
    
    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        embedding = nn.relu(embedding)
        
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        critic = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        
        return hidden, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def batchify_per_agent(x: dict, agent_list):
    x = jnp.stack([x[a] for a in agent_list])
    return x


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def unbatchify_actions(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config, env):
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]

    env = WorldStateWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticRNN(config=config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        ac_init_hstate = jax.vmap(
            lambda _: ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        )(jnp.arange(env.num_agents))
        actor_network_params = jax.vmap(
            lambda _: actor_network.init(_rng_actor, ac_init_hstate[0], ac_init_x)
        )(jnp.arange(env.num_agents))
        
        cr_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.world_state_size(),)),  #  + env.observation_space(env.agents[0]).shape[0]
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)
        
        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        actor_train_state = jax.vmap(
            lambda params: TrainState.create(
            apply_fn=actor_network.apply,
            params=params,
            tx=actor_tx,
            )
        )(actor_network_params)
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]).reshape(env.num_agents, config["NUM_ENVS"], -1)
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]).reshape(env.num_agents, config["NUM_ENVS"], -1)

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            
            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, hstates, rng = runner_state

                """
                last_obs = dict with element for each agent and each size (num_env, obsv_size)
                last_done = boolean jnp.ndarray with size (num_env*num_agents, ) -> agent_1_env_1, agent_1_env_2, agent_2_env_1, agent_2_env_2, ...

                train_states = (
                    actor_train_state: TrainState,  # contains params and optimizer state for actor network
                    critic_train_state: TrainState,  # contains params and optimizer state for critic network
                ) # TrainState.params is a dict with size = (N, ...) where N = num_agents
                hstates = (
                    ac_init_hstate: jnp.ndarray,  # (num_agents*num_envs, gru_hidden_dim)
                    cr_init_hstate: jnp.ndarray,  # (num_agents*num_envs, gru_hidden_dim)
                )

                env_state: jnp.ndarray,  # (num_envs, env_state_size)
                """

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                _rngs = jax.random.split(_rng, env.num_agents)
                obs_batch = batchify_per_agent(
                    x=last_obs,
                    agent_list=env.agents,
                ) # (num_agents*num_envs, obs_dim) -> (num_agents, num_envs, obs_dim)
                done_agent_batch = last_done.reshape(
                    (env.num_agents, config["NUM_ENVS"])
                ) # (num_agents*num_envs,) -> (num_agents, num_envs)

                ac_hstates = hstates[0].reshape(
                    (env.num_agents, config["NUM_ENVS"], -1)
                )  # (num_agents*num_envs, gru_hidden_dim) -> (num_agents, num_envs, gru_hidden_dim)
                
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

                # VALUE
                # output of wrapper is (num_envs, num_agents, world_state_size)
                # swap axes to (num_agents, num_envs, world_state_size) before reshaping to (num_actors, world_state_size)
                world_state = last_obs["world_state"].swapaxes(0,1)
                world_state = world_state.reshape((config["NUM_ACTORS"],-1))
                cr_in = (
                    world_state[None, :],
                    last_done[np.newaxis, :],
                )
                cr_hstate, value = critic_network.apply(train_states[1].params, hstates[1], cr_in)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_actions)
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    actions_per_agent,
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_probs_per_agent,
                    obs_batch,
                    world_state,
                    info,
                )
                runner_state = (train_states, env_state, obsv, done_batch, (ac_hstates_per_agent, cr_hstate), rng)
                return runner_state, transition

            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, hstates, rng = runner_state
      
            last_world_state = last_obs["world_state"].swapaxes(0,1)  
            last_world_state = last_world_state.reshape((config["NUM_ACTORS"],-1))
            cr_in = (
                last_world_state[None, :],
                last_done[np.newaxis, :],
            )
            _, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)
            advantages_per_agent = advantages.reshape(
                (env.num_agents, config["NUM_ENVS"])
            )
            targets_per_agent = targets.reshape(
                (env.num_agents, config["NUM_ENVS"])
            )

            # UPDATE NETWORK
            # ------------------------------------------------------------------ #
            #  E P O C H   U P D A T E   – one full pass over collected rollout  #
            # ------------------------------------------------------------------ #
            def _update_epoch(update_state, _):
                """
                update_state =
                    (train_states,            # (actor_states [A], critic_state)
                     init_hstates,            # (ac_h[AxE,H], cr_h[AxE,H])
                     traj_batch,              # Transition
                     advantages,              # (T, A·E)
                     targets,                 # (T, A·E)
                     rng)
                """
                (actor_state, critic_state), (ac_h0, cr_h0), traj, adv, tgt, rng = update_state
                A, E          = env.num_agents, config["NUM_ENVS"]
                T             = traj.action.shape[0]          # rollout length
                H             = ac_h0.shape[-1]

                # ---------- reshape helpers (actor‑axis first) -----------------
                def _ae(x, trailing=()):
                    """(T, A·E, …) ↦ (A, T, E, …)"""
                    x = x.reshape(T, A, E, *trailing).transpose(1, 0, 2, *range(3, x.ndim+1))
                    return x                                                 # (A, T, E, …)

                obs_a     = _ae(traj.obs,        trailing=(traj.obs.shape[-1],))   # (A,T,E,obs)
                done_a    = _ae(traj.done)                                         # (A,T,E)
                act_a     = _ae(traj.action)                                       # (A,T,E)
                old_lp_a  = _ae(traj.log_prob)                                     # (A,T,E)
                gae_a     = _ae(adv)                                               # (A,T,E)
                ac_h0_a   = ac_h0.reshape(A, E, H)                                 # (A,E,H)

                # ---------- actor update ---------------------------------------
                def _actor_update(state, h0, obs, done, act, old_lp, gae):
                    """
                    Args are for ONE agent (E envs, T timesteps)
                    Shapes:
                        h0      : (E,H)
                        obs     : (T,E,obs_dim)
                        done    : (T,E)
                        act     : (T,E)
                        old_lp  : (T,E)
                        gae     : (T,E)
                    """
                    def loss_fn(params):
                        _, policy = actor_network.apply(params, h0, (obs, done))
                        logp      = policy.log_prob(act)
                        ratio     = jnp.exp(logp - old_lp)
                        gae_n     = (gae - gae.mean()) / (gae.std() + 1e-8)

                        unclipped =  ratio        * gae_n
                        clipped   =  jnp.clip(ratio,
                                                1.0 - config["CLIP_EPS"],
                                                1.0 + config["CLIP_EPS"]) * gae_n
                        pg_loss   = -jnp.minimum(unclipped, clipped).mean()
                        entropy   =  policy.entropy().mean()

                        return pg_loss - config["ENT_COEF"] * entropy

                    grads = jax.grad(loss_fn)(state.params)
                    return state.apply_gradients(grads=grads)

                actor_state = jax.vmap(
                    _actor_update,
                    in_axes=(0, 0, 0, 0, 0, 0, 0),   # map over agent axis
                )(actor_state, ac_h0_a, obs_a, done_a, act_a, old_lp_a, gae_a)

                # ---------- critic update (central, single set of params) ------
                # world_state, value & targets already carry the (T, A·E, …) shape
                _, v_pred = critic_network.apply(
                    critic_state.params, cr_h0, (traj.world_state, traj.done)
                )

                v_pred_clipped = traj.value + (
                    v_pred - traj.value
                ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])

                v_loss = jnp.maximum((v_pred - tgt) ** 2,
                                     (v_pred_clipped - tgt) ** 2).mean() * 0.5

                def critic_loss_fn(params):
                    _, v = critic_network.apply(params, cr_h0,
                                                (traj.world_state, traj.done))
                    v_clip = traj.value + (v - traj.value).clip(-config["CLIP_EPS"],
                                                                config["CLIP_EPS"])
                    loss = jnp.maximum((v - tgt) ** 2,
                                       (v_clip - tgt) ** 2).mean() * 0.5
                    return config["VF_COEF"] * loss

                critic_grads   = jax.grad(critic_loss_fn)(critic_state.params)
                critic_state   = critic_state.apply_gradients(grads=critic_grads)

                # ---------- pack & return --------------------------------------
                new_update_state = (
                    (actor_state, critic_state),
                    (ac_h0, cr_h0),          # unchanged across inner epochs
                    traj, adv, tgt,
                    rng,
                )
                return new_update_state, None  # second value unused

            # ------------------------------------------------------------------ #

            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            
            train_states = update_state[0]
            metric = traj_batch.info
            metric["update_steps"] = update_steps
            rng = update_state[-1]

            def callback(metric, actor_state: TrainState, step):
                env_step = (
                    metric["update_steps"]
                    * config["NUM_ENVS"]
                    * config["NUM_STEPS"]
                )
                to_log = {
                    "env_step": env_step,
                }
                
                if metric["returned_episode"].any():
                    to_log.update(jax.tree.map(
                        lambda x: x[metric["returned_episode"]].mean(),
                        metric["user_info"]
                    ))
                    to_log["episode_lengths"] = metric["returned_episode_lengths"][metric["returned_episode"]].mean()
                    to_log["episode_returns"] = metric["returned_episode_returns"][metric["returned_episode"]].mean()
                
                if config["SAVE_DURING_TRAINING"]:
                    save_dir = f"/app/Craftax/train/saved_states/{config['RUN_NAME']}"
                    os.makedirs(save_dir, exist_ok=True)
                    if step == 0:
                        with open(f"{save_dir}/config.pkl", "wb+") as f:
                            pickle.dump(config, f)
                    if (step < 2500 and step % (config["SAVE_INTERVAL"]//10) == 0) or (step % config["SAVE_INTERVAL"])==0:
                        state_bytes = flax.serialization.to_bytes(actor_state)
                        with open(f"{save_dir}/actor_state_{env_step}", "wb+") as f:
                            f.write(state_bytes)
                
                print(to_log)
                wandb.log(to_log)
                            
            jax.experimental.io_callback(callback, None, metric, train_states[0], update_steps)
            update_steps = update_steps + 1
            runner_state = (train_states, env_state, last_obs, last_done, hstates, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train

def single_run(config):
    alg_name = config.get("ALG_NAME", "mappo-rnn")
    env = CraftaxEnv()
    env_name = "craftax-ma-symbolic"

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=config["RUN_NAME"],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))

    
# %%
if __name__ == "__main__":
    config = {
        "WANDB_MODE": "offline",
        "PROJECT": "pqn-vdn-rnn_craftax-ma-3-agents",
        "RUN_NAME": "MAPPO - Base - No Parameter Sharing",
        "ENTITY": "b2alomar-university-of-waterloo",

        "ALG_NAME": "mappo-rnn",
        "TOTAL_TIMESTEPS": 1e9,
        "NUM_ENVS": 512,
        "NUM_STEPS": 64,
        "NUM_MINIBATCHES": 8,
        "UPDATE_EPOCHS": 4,  # <-- renamed from NUM_EPOCHS
        "GRU_HIDDEN_DIM": 512,  # <-- renamed from HIDDEN_SIZE
        "FC_DIM_SIZE": 128,     # <-- inferred from usage in ActorCriticRNN
        "ACTIVATION": "tanh",
        "GAE_LAMBDA": 0.8,  # <-- renamed from LAMBDA
        "GAMMA": 0.99,
        "CLIP_EPS": 0.2,
        "SCALE_CLIP_EPS": False,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,

        "ANNEAL_LR": True,  # <-- renamed from LR_LINEAR_DECAY
        "LR": 2e-4,
        "MAX_GRAD_NORM": 1.0,
        "LR_WARMUP": 0.0,  # <-- added for learning rate schedule
        "REW_SHAPING_HORIZON": 1e6,

        # env specific
        "ENV_NAME": "Craftax-Symbolic-v1",
        "USE_OPTIMISTIC_RESETS": True,
        "OPTIMISTIC_RESET_RATIO": 16,
        "LOG_ACHIEVEMENTS": False,

        # evaluation
        "SAVE_DURING_TRAINING": True,
        "SAVE_INTERVAL": 2500,

        "NUM_SEEDS": 1,
        "SEED": 0,
    }
    single_run(config)

