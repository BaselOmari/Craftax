# # %%
# import pufferlib.emulation
# import pufferlib.wrappers

# import nmmo

# def nmmo_creator():
#     config = nmmo.config.Medium()
#     # config.PLAYER_N = 3
#     env = nmmo.Env(config)
#     env = pufferlib.wrappers.PettingZooTruncatedWrapper(env)
#     return pufferlib.emulation.PettingZooPufferEnv(env=env)

# env = nmmo_creator()
# obs, _ = env.reset()
# structured_obs = obs[1].view(env.obs_dtype)
# print('NMMO observation space:', structured_obs.dtype)
# print('Packed shape:', obs[1].shape)

# # %%
# import time

# def all_done(dones):
#     for b in dones.values():
#         if not b:
#             return False
#     return True

# env.reset()
# start_time = time.time()
# for i in range(1000):
#     actions = {a: env.action_space(a).sample() for a in env.agents}
#     obs, rewards, dones, truncs, infos = env.step(actions)
#     if all_done(dones):
#         env.reset()
# end_time = time.time()

# print(f"Time taken for 1000 iterations: {end_time - start_time:.2f} seconds")

# # %%
# import pufferlib.vector
# from pufferlib.ocean import NMMO3
# vecenv = pufferlib.vector.make(
#     NMMO3,
#     env_kwargs={
#         'num_envs': 64,
#         "width": 512,
#         "height": 512,
#     },
# )

# # %%
# obs, _ = vecenv.reset()

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def orthogonal_init(gain=1.0):
    def init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    return init


class ScannedRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def initialize_carry(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim)

    def forward(self, carry, inputs):
        # inputs: (seq_len, batch, feature), resets: (seq_len, batch)
        ins, resets = inputs  # ins: (T, B, D), resets: (T, B)
        outputs = []
        for t in range(ins.size(0)):
            reset_t = resets[t].unsqueeze(-1).to(dtype=carry.dtype)
            carry = torch.where(
                reset_t.bool(),
                self.initialize_carry(carry.size(0)).to(carry.device),
                carry,
            )
            carry = self.gru_cell(ins[t], carry)
            outputs.append(carry)
        return carry, torch.stack(outputs, dim=0)


class ActorCriticRNN(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, config["FC_DIM_SIZE"])
        self.rnn = ScannedRNN(config["FC_DIM_SIZE"], config["GRU_HIDDEN_DIM"])
        self.actor_fc1 = nn.Linear(config["GRU_HIDDEN_DIM"], config["GRU_HIDDEN_DIM"])
        self.actor_fc2 = nn.Linear(config["GRU_HIDDEN_DIM"], action_dim)
        self.critic_fc1 = nn.Linear(config["GRU_HIDDEN_DIM"], config["FC_DIM_SIZE"])
        self.critic_fc2 = nn.Linear(config["FC_DIM_SIZE"], 1)

        # Initialize weights
        orthogonal_init(gain=torch.sqrt(torch.tensor(2.0)))(self.fc1)
        orthogonal_init(gain=2.0)(self.actor_fc1)
        orthogonal_init(gain=0.01)(self.actor_fc2)
        orthogonal_init(gain=2.0)(self.critic_fc1)
        orthogonal_init(gain=1.0)(self.critic_fc2)

    def forward(self, hidden, x):
        obs, dones = x  # obs: (T, B, D), dones: (T, B)
        T, B, _ = obs.shape

        x = F.relu(self.fc1(obs))  # (T, B, FC_DIM_SIZE)

        rnn_in = (x, dones)
        hidden, embedding = self.rnn(hidden, rnn_in)  # embedding: (T, B, GRU_HIDDEN_DIM)

        actor = F.relu(self.actor_fc1(embedding))
        action_logits = self.actor_fc2(actor)

        pi = Categorical(logits=action_logits)

        critic = F.relu(self.critic_fc1(embedding))
        value = self.critic_fc2(critic).squeeze(-1)

        return hidden, pi, value


# %%  
import pufferlib.vector
from pufferlib.ocean import NMMO3
num_envs = 128
vecenv = pufferlib.vector.make(
    NMMO3,
    env_kwargs={
        "num_players": 3,
        'num_envs': num_envs,
        "width": 512,
        "height": 512,
    },
)
obs_og, _ = vecenv.reset()
obs_og = torch.as_tensor(obs_og)

config = {
    "FC_DIM_SIZE": 128,
    "GRU_HIDDEN_DIM": 512,
}
policy = ActorCriticRNN(vecenv.single_observation_space.shape[0], vecenv.single_action_space.n, config)


# %%
import time
import torch.optim as optim

# Hyperparameters
TOTAL_TIMESTEPS = 1_000_000
ROLLOUT_LENGTH = 64
GAMMA = 0.99
LR = 2.5e-4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
device = torch.device(1)

# Move policy to device
policy = policy.to(device)

optimizer = optim.Adam(policy.parameters(), lr=LR)

obs = torch.zeros((ROLLOUT_LENGTH, obs_og.shape[0]) + vecenv.single_observation_space.shape).to(device)
actions = torch.zeros((ROLLOUT_LENGTH, obs_og.shape[0]) + vecenv.single_action_space.shape).to(device)
logprobs = torch.zeros((ROLLOUT_LENGTH, obs_og.shape[0])).to(device)
rewards = torch.zeros((ROLLOUT_LENGTH, obs_og.shape[0])).to(device)
dones = torch.zeros((ROLLOUT_LENGTH, obs_og.shape[0])).to(device)
values = torch.zeros((ROLLOUT_LENGTH, obs_og.shape[0])).to(device)


# %%
import numpy as np

# Initial state
global_step = 0
next_obs, _ = vecenv.reset()
next_obs = torch.as_tensor(next_obs, dtype=torch.float32).to(device)
next_obs = next_obs.unsqueeze(0)  # Shape: (1, num_envs, obs_dim)
next_done = torch.zeros(obs_og.shape[0]).to(device)
hidden_state = policy.rnn.initialize_carry(obs_og.shape[0]).to(device)

for iteration in range(TOTAL_TIMESTEPS // (num_envs * ROLLOUT_LENGTH)):
    start_time = time.time()
    for step in range(ROLLOUT_LENGTH):
        global_step += num_envs
        obs[step] = next_obs.squeeze(0)
        dones[step] = next_done

        with torch.no_grad():
            hidden_state, pi, value = policy(hidden_state, (next_obs, next_done.unsqueeze(0)))
            action = pi.sample()
            logprob = pi.log_prob(action)

        actions[step] = action
        logprobs[step] = logprob
        values[step] = value.squeeze(0)

        next_obs_raw, reward, terminations, truncations, infos = vecenv.step(action.cpu().numpy())
        next_done = np.logical_or(terminations, truncations)
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs = torch.as_tensor(next_obs_raw, dtype=torch.float32).to(device).unsqueeze(0)
        next_done = torch.as_tensor(next_done, dtype=torch.float32).to(device)

    with torch.no_grad():
        next_value = policy(hidden_state, (next_obs, next_done.unsqueeze(0)))[2].reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(ROLLOUT_LENGTH)):
            if t == ROLLOUT_LENGTH - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + GAMMA * 0.95 * nextnonterminal * lastgaelam
        returns = advantages + values

    # Flatten batch
    b_obs = obs.reshape((-1,) + vecenv.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # PPO update
    
    b_inds = np.arange(num_envs)
    minibatch_size = num_envs//4
    for epoch in range(4):
        np.random.shuffle(b_inds)
        for start in range(0, num_envs, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            mb_obs = b_obs[mb_inds].unsqueeze(0)  # (1, batch, obs_dim)
            _, pi, newvalue = policy(
                policy.rnn.initialize_carry(len(mb_inds)).to(device),
                (mb_obs, torch.zeros(1, len(mb_inds)).to(device))
            )
            newlogprob = pi.log_prob(b_actions[mb_inds])
            entropy = pi.entropy().mean()

            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            mb_advantages = b_advantages[mb_inds]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            pg_loss = torch.max(
                -mb_advantages * ratio,
                -mb_advantages * torch.clamp(ratio, 1 - 0.2, 1 + 0.2),
            ).mean()

            newvalue = newvalue.view(-1)
            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
            v_clipped = b_values[mb_inds] + torch.clamp(
                newvalue - b_values[mb_inds], -0.2, 0.2
            )
            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            loss = pg_loss - ENTROPY_COEF * entropy + VALUE_COEF * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
    
    end_time = time.time()
    print(f"Step {(iteration + 1) * num_envs * ROLLOUT_LENGTH} -- Time {end_time - start_time}s")

# %%
