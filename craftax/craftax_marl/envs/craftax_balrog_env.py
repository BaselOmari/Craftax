# %%
import itertools

import jax
import crafter
import gym
import numpy as np
from PIL import Image

from craftax_marl.constants import *
from craftax_marl.renderer.renderer_text import render_craftax_text, get_instruction_prompt
from craftax_marl.craftax_state import EnvState, EnvParams, StaticEnvParams
from craftax_marl.game_logic import craftax_step
from craftax_marl.world_gen.world_gen import generate_world


# from balrog.environments import Strings

ACTIONS = [
    "Noop",
    "Move West",
    "Move East",
    "Move North",
    "Move South",
    "Do",
    "Sleep",
    "Place Stone",
    "Place Table",
    "Place Furnace",
    "Place Plant",
    "Make Wood Pickaxe",
    "Make Stone Pickaxe",
    "Make Iron Pickaxe",
    "Make Wood Sword",
    "Make Stone Sword",
    "Make Iron Sword",
    "Rest",
    "Descend",
    "Ascend",
    "Make Diamond Pickaxe",
    "Make Diamond Sword",
    "Make Iron Armour",
    "Make Diamond Armour",
    "Shoot Arrow",
    "Make Arrow",
    "Cast Spell",
    "Place Torch",
    "Drink Potion Red",
    "Drink Potion Green",
    "Drink Potion Blue",
    "Drink Potion Pink",
    "Drink Potion Cyan",
    "Drink Potion Yellow",
    "Read Book",
    "Enchant Sword",
    "Enchant Armour",
    "Make Torch",
    "Level Up Dexterity",
    "Level Up Strength",
    "Level Up Intelligence",
    "Enchant Bow",
    "Request Food",
    "Request Drink",
    "Request Wood",
    "Request Stone",
    "Request Iron",
    "Request Coal",
    "Request Diamond",
    "Request Ruby",
    "Request Sapphire",
    "Give Player 1",
    "Give Player 2",
    "Give Player 3",
]

# id_to_item = [0] * 19


# dummyenv = crafter.Env()
# for name, ind in itertools.chain(dummyenv._world._mat_ids.items(), dummyenv._sem_view._obj_ids.items()):
#     name = (
#         str(name)[str(name).find("objects.") + len("objects.") : -2].lower() if "objects." in str(name) else str(name)
#     )
#     id_to_item[ind] = name
# player_idx = id_to_item.index("player")
# del dummyenv

# print(id_to_item)
# %%

vitals = [
    "health",
    "food",
    "drink",
    "energy",
]

rot = np.array([[0, -1], [1, 0]])
directions = ["front", "right", "back", "left"]


def describe_inventory(info):
    result = ""

    status_str = "Your status:\n{}".format("\n".join(["- {}: {}/9".format(v, info["inventory"][v]) for v in vitals]))
    result += status_str + "\n\n"

    inventory_str = "\n".join(
        ["- {}: {}".format(i, num) for i, num in info["inventory"].items() if i not in vitals and num != 0]
    )
    inventory_str = (
        "Your inventory:\n{}".format(inventory_str) if inventory_str else "You have nothing in your inventory."
    )
    result += inventory_str  # + "\n\n"

    return result.strip()


REF = np.array([0, 1])


def rotation_matrix(v1, v2):
    dot = np.dot(v1, v2)
    cross = np.cross(v1, v2)
    rotation_matrix = np.array([[dot, -cross], [cross, dot]])
    return rotation_matrix


def describe_loc(ref, P):
    desc = []
    if ref[1] > P[1]:
        desc.append("north")
    elif ref[1] < P[1]:
        desc.append("south")
    if ref[0] > P[0]:
        desc.append("west")
    elif ref[0] < P[0]:
        desc.append("east")

    return "-".join(desc)


def describe_env(info):
    assert info["semantic"][info["player_pos"][0], info["player_pos"][1]] == player_idx
    semantic = info["semantic"][
        info["player_pos"][0] - info["view"][0] // 2 : info["player_pos"][0] + info["view"][0] // 2 + 1,
        info["player_pos"][1] - info["view"][1] // 2 + 1 : info["player_pos"][1] + info["view"][1] // 2,
    ]
    center = np.array([info["view"][0] // 2, info["view"][1] // 2 - 1])
    result = ""
    x = np.arange(semantic.shape[1])
    y = np.arange(semantic.shape[0])
    x1, y1 = np.meshgrid(x, y)
    loc = np.stack((y1, x1), axis=-1)
    dist = np.absolute(center - loc).sum(axis=-1)
    obj_info_list = []

    facing = info["player_facing"]
    max_y, max_x = semantic.shape
    target_x = center[0] + facing[0]
    target_y = center[1] + facing[1]

    if 0 <= target_x < max_x and 0 <= target_y < max_y:
        target_id = semantic[int(target_x), int(target_y)]
        target_item = id_to_item[target_id]
        obs = "You face {} at your front.".format(target_item)
    else:
        obs = "You face nothing at your front."

    for idx in np.unique(semantic):
        if idx == player_idx:
            continue

        smallest = np.unravel_index(np.argmin(np.where(semantic == idx, dist, np.inf)), semantic.shape)
        obj_info_list.append(
            (
                id_to_item[idx],
                dist[smallest],
                describe_loc(np.array([0, 0]), smallest - center),
            )
        )

    if len(obj_info_list) > 0:
        status_str = "You see:\n{}".format(
            "\n".join(["- {} {} steps to your {}".format(name, dist, loc) for name, dist, loc in obj_info_list])
        )
    else:
        status_str = "You see nothing away from you."
    result += status_str + "\n\n"
    result += obs.strip()

    return result.strip()


def describe_act(action):
    result = ""

    action_str = action.replace("do_", "interact_")
    action_str = action_str.replace("move_up", "move_north")
    action_str = action_str.replace("move_down", "move_south")
    action_str = action_str.replace("move_left", "move_west")
    action_str = action_str.replace("move_right", "move_east")

    act = "You took action {}.".format(action_str)
    result += act

    return result.strip()


def describe_status(info):
    if info["sleeping"]:
        return "You are sleeping, and will not be able take actions until energy is full.\n\n"
    elif info["dead"]:
        return "You died.\n\n"
    else:
        return ""


def describe_frame(info):
    try:
        result = ""

        result += describe_status(info)
        result += "\n\n"
        result += describe_env(info)
        result += "\n\n"

        return result.strip(), describe_inventory(info)
    except Exception:
        breakpoint()
        return "Error, you are out of the map."

class CraftaxTextEnvironment(gym.Env):
    default_iter = 10
    default_steps = 10000

    def __init__(
        self,
        task="",
        max_episode_steps=2,
    ):
        super().__init__()
        self.score_tracker = 0
        self.language_action_space = Strings(ACTIONS)
        self.default_action = "Noop"
        self.max_steps=max_episode_steps
        self.achievements = None

        self.static_env_params = StaticEnvParams()
        self.params = EnvParams()
        
        self.agents = [
            f"agent_{i}" for i in range(self.static_env_params.player_count)
        ]

        self.state = None
        self.key = jax.random.PRNGKey(0)
    
    def get_text_action(self, action):
        return self.language_action_space._values[action]

    def _step_impl(self, actions):
        self.state, reward = craftax_step(self.key, self.state, actions, self.default_params, self.static_env_params)
        done = self.is_terminal(self.state, self.params)
        info = {}
        (long_term_context, short_term_context) = render_craftax_text(self.state, self.static_env_params)
        obs = {
            "text": {
                "long_term_context": long_term_context,
                "short_term_context": short_term_context,
            }
        }
        return obs, reward[0], done, info

    def reset(self):
        self.key, _key = jax.random.split(self.key)
        self.state = generate_world(_key, self.params, self.static_env_params)
        obs, reward, done, info = self._step_impl([0 for ag in self.agents])
        return obs

    def step(self, actions):
        obs, reward, done, info = self._step_impl(
            [self.language_action_space.map(a) for a in actions]
        )
        return obs, reward, done, info

    def get_stats(self):
        return {}
    
    def get_instruction_prompt(self):
        return get_instruction_prompt()
