from craftax_marl.constants import *
from craftax_marl.craftax_state import EnvState, StaticEnvParams
from craftax_marl.util.game_logic_utils import is_boss_vulnerable
from craftax_marl.envs.craftax_balrog_env import ACTIONS
import jax
import jax.numpy as jnp

def render_craftax_text(state: EnvState, static_params: StaticEnvParams):
    long_term_context = ""
    long_term_context += describe_status(state)
    long_term_context += "\n\n"
    long_term_context += describe_env(state)
    long_term_context += "\n\n"

    short_term_context = describe_inventory()
    return long_term_context, short_term_context


def describe_env(state):
    result = ""

    return result

def describe_direction(state: EnvState, front, dimensions, player_idx):
    # TODO: CHECK X and Y are in correct order
    (direction_x, direction_y) = DIRECTIONS[state.player_direction[player_idx]]
    if not front:
        direction_x *= -1
        direction_y *= -1


def build_semantic_map(state: EnvState):
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    # Don't display any item if light < 0.05
    map = state.map[state.player_level]
    padded_grid = jnp.pad(
        map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )
    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2
    map_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
        padded_grid, tl_corner, OBS_DIM
    )
    

    # Block Map (Per player)
    block_map = []
    for player_block_map in map_view:
        block_dict = {}
        for r, row in enumerate(player_block_map):
            for c, block in enumerate(row):
                if block == BlockType.OUT_OF_BOUNDS:
                    continue

                block_loc = get_location(r,c)
                if block not in block_dict or block_loc[0] < block_dict[block][0]:
                    block_dict[block] = block_loc



    # Item Map (Per player)
    # Mob Map (Per mob)
    # Teammate Map (Per teammate)    

    return
    

    

def describe_env(state):
    semantic = info["semantic"][
        info["player_pos"][0] - info["view"][0] // 2 : info["player_pos"][0] + info["view"][0] // 2 + 1,
        info["player_pos"][1] - info["view"][1] // 2 + 1 : info["player_pos"][1] + info["view"][1] // 2,
    ]
    center = np.array([info["view"][0] // 2, info["view"][1] // 2 - 1])
    result = ""
    obs_dims = (9, 11)
    

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


ACTION_DICT = {
    "Noop": "do nothing",
    "Move West": "move west on flat ground",
    "Move East": "move east on flat ground",
    "Move North": "move north on flat ground",
    "Move South": "move south on flat ground",
    "Do": "Multiuse action to collect material, drink from lake and hit creature in front",
    "Sleep": "sleep when energy level is below maximum",
    "Place Stone": "place a stone in front",
    "Place Table": "place a table",
    "Place Furnace": "place a furnace",
    "Place Plant": "place a plant",
    "Make Wood Pickaxe": "craft a wood pickaxe with a nearby table and wood in inventory",
    "Make Stone Pickaxe": "craft a stone pickaxe with a nearby table, wood, and stone in inventory",
    "Make Iron Pickaxe": "craft an iron pickaxe with a nearby table and furnace, wood, coal, and iron in inventory",
    "Make Wood Sword": "craft a wood sword with a nearby table and wood in inventory",
    "Make Stone Sword": "craft a stone sword with a nearby table, wood, and stone in inventory",
    "Make Iron Sword": "craft an iron sword with a nearby table and furnace, wood, coal, and iron in inventory",
    "Rest": "rest to recover some health and energy",
    "Descend": "go down a level in a cave or dungeon",
    "Ascend": "go up a level in a cave or dungeon",
    "Make Diamond Pickaxe": "craft a diamond pickaxe with a nearby table and furnace, wood, coal, and diamond in inventory",
    "Make Diamond Sword": "craft a diamond sword with a nearby table and furnace, wood, coal, and diamond in inventory",
    "Make Iron Armour": "craft iron armour with a nearby table and furnace, coal, and iron in inventory",
    "Make Diamond Armour": "craft diamond armour with a nearby table and furnace, coal, and diamond in inventory",
    "Shoot Arrow": "shoot an arrow with a bow at a target",
    "Make Arrow": "craft arrows using sticks and stones or iron",
    "Cast Spell": "cast an available magic spell using mana",
    "Place Torch": "place a torch to light up an area",
    "Drink Potion Red": "drink a red potion to restore health",
    "Drink Potion Green": "drink a green potion to remove poison or negative effects",
    "Drink Potion Blue": "drink a blue potion to restore mana",
    "Drink Potion Pink": "drink a pink potion for a temporary strength boost",
    "Drink Potion Cyan": "drink a cyan potion for a temporary speed boost",
    "Drink Potion Yellow": "drink a yellow potion for temporary fire resistance",
    "Read Book": "read a book to gain knowledge or unlock new skills",
    "Enchant Sword": "enchant a sword using a nearby enchanting table and magic essence",
    "Enchant Armour": "enchant armour using a nearby enchanting table and magic essence",
    "Make Torch": "craft a torch using stick and coal",
    "Level Up Dexterity": "increase the player's dexterity stat",
    "Level Up Strength": "increase the player's strength stat",
    "Level Up Intelligence": "increase the player's intelligence stat",
    "Enchant Bow": "enchant a bow using a nearby enchanting table and magic essence",
    "Request Food": "request food from nearby sources or teammates",
    "Request Drink": "request drink from nearby sources or teammates",
    "Request Wood": "request wood from nearby sources or teammates",
    "Request Stone": "request stone from nearby sources or teammates",
    "Request Iron": "request iron from nearby sources or teammates",
    "Request Coal": "request coal from nearby sources or teammates",
    "Request Diamond": "request diamond from nearby sources or teammates",
    "Request Ruby": "request ruby from nearby sources or teammates",
    "Request Sapphire": "request sapphire from nearby sources or teammates",
    "Give Player 1": "give the selected item or resource to player 1",
    "Give Player 2": "give the selected item or resource to player 2",
    "Give Player 3": "give the selected item or resource to player 3",
}

def get_instruction_prompt(task=None):
    action_strings = ",\n".join(f"{action}: {ACTION_DICT[action]}" for action in ACTIONS)
    instruction_prompt = f"""
You are an agent playing Crafter. The following are the only valid actions you can take in the game, followed by a short description of each action:

{action_strings}.

These are the game achievements you can get:
1. Collect Wood (Level 1)
2. Place Table (Level 1)
3. Eat Cow (Level 1)
4. Collect Sampling (Level 1)
5. Collect Drink (Level 1)
6. Make Wood Pickaxe (Level 1)
7. Make Wood Sword (Level 1)
8. Place Plant (Level 1)
9. Defeat Zombie (Level 1)
10. Collect Stone (Level 1)
11. Place Stone (Level 1)
12. Eat Plant (Level 1)
13. Defeat Skeleton (Level 1)
14. Make Stone Pickaxe (Level 1)
15. Make Stone Sword (Level 1)
16. Wake Up (Level 1)
17. Place Furnace (Level 1)
18. Collect Coal (Level 1)
19. Collect Iron (Level 1)
20. Make Iron Pickaxe (Level 1)
21. Make Iron Sword (Level 1)
22. Collect Diamond (Level 1)
23. Make Arrow (Level 1)
24. Make Torch (Level 1)
25. Place Torch (Level 1)
26. Make Diamond Sword (Level 3)
27. Make Iron Armour (Level 3)
28. Make Diamond Armour (Level 3)
29. Enter Gnomish Mines (Level 3)
30. Enter Dungeon (Level 3)
31. Enter Sewers (Level 5)
32. Enter Vault (Level 5)
33. Enter Troll Mines (Level 5)
34. Enter Fire Realm (Level 8)
35. Enter Ice Realm (Level 8)
36. Enter Graveyard (Level 8)
37. Defeat Gnome Warrior (Level 3)
38. Defeat Gnome Archer (Level 3)
39. Defeat Orc Solider (Level 3)
40. Defeat Orc Mage (Level 3)
41. Defeat Lizard (Level 5)
42. Defeat Kobold (Level 5)
43. Defeat Troll (Level 5)
44. Defeat Deep Thing (Level 5)
45. Defeat Pigman (Level 8)
46. Defeat Fire Elemental (Level 8)
47. Defeat Frost Troll (Level 8)
48. Defeat Ice Elemental (Level 8)
49. Damage Necromancer (Level 8)
50. Defeat Necromancer (Level 8)
51. Eat Bat (Level 3)
52. Eat Snail (Level 3)
53. Find Bow (Level 3)
54. Fire Bow (Level 3)
55. Collect Sapphire (Level 3)
56. Learn Spell (Level 5)
57. Cast Spell (Level 5)
60. Collect Ruby (Level 3)
61. Make Diamond Pickaxe (Level 3)
62. Open Chest (Level 3)
63. Drink Potion (Level 3)
64. Enchant Sword (Level 5)
65. Enchant Armour (Level 5)
66. Defeat Knight (Level 5)
67. Defeat Archer (Level 5)

In a moment I will present a history of actions and observations from the game.
Your goal is to get as far as possible by completing all the achievements.

PLAY!
""".strip()

    return instruction_prompt