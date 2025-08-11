import jax

from craftax_marl.constants import *
from craftax_marl.craftax_state import EnvState, EnvParams, StaticEnvParams
from craftax_marl.util.game_logic_utils import is_boss_vulnerable
from craftax_marl.renderer.text_helper import get_instruction_prompt

"""
OUTPUT STYLE:
- Instruction Prompt
    - You are an agent playing Crafter. The following are the only valid actions you can take in the game, followed by a short description of each action:
    - Action Strings
    - Achievements
    - In a moment I will present a history of actions and observations from the game. Your goal is to get as far as possible by completing all the achievements. Output a single action in the format "action: <action_string>". PLAY!

- Current Observation
    - Status
        - Health, Specialization, etc.
    - Inventory
    - Surrounding Items
"""
level_names = ["Overworld", "Dungeon", "Gnomish Mines", "Sewers", "Vaults", "Troll Mines", "Fire Realm", "Ice Realm", "Graveyard"]
melee_mob_names = [
    "Zombie",         # 0
    "Orc Soldier",    # 1
    "Gnome Warrior",  # 2
    "Lizard",         # 3
    "Knight",         # 4
    "Troll",          # 5
    "Pig Man",        # 6
    "Frost Troll",    # 7
    "Boss"            # 8 (no melee mob listed)
]

ranged_mob_names = [
    "Skeleton",       # 0
    "Orc Mage",       # 1
    "Gnome Archer",   # 2
    "Kobold",         # 3
    "Archer",         # 4
    "Deep Thing",     # 5
    "Fire Elemental", # 6
    "Ice Elemental",  # 7
    "Boss"            # 8 (no ranged mob listed)
]

passive_mob_names = [
    "Cow",            # 0
    "Snail",          # 1
    "Bat",            # 2
    "Snail",          # 3
    "Snail",          # 4
    "Bat",             # 5
    "Bat",             # 6
    "Boss",             # 7
    "Boss"              # 8
]

def get_agent_inventory(state: EnvState, agent_idx: int):
    inventory_text = "Your Inventory:\n"
    inventory_text += f"  Food: {state.player_food[agent_idx]}\n"
    inventory_text += f"  Drink: {state.player_drink[agent_idx]}\n"
    inventory_text += f"  Wood: {state.inventory.wood[agent_idx]}\n"
    inventory_text += f"  Stone: {state.inventory.stone[agent_idx]}\n"
    inventory_text += f"  Coal: {state.inventory.coal[agent_idx]}\n"
    inventory_text += f"  Iron: {state.inventory.iron[agent_idx]}\n"
    inventory_text += f"  Diamond: {state.inventory.diamond[agent_idx]}\n"
    inventory_text += f"  Sapphire: {state.inventory.sapphire[agent_idx]}\n"
    inventory_text += f"  Ruby: {state.inventory.ruby[agent_idx]}\n"
    inventory_text += f"  Sapling: {state.inventory.sapling[agent_idx]}\n"
    inventory_text += f"  Torch: {state.inventory.torches[agent_idx]}\n"
    inventory_text += f"  Arrow: {state.inventory.arrows[agent_idx]}\n"
    inventory_text += f"  Book: {state.inventory.books[agent_idx]}\n"

    def level_to_material(level):
        if level == 1:
            return "Wood"
        elif level == 2:
            return "Stone"
        elif level == 3:
            return "Iron"
        elif level == 4:
            return "Diamond"

    def level_to_enchantment(level):
        if level == 0:
            return "No"
        if level == 1:
            return "Fire"
        elif level == 2:
            return "Ice"

    if state.inventory.pickaxe[agent_idx] > 0:
        inventory_text += "  " + level_to_material(state.inventory.pickaxe[agent_idx]) + " Pickaxe\n"
    if state.inventory.sword[agent_idx] > 0:
        inventory_text += "  " + level_to_material(state.inventory.sword[agent_idx]) + " Sword"
        inventory_text += (
            " with " + level_to_enchantment(state.sword_enchantment[agent_idx]) + " enchantment\n"
        )
    if state.inventory.bow[agent_idx] > 0:
        inventory_text += (
            "  " + "Bow with " + level_to_enchantment(state.bow_enchantment[agent_idx]) + " enchantment\n"
        )
    inventory_text += f"  Red potion: {state.inventory.potions[agent_idx][0]}\n"
    inventory_text += f"  Green potion: {state.inventory.potions[agent_idx][1]}\n"
    inventory_text += f"  Blue potion: {state.inventory.potions[agent_idx][2]}\n"
    inventory_text += f"  Pink potion: {state.inventory.potions[agent_idx][3]}\n"
    inventory_text += f"  Cyan potion: {state.inventory.potions[agent_idx][4]}\n"
    inventory_text += f"  Yellow potion: {state.inventory.potions[agent_idx][5]}\n"

    def get_armour_level(level):
        if level == 1:
            return "Iron"
        elif level == 2:
            return "Diamond"

    if state.inventory.armour[agent_idx][0] > 0:
        inventory_text += f"  {get_armour_level(state.inventory.armour[agent_idx][0])} Helmet"
        inventory_text += (
            " with "
            + level_to_enchantment(state.armour_enchantments[agent_idx][0])
            + " enchantment\n"
        )

    if state.inventory.armour[agent_idx][1] > 0:
        inventory_text += f"  {get_armour_level(state.inventory.armour[agent_idx][1])} Chestplate"
        inventory_text += (
            " with "
            + level_to_enchantment(state.armour_enchantments[agent_idx][1])
            + " enchantment\n"
        )

    if state.inventory.armour[agent_idx][2] > 0:
        inventory_text += f"  {get_armour_level(state.inventory.armour[agent_idx][2])} Leggings"
        inventory_text += (
            " with "
            + level_to_enchantment(state.armour_enchantments[agent_idx][2])
            + " enchantment\n"
        )

    if state.inventory.armour[agent_idx][3] > 0:
        inventory_text += f"  {get_armour_level(state.inventory.armour[agent_idx][3])} Boots"
        inventory_text += (
            " with "
            + level_to_enchantment(state.armour_enchantments[agent_idx][3])
            + " enchantment\n"
        )
    inventory_text += "\n"
    return inventory_text

@jax.jit
def get_map_view(state: EnvState, agent_idx: int):
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)
    map = state.map[state.player_level]
    
    # Pad the map to handle out-of-bounds areas
    padded_grid = jnp.pad(
        map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )
    # Calculate the top-left corner of the observation window
    tl_corner = state.player_position[agent_idx] - obs_dim_array // 2 + MAX_OBS_DIM + 2
    # Extract the observation window from the padded grid
    map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)

    # Items
    padded_items_map = jnp.pad(
        state.item_map[state.player_level],
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=ItemType.NONE.value,
    )
    # Create item map view for each player
    item_map_view = jax.lax.dynamic_slice(padded_items_map, tl_corner, OBS_DIM)

    # Lights
    padded_light_map = jnp.pad(
        state.light_map[state.player_level],
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=0.0,
    )

    # create light map for each player
    light_map_view = jax.lax.dynamic_slice(
        padded_light_map, tl_corner, OBS_DIM
    )
    light_map_view = light_map_view > 0.05

    return map_view, item_map_view, light_map_view

def get_surroundings_observations(state: EnvState, agent_idx: int):
    map_view, item_map_view, light_map_view = get_map_view(state, agent_idx)
    text = "Surroundings Observations:\n"

    @jax.jit
    def exists_and_closest_to_center(arr, target, light_arr):
        # arr shape is (9, 11)
        H, W = arr.shape
        cy, cx = H // 2, W // 2  # center -> (4, 5) for (9, 11)

        mask = (arr == target)
        found = jnp.any(mask)

        # Manhattan (taxi-driver) distance to center
        rs = jnp.arange(H)[:, None]
        cs = jnp.arange(W)[None, :]
        dist_taxi = jnp.abs(rs - cy) + jnp.abs(cs - cx)

        # mask out non-targets with +inf so they won't be chosen
        dist_masked = jnp.where(mask, dist_taxi, jnp.inf)

        # find the 1D index of closest target (ties -> row-major order)
        idx1d = jnp.argmin(dist_masked)
        row = idx1d // W
        col = idx1d % W

        # If not found, set to (-1, -1)
        row = jnp.where(found, row, -1)
        col = jnp.where(found, col, -1)

        found = jnp.logical_and(found, light_arr[row, col])  # only return if light is present

        return found, row, col

    def describe_offset(row, col, cy=4, cx=5):
        """Return a human-friendly direction string from (cy,cx) to (row,col).
        Uses the problem's convention: lower row = north/south is row-based,
        lower col = WEST, higher col = EAST (per example (3,4) = 1N,1W)."""
        dy = int(row) - cy   # negative -> north, positive -> south
        dx = int(col) - cx   # negative -> west,  positive -> east (per example)

        if dy == 0 and dx == 0:
            return "here"

        vert = ""
        horiz = ""
        steps_v = abs(dy)
        steps_h = abs(dx)

        if dy < 0:
            vert = "north"
        elif dy > 0:
            vert = "south"

        if dx < 0:
            horiz = "west"
        elif dx > 0:
            horiz = "east"

        # Diagonal phrasing when steps match (e.g., "2 steps north-west")
        if steps_v > 0 and steps_h > 0 and steps_v == steps_h:
            return f"{steps_v} steps {vert}-{horiz}"

        parts = []
        if steps_v > 0:
            parts.append(f"{steps_v} {'step' if steps_v == 1 else 'steps'} {vert}")
        if steps_h > 0:
            parts.append(f"{steps_h} {'step' if steps_h == 1 else 'steps'} {horiz}")
        return " and ".join(parts)

    def add_mob(mob_type, position, mask, mob_index, player_position):
        mob_row, mob_col = position[mob_index]
        player_row, player_col = player_position
        dx = jnp.abs(mob_col - player_col)
        dy = jnp.abs(mob_row - player_row)
        if not mask[mob_index] or dx > 4 or dy > 5:
            return ""
        direction = describe_offset(mob_row, mob_col, cy=player_row, cx=player_col)
        return f"  {mob_type}: {direction}\n"

    skip_block_types = [
        BlockType.INVALID.value, BlockType.OUT_OF_BOUNDS.value, BlockType.GRASS.value, BlockType.PATH.value,
        BlockType.SAND.value, BlockType.WALL.value, BlockType.DARKNESS.value, BlockType.WALL_MOSS.value,
        BlockType.FIRE_GRASS.value, BlockType.ICE_GRASS.value,
    ]
    for block_type in BlockType:
        if block_type.value in skip_block_types:
            continue
        found, row, col = exists_and_closest_to_center(map_view, block_type.value, light_map_view)
        if bool(found):
            r = int(row)
            c = int(col)
            direction = describe_offset(r, c, cy=4, cx=5)
            pretty_name = block_type.name.replace("_", " ").title()
            text += f"  {pretty_name}: {direction}\n"
    
    for item_type in ItemType:
        if item_type.value == ItemType.NONE.value:
            continue
        found, row, col = exists_and_closest_to_center(item_map_view, item_type.value, light_map_view)
        if bool(found):
            r = int(row)
            c = int(col)
            direction = describe_offset(r, c, cy=4, cx=5)
            pretty_name = item_type.name.replace("_", " ").title()
            text += f"  {pretty_name}: {direction}\n"
    
    for mob_index in range(state.melee_mobs.mask.shape[1]):
        text += add_mob(melee_mob_names[state.player_level], state.melee_mobs.position[state.player_level], state.melee_mobs.mask[state.player_level], mob_index, state.player_position[agent_idx])
    
    for mob_index in range(state.ranged_mobs.mask.shape[1]):
        text += add_mob(ranged_mob_names[state.player_level], state.ranged_mobs.position[state.player_level], state.ranged_mobs.mask[state.player_level], mob_index, state.player_position[agent_idx])

    for mob_index in range(state.passive_mobs.mask.shape[1]):
        text += add_mob(passive_mob_names[state.player_level], state.passive_mobs.position[state.player_level], state.passive_mobs.mask[state.player_level], mob_index, state.player_position[agent_idx])
    
    for mob_index in range(state.mob_projectiles.mask.shape[1]):
        text += add_mob("Mob Projectile", state.mob_projectiles.position[state.player_level], state.mob_projectiles.mask[state.player_level], mob_index, state.player_position[agent_idx])

    text += "\n"
    return text

def get_map_observation(state: EnvState, agent_idx: int):
    text = "Level Observations:\n"
    text += f"  Floor: {level_names[state.player_level]}\n"
    text += f"  Ladder Open: {'Yes' if state.monsters_killed[state.player_level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL else 'No'}\n"
    text += f"  Light: {state.light_level:.2f}\n"
    text += f"  Is Boss Vulnerable: {'Yes' if is_boss_vulnerable(state) else 'No'}\n"
    text += "\n"

    text += get_surroundings_observations(state, agent_idx)

    text += "\n"
    return text

def get_teammate_dashboard(state: EnvState, agent_idx: int):
    def get_request_item_name(request_type):
        if request_type == Action.REQUEST_FOOD.value:
            return "Food"
        elif request_type == Action.REQUEST_DRINK.value:
            return "Drink"
        elif request_type == Action.REQUEST_WOOD.value:
            return "Wood"
        elif request_type == Action.REQUEST_STONE.value:
            return "Stone"
        elif request_type == Action.REQUEST_IRON.value:
            return "Iron"
        elif request_type == Action.REQUEST_COAL.value:
            return "Coal"
        elif request_type == Action.REQUEST_DIAMOND.value:
            return "Diamond"
        elif request_type == Action.REQUEST_RUBY.value:
            return "Ruby"
        elif request_type == Action.REQUEST_SAPPHIRE.value:
            return "Sapphire"
    
    def describe_relative(agent_pos, mate_pos):
        # dy>0 means teammate is SOUTH (since higher row is south), dy<0 means NORTH
        dy = mate_pos[0] - agent_pos[0]
        # dx>0 means teammate is WEST (since higher col is west), dx<0 means EAST
        dx = mate_pos[1] - agent_pos[1]

        vert = ""
        horiz = ""
        steps_v = abs(dy)
        steps_h = abs(dx)

        if dy < 0:
            vert = "north"
        elif dy > 0:
            vert = "south"

        if dx < 0:
            horiz = "east"
        elif dx > 0:
            horiz = "west"

        # Same tile
        if steps_v == 0 and steps_h == 0:
            return "here"

        # Diagonal phrasing when steps match (e.g., 4 steps north-east)
        if steps_v > 0 and steps_h > 0 and steps_v == steps_h:
            return f"{steps_v} steps {vert}-{horiz}"

        # Otherwise, be explicit on both axes
        parts = []
        if steps_v > 0:
            parts.append(f"{steps_v} {'step' if steps_v==1 else 'steps'} {vert}")
        if steps_h > 0:
            parts.append(f"{steps_h} {'step' if steps_h==1 else 'steps'} {horiz}")
        return " and ".join(parts)
    
    teammate_name = ["Warrior", "Forager", "Miner"]
    text = "Teammate Status:\n"
    for i in range(len(state.player_alive)):
        if i != agent_idx:
            text += f"  {teammate_name[i]}:\n"
            text += f"    Alive: {'Yes' if state.player_alive[i] else 'No'}\n"
            text += f"    Health: {state.player_health[i]}\n"
            rel = describe_relative(state.player_position[agent_idx], state.player_position[i])
            text += f"    Relative position: {rel}\n"
            if state.request_duration[i] > 0:
                text += f"    Requesting: {get_request_item_name(state.request_type[i])}\n"

    text += "\n"
    return text

def get_agent_status(state: EnvState, agent_idx: int):
    direction_heading = ["", "West", "East", "North", "South"]
    text = "Your Status:\n"
    text += f"  Alive: {'Yes' if state.player_alive[agent_idx] else 'No'}\n"
    text += f"  Health: {state.player_health[agent_idx]}\n"
    text += f"  Direction: {direction_heading[state.player_direction[agent_idx]]}\n"
    text += f"  Energy: {state.player_energy[agent_idx]}\n"
    text += f"  Mana: {state.player_mana[agent_idx]}\n"
    text += f"  XP: {state.player_xp[agent_idx]}\n"
    text += f"  Dexterity: {state.player_dexterity[agent_idx]}\n"
    text += f"  Strength: {state.player_strength[agent_idx]}\n"
    text += f"  Intelligence: {state.player_intelligence[agent_idx]}\n"
    text += f"  Learned Spell: {'Yes' if state.learned_spells[agent_idx] else 'No'}\n"
    text += f"  Is Sleeping: {'Yes' if state.is_sleeping[agent_idx] else 'No'}\n"
    text += f"  Is Resting: {'Yes' if state.is_resting[agent_idx] else 'No'}\n"
    text += "\n"
    return text

def get_agent_observation_prompt(state: EnvState, agent_idx: int):
    text = f"Step {state.timestep} Observations:\n"
    text += get_agent_status(state, agent_idx)
    text += get_agent_inventory(state, agent_idx)
    text += get_map_observation(state, agent_idx)
    text += get_teammate_dashboard(state, agent_idx)
    text += "\n"
    return text

def get_observation_prompt(state: EnvState, agents: list[str]) :
    return [get_agent_observation_prompt(state, agent_idx) for agent_idx in range(len(agents))]

def render_craftax_text(state: EnvState, agents: list[str]):
    instruction_prompt = get_instruction_prompt()
    observation_prompt = get_observation_prompt(state, agents)

    return {agent: {"instruction": instruction_prompt[i], "obs": observation_prompt[i]} for i, agent in enumerate(agents)}
