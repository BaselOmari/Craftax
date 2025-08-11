from craftax_marl.constants import *

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
    "Give Warrior": "give the warrior their requested item",
    "Give Forager": "give the forager their requested item",
    "Give Miner": "give the miner their requested item",
}

def get_instruction_prompt():
    action_strings = "\n".join(f"- {action}: {ACTION_DICT[action]}" for action in ACTION_DICT.keys())

    # Achievement dictionary with points (name: points)
    ACHIEVEMENT_POINTS = {
        "COLLECT_WOOD": 1, "COLLECT_FOOD": 1,"PLACE_TABLE": 1, "EAT_COW": 1, "COLLECT_SAPLING": 1, "COLLECT_DRINK": 1,
        "MAKE_WOOD_PICKAXE": 1, "MAKE_WOOD_SWORD": 1, "PLACE_PLANT": 1, "DEFEAT_ZOMBIE": 1,
        "COLLECT_STONE": 1, "PLACE_STONE": 1, "EAT_PLANT": 1, "DEFEAT_SKELETON": 1,
        "MAKE_STONE_PICKAXE": 1, "MAKE_STONE_SWORD": 1, "WAKE_UP": 1, "PLACE_FURNACE": 1,
        "COLLECT_COAL": 1, "COLLECT_IRON": 1, "MAKE_IRON_PICKAXE": 1, "MAKE_IRON_SWORD": 1,
        "COLLECT_DIAMOND": 1, "MAKE_ARROW": 1, "MAKE_TORCH": 1, "PLACE_TORCH": 1,
        "MAKE_DIAMOND_SWORD": 3, "MAKE_IRON_ARMOUR": 3, "MAKE_DIAMOND_ARMOUR": 3,
        "ENTER_GNOMISH_MINES": 3, "ENTER_DUNGEON": 3, "ENTER_SEWERS": 5, "ENTER_VAULT": 5,
        "ENTER_TROLL_MINES": 5, "ENTER_FIRE_REALM": 8, "ENTER_ICE_REALM": 8, "ENTER_GRAVEYARD": 8,
        "DEFEAT_GNOME_WARRIOR": 3, "DEFEAT_GNOME_ARCHER": 3, "DEFEAT_ORC_SOLDIER": 3,
        "DEFEAT_ORC_MAGE": 3, "DEFEAT_LIZARD": 5, "DEFEAT_KOBOLD": 5, "DEFEAT_TROLL": 5,
        "DEFEAT_DEEP_THING": 5, "DEFEAT_PIGMAN": 8, "DEFEAT_FIRE_ELEMENTAL": 8,
        "DEFEAT_FROST_TROLL": 8, "DEFEAT_ICE_ELEMENTAL": 8, "DAMAGE_NECROMANCER": 8,
        "DEFEAT_NECROMANCER": 8, "EAT_BAT": 3, "EAT_SNAIL": 3, "FIND_BOW": 3, "FIRE_BOW": 3,
        "COLLECT_SAPPHIRE": 3, "LEARN_SPELL": 5, "CAST_SPELL": 5, "COLLECT_RUBY": 3,
        "MAKE_DIAMOND_PICKAXE": 3, "OPEN_CHEST": 3, "DRINK_POTION": 3,
        "ENCHANT_SWORD": 5, "ENCHANT_ARMOUR": 5, "DEFEAT_KNIGHT": 5, "DEFEAT_ARCHER": 5
    }

    # Agent-specific achievements
    FORAGER = [
        "COLLECT_WOOD", "PLACE_TABLE", "EAT_COW", "COLLECT_SAPLING", "COLLECT_DRINK", "PLACE_PLANT",
        "DEFEAT_ZOMBIE", "COLLECT_STONE", "PLACE_STONE", "EAT_PLANT", "WAKE_UP", "PLACE_FURNACE",
        "COLLECT_COAL", "COLLECT_IRON", "COLLECT_DIAMOND", "COLLECT_FOOD", "MAKE_IRON_ARMOUR",
        "MAKE_DIAMOND_ARMOUR", "ENTER_GNOMISH_MINES", "ENTER_DUNGEON", "ENTER_SEWERS", "ENTER_VAULT",
        "ENTER_TROLL_MINES", "EAT_BAT", "EAT_SNAIL", "FIND_BOW", "FIRE_BOW", "COLLECT_SAPPHIRE",
        "OPEN_CHEST", "DRINK_POTION", "ENTER_FIRE_REALM", "ENTER_ICE_REALM", "ENTER_GRAVEYARD",
        "DEFEAT_GNOME_WARRIOR", "DEFEAT_GNOME_ARCHER", "DEFEAT_ORC_SOLDIER", "DEFEAT_ORC_MAGE",
        "DEFEAT_LIZARD", "DEFEAT_KOBOLD", "DEFEAT_TROLL", "LEARN_SPELL", "CAST_SPELL",
        "COLLECT_RUBY", "ENCHANT_ARMOUR", "DEFEAT_DEEP_THING", "DEFEAT_PIGMAN", "DEFEAT_FIRE_ELEMENTAL",
        "DEFEAT_FROST_TROLL", "DEFEAT_ICE_ELEMENTAL", "DAMAGE_NECROMANCER", "DEFEAT_NECROMANCER",
        "DEFEAT_KNIGHT", "DEFEAT_ARCHER"
    ]

    MINER = [
        "COLLECT_DRINK", "MAKE_WOOD_PICKAXE", "MAKE_STONE_PICKAXE", "MAKE_IRON_PICKAXE",
        "COLLECT_STONE", "PLACE_STONE", "PLACE_FURNACE", "COLLECT_COAL", "COLLECT_IRON",
        "COLLECT_DIAMOND", "MAKE_TORCH", "PLACE_TORCH", "COLLECT_FOOD", "MAKE_DIAMOND_PICKAXE",
        "OPEN_CHEST", "DRINK_POTION", "ENTER_FIRE_REALM", "ENTER_ICE_REALM", "ENTER_GRAVEYARD",
        "DEFEAT_GNOME_WARRIOR", "DEFEAT_GNOME_ARCHER", "DEFEAT_ORC_SOLDIER", "DEFEAT_ORC_MAGE",
        "DEFEAT_LIZARD", "DEFEAT_KOBOLD", "DEFEAT_TROLL", "LEARN_SPELL", "CAST_SPELL",
        "COLLECT_RUBY", "ENCHANT_ARMOUR", "DEFEAT_DEEP_THING", "DEFEAT_PIGMAN", "DEFEAT_FIRE_ELEMENTAL",
        "DEFEAT_FROST_TROLL", "DEFEAT_ICE_ELEMENTAL", "DAMAGE_NECROMANCER", "DEFEAT_NECROMANCER",
        "DEFEAT_KNIGHT", "DEFEAT_ARCHER"
    ]

    WARRIOR = [
        "COLLECT_WOOD", "MAKE_WOOD_SWORD", "DEFEAT_ZOMBIE", "DEFEAT_SKELETON", "MAKE_STONE_SWORD",
        "MAKE_IRON_SWORD", "MAKE_ARROW", "MAKE_DIAMOND_SWORD", "MAKE_IRON_ARMOUR", "MAKE_DIAMOND_ARMOUR",
        "FIND_BOW", "FIRE_BOW", "ENTER_GNOMISH_MINES", "ENTER_DUNGEON", "ENTER_SEWERS", "ENTER_VAULT",
        "ENTER_TROLL_MINES", "OPEN_CHEST", "DRINK_POTION", "ENTER_FIRE_REALM", "ENTER_ICE_REALM",
        "ENTER_GRAVEYARD", "DEFEAT_GNOME_WARRIOR", "DEFEAT_GNOME_ARCHER", "DEFEAT_ORC_SOLDIER",
        "DEFEAT_ORC_MAGE", "DEFEAT_LIZARD", "DEFEAT_KOBOLD", "DEFEAT_TROLL", "LEARN_SPELL",
        "CAST_SPELL", "ENCHANT_SWORD", "ENCHANT_ARMOUR", "DEFEAT_DEEP_THING", "DEFEAT_PIGMAN",
        "DEFEAT_FIRE_ELEMENTAL", "DEFEAT_FROST_TROLL", "DEFEAT_ICE_ELEMENTAL", "DAMAGE_NECROMANCER",
        "DEFEAT_NECROMANCER", "DEFEAT_KNIGHT", "DEFEAT_ARCHER"
    ]

    def build_prompt(agent_type, achievement_list):
        formatted_achievements = "\n".join(
            f"- {ach.replace('_', ' ').title()} ({ACHIEVEMENT_POINTS[ach]} Point{'s' if ACHIEVEMENT_POINTS[ach] > 1 else ''})"
            for ach in sorted(achievement_list)
        )
        return f"""
You are a **{agent_type}** agent playing Multi-Agent Craftax, with 2 other teammates.

The following are the only valid actions you can take in the game, followed by a short description of each action:

{action_strings}

As a **{agent_type}**, the achievements you can work towards include:
{formatted_achievements}

In a moment I will present a history of actions and observations from the game.
Your goal is to get as far as possible by completing all the achievements you are capable of achieving.
Output a single action in the format "action: <action_string>".
"""

    return [
        build_prompt("Warrior", WARRIOR),
        build_prompt("Forager", FORAGER),
        build_prompt("Miner", MINER)
    ]

# Map string names from the prompt to enum members
action_map = {
    "Noop": Action.NOOP,
    "Move West": Action.LEFT,
    "Move East": Action.RIGHT,
    "Move North": Action.UP,
    "Move South": Action.DOWN,
    "Do": Action.DO,
    "Sleep": Action.SLEEP,
    "Place Stone": Action.PLACE_STONE,
    "Place Table": Action.PLACE_TABLE,
    "Place Furnace": Action.PLACE_FURNACE,
    "Place Plant": Action.PLACE_PLANT,
    "Make Wood Pickaxe": Action.MAKE_WOOD_PICKAXE,
    "Make Stone Pickaxe": Action.MAKE_STONE_PICKAXE,
    "Make Iron Pickaxe": Action.MAKE_IRON_PICKAXE,
    "Make Wood Sword": Action.MAKE_WOOD_SWORD,
    "Make Stone Sword": Action.MAKE_STONE_SWORD,
    "Make Iron Sword": Action.MAKE_IRON_SWORD,
    "Rest": Action.REST,
    "Descend": Action.DESCEND,
    "Ascend": Action.ASCEND,
    "Make Diamond Pickaxe": Action.MAKE_DIAMOND_PICKAXE,
    "Make Diamond Sword": Action.MAKE_DIAMOND_SWORD,
    "Make Iron Armour": Action.MAKE_IRON_ARMOUR,
    "Make Diamond Armour": Action.MAKE_DIAMOND_ARMOUR,
    "Shoot Arrow": Action.SHOOT_ARROW,
    "Make Arrow": Action.MAKE_ARROW,
    "Cast Spell": Action.CAST_SPELL,
    "Place Torch": Action.PLACE_TORCH,
    "Drink Potion Red": Action.DRINK_POTION_RED,
    "Drink Potion Green": Action.DRINK_POTION_GREEN,
    "Drink Potion Blue": Action.DRINK_POTION_BLUE,
    "Drink Potion Pink": Action.DRINK_POTION_PINK,
    "Drink Potion Cyan": Action.DRINK_POTION_CYAN,
    "Drink Potion Yellow": Action.DRINK_POTION_YELLOW,
    "Read Book": Action.READ_BOOK,
    "Enchant Sword": Action.ENCHANT_SWORD,
    "Enchant Armour": Action.ENCHANT_ARMOUR,
    "Make Torch": Action.MAKE_TORCH,
    "Level Up Dexterity": Action.LEVEL_UP_DEXTERITY,
    "Level Up Strength": Action.LEVEL_UP_STRENGTH,
    "Level Up Intelligence": Action.LEVEL_UP_INTELLIGENCE,
    "Enchant Bow": Action.ENCHANT_BOW,
    "Request Food": Action.REQUEST_FOOD,
    "Request Drink": Action.REQUEST_DRINK,
    "Request Wood": Action.REQUEST_WOOD,
    "Request Stone": Action.REQUEST_STONE,
    "Request Iron": Action.REQUEST_IRON,
    "Request Coal": Action.REQUEST_COAL,
    "Request Diamond": Action.REQUEST_DIAMOND,
    "Request Ruby": Action.REQUEST_RUBY,
    "Request Sapphire": Action.REQUEST_SAPPHIRE,
    "Give Warrior": Action.GIVE_WARRIOR,
    "Give Forager": Action.GIVE_FORAGER,
    "Give Miner": Action.GIVE_MINER,
}