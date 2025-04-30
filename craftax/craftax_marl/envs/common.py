from typing import List

from craftax_marl.craftax_state import EnvState, StaticEnvParams
from craftax_marl.constants import *


def compute_score(state: EnvState, done: bool, player_names: List, static_params: StaticEnvParams):
    achievements = state.achievements * done * 100.0
    info = {}
    for achievement in Achievement:
        achievement_name = f"Achievements/{achievement.name.lower()}"
        player_total = jnp.array(0.0)
        for player in range(static_params.player_count):
            player_achievement_name = f"{achievement_name}/{player_names[player]}"
            player_achievement_value = achievements[player, achievement.value]
            info[player_achievement_name] = player_achievement_value
            player_total = jnp.maximum(player_total, player_achievement_value)
        info[achievement_name] = player_total
    # Geometric mean with an offset of 1%
    info["score"] = jnp.exp(jnp.mean(jnp.log(1 + achievements))) - 1.0
    return info


def compute_score_mappo(state: EnvState, done: bool, player_names: List, static_params: StaticEnvParams):
    achievements = state.achievements * done * 100.0
    info = {}
    for achievement in Achievement:
        achievement_name = f"Achievements/{achievement.name.lower()}"
        info[achievement_name] = jnp.repeat(
            achievements[:, achievement.value].max(), 
            static_params.player_count
        )
    info["trade_count"] = jnp.repeat(state.trade_count, static_params.player_count)
    info["ff_damage_dealt"] = jnp.repeat(state.ff_damage_dealt, static_params.player_count)
    info["revives"] = jnp.repeat(state.revives, static_params.player_count)
    info["final_player_recover"] = state.player_recover
    return info
