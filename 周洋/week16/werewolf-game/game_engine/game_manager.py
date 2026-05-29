"""Manages all active game instances."""

from typing import Optional
from .game_state import GameEngine

_instances: dict[str, GameEngine] = {}


class GameManager:

    @staticmethod
    def create_game(
        config_name: str = "standard_6",
        player_names: Optional[list[str]] = None,
        shuffle: bool = True,
        use_llm: bool = False,
    ) -> GameEngine:
        agent_manager = None
        if use_llm:
            from agent.agent_manager import AgentManager
            # Create engine first to get players, then attach agent manager
            engine = GameEngine(
                config_name=config_name,
                player_names=player_names,
                shuffle=shuffle,
            )
            agent_manager = AgentManager(engine.players)
            engine.use_llm = True
            engine.agent_manager = agent_manager
        else:
            engine = GameEngine(
                config_name=config_name,
                player_names=player_names,
                shuffle=shuffle,
            )

        _instances[engine.game_id] = engine
        return engine

    @staticmethod
    def get_game(game_id: str) -> Optional[GameEngine]:
        return _instances.get(game_id)

    @staticmethod
    def list_games() -> list[dict]:
        return [
            {
                "game_id": g.game_id,
                "config_name": g.config_name,
                "phase": g.get_phase_name(),
                "day_number": g.day_number,
                "alive_count": g.get_alive_count(),
                "is_game_over": g.is_game_over,
                "winner": g.winner,
            }
            for g in _instances.values()
        ]

    @staticmethod
    def delete_game(game_id: str) -> bool:
        if game_id in _instances:
            del _instances[game_id]
            return True
        return False
