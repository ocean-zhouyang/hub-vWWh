"""AgentManager — 持有所有 PlayerAgent 实例，处理协调逻辑（如狼人投票聚合）。"""

import random
import sys
from collections import Counter
from typing import Optional

from .player_agent import PlayerAgent


class AgentManager:
    """持有每个玩家的独立 Agent，对外暴露与 GameEngine 对接的统一接口。"""

    def __init__(self, players: list):
        wolves = [p for p in players if p.role == "werewolf"]
        self.agents: dict[int, PlayerAgent] = {}

        for p in players:
            known_wolves = None
            if p.role == "werewolf":
                known_wolves = [
                    {"player_id": w.player_id, "name": w.name}
                    for w in wolves if w.player_id != p.player_id
                ]
            self.agents[p.player_id] = PlayerAgent(
                player_id=p.player_id,
                name=p.name,
                role=p.role,
                camp=p.camp,
                known_wolves=known_wolves,
            )

    def get_agent(self, player_id: int) -> Optional[PlayerAgent]:
        return self.agents.get(player_id)

    # ── 狼人刀人（需聚合） ────────────────────────────────

    def decide_wolf_kill(self, all_players: list) -> tuple[list[dict], Optional[int]]:
        wolves = [p for p in all_players if p.is_alive and p.role == "werewolf"]
        targets = [p for p in all_players if p.is_alive and p.role != "werewolf"]
        if not wolves or not targets:
            return [], None

        wolf_votes = []
        for wolf in wolves:
            agent = self.agents[wolf.player_id]
            target = agent.decide_wolf_kill(all_players)
            wolf_votes.append({"player_id": wolf.player_id, "target": target})

        # 按多数决确定最终目标
        cnt = Counter(v["target"] for v in wolf_votes)
        top = max(cnt.values())
        final_target = random.choice([t for t, c in cnt.items() if c == top])

        target_name = next(
            (p.name for p in all_players if p.player_id == final_target), str(final_target)
        )
        print(
            f"\n[LLM] 📋 狼队投票结果: 击杀 {target_name}（玩家{final_target}）\n",
            file=sys.stderr,
        )
        sys.stderr.flush()

        return wolf_votes, final_target

    # ── 预言家查验 ────────────────────────────────────────

    def decide_seer_check(self, all_players: list) -> tuple[Optional[int], Optional[str]]:
        seers = [p for p in all_players if p.is_alive and p.role == "seer"]
        if not seers:
            return None, None
        agent = self.agents[seers[0].player_id]
        return agent.decide_seer_check(all_players)

    # ── 女巫行动 ──────────────────────────────────────────

    def decide_witch_action(self, all_players: list,
                            wolf_target: Optional[int]) -> tuple[bool, Optional[int]]:
        witches = [p for p in all_players if p.is_alive and p.role == "witch"]
        if not witches:
            return False, None
        agent = self.agents[witches[0].player_id]
        return agent.decide_witch_action(all_players, wolf_target)

    # ── 发言 ──────────────────────────────────────────────

    def decide_speech(self, player_id: int, all_players: list,
                      day_number: int, night_event: str,
                      speeches_so_far: list[dict]) -> str:
        agent = self.agents.get(player_id)
        if not agent:
            return "我没什么好说的。"
        return agent.generate_speech(all_players, day_number, night_event, speeches_so_far)

    # ── 投票 ──────────────────────────────────────────────

    def decide_vote(self, player_id: int, all_players: list,
                    day_number: int, speeches: list[dict]) -> int:
        agent = self.agents.get(player_id)
        alive_others = [p for p in all_players
                        if p.is_alive and p.player_id != player_id]
        if not agent or not alive_others:
            return alive_others[0].player_id if alive_others else player_id
        return agent.decide_vote(all_players, day_number, speeches)

    def get_thoughts(self, player_id: int) -> list[dict]:
        agent = self.agents.get(player_id)
        return agent.thoughts if agent else []
