"""独立 PlayerAgent — 每个玩家一个 agent，各自持有私有状态和记忆。"""

import json
import re
import sys
from typing import Optional

from .llm_client import call_llm
from . import prompts


ROLE_LABELS = {
    "werewolf": "🐺 狼人", "seer": "🔮 预言家",
    "witch": "🧪 女巫", "hunter": "🏹 猎人",
    "villager": "👤 村民",
}

ROLE_NAMES_CN = {
    "werewolf": "狼人", "seer": "预言家",
    "witch": "女巫", "hunter": "猎人",
    "villager": "村民",
}


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(text[start:end + 1])


class PlayerAgent:
    """一个独立 AI 玩家。持有私有信息、决策能力和事件记忆。"""

    def __init__(self, player_id: int, name: str, role: str, camp: str,
                 known_wolves: Optional[list[dict]] = None):
        self.player_id = player_id
        self.name = name
        self.role = role
        self.camp = camp
        self.known_wolves = known_wolves or []

        # 私有状态
        self.seer_history: list[tuple[int, str]] = []
        self.witch_save_used = False
        self.witch_poison_used = False
        self.thoughts: list[dict] = []
        self.events: list[str] = []  # 记忆 — 按时间顺序记录该玩家经历的事件

    # ── 内部方法 ──────────────────────────────────────────

    def _log(self, phase: str, thought: str, decision: str):
        self.thoughts.append({"phase": phase, "thought": thought, "decision": decision})
        label = ROLE_LABELS.get(self.role, self.role)
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"[LLM] {self.name} ({label}) — {phase}", file=sys.stderr)
        if thought:
            print(f"[LLM] 💭 思考: {thought}", file=sys.stderr)
        print(f"[LLM] ✅ 决策: {decision}", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
        sys.stderr.flush()

    def _remember(self, event: str):
        self.events.append(f"[第{event}]")

    def _alive_list(self, all_players: list) -> list[dict]:
        return [p.to_dict() for p in all_players]

    def _role_cn(self) -> str:
        return ROLE_NAMES_CN.get(self.role, self.role)

    # ── 狼人刀人 ──────────────────────────────────────────

    def decide_wolf_kill(self, all_players: list) -> int:
        """返回要击杀的玩家 ID。"""
        targets = [p for p in all_players if p.is_alive and p.role != "werewolf"]
        alive_list = self._alive_list(all_players)
        user = prompts.build_wolf_kill_user(
            self.player_id, self.name, self.known_wolves, alive_list,
        )
        try:
            resp = call_llm(prompts.WOLF_KILL_SYSTEM,
                            user + "\n\n" + prompts.WOLF_KILL_OUTPUT_EXAMPLE)
            data = _extract_json(resp)
            target = int(data["target_player_id"])
            valid = {p.player_id for p in targets}
            if target not in valid:
                target = targets[0].player_id
        except Exception:
            target = targets[0].player_id

        self._log("黑夜-狼刀", data.get("thought", "") if 'data' in dir() else "",
                  f"击杀 玩家{target}")
        return target

    # ── 预言家查验 ────────────────────────────────────────

    def decide_seer_check(self, all_players: list) -> tuple[Optional[int], Optional[str]]:
        """返回 (target_id, result)。"""
        alive_list = self._alive_list(all_players)
        user = prompts.build_seer_check_user(self.player_id, self.name, alive_list)

        try:
            resp = call_llm(prompts.SEER_CHECK_SYSTEM,
                            user + "\n\n" + prompts.SEER_CHECK_OUTPUT_EXAMPLE)
            data = _extract_json(resp)
            target_id = int(data["target_player_id"])
            valid = {p.player_id for p in all_players
                     if p.is_alive and p.player_id != self.player_id}
            if target_id not in valid:
                return None, None
        except Exception:
            return None, None

        target = next(p for p in all_players if p.player_id == target_id)
        result = "wolf" if target.role == "werewolf" else "good"
        result_label = "🐺 狼人" if result == "wolf" else "😇 好人"
        self.seer_history.append((target_id, result))
        self._log("黑夜-查验", data.get("thought", ""),
                  f"查验 玩家{target_id}({target.name}) → {result_label}")
        return target_id, result

    # ── 女巫行动 ──────────────────────────────────────────

    def decide_witch_action(self, all_players: list,
                            wolf_target: Optional[int]) -> tuple[bool, Optional[int]]:
        """返回 (是否解救, 毒药目标ID)。"""
        victim = next((p for p in all_players if p.player_id == wolf_target), None)
        victim_dict = {"player_id": victim.player_id, "name": victim.name} if victim else None
        alive_list = self._alive_list(all_players)
        user = prompts.build_witch_action_user(
            self.player_id, self.name, victim_dict,
            self.witch_save_used, self.witch_poison_used, alive_list,
        )
        try:
            resp = call_llm(prompts.WITCH_ACTION_SYSTEM,
                            user + "\n\n" + prompts.WITCH_ACTION_OUTPUT_EXAMPLE)
            data = _extract_json(resp)
        except Exception:
            return False, None

        saved = False
        poison_target = None

        if data.get("use_save") and not self.witch_save_used and victim is not None:
            if victim.player_id != self.player_id:
                saved = True
                self.witch_save_used = True

        pt = data.get("poison_target_player_id")
        if pt is not None and not self.witch_poison_used:
            pt = int(pt)
            valid = {p.player_id for p in all_players
                     if p.is_alive and p.player_id != self.player_id}
            if pt in valid:
                poison_target = pt
                self.witch_poison_used = True

        parts = []
        if saved:
            parts.append(f"使用解药救活 玩家{victim.player_id}({victim.name})")
        if poison_target is not None:
            pn = next((p.name for p in all_players if p.player_id == poison_target), str(poison_target))
            parts.append(f"使用毒药毒死 玩家{poison_target}({pn})")
        if not parts:
            parts.append("什么都不做")
        self._log("黑夜-女巫", data.get("thought", ""), "；".join(parts))
        return saved, poison_target

    # ── 发言 ──────────────────────────────────────────────

    def generate_speech(self, all_players: list, day_number: int,
                        night_event: str, speeches_so_far: list[dict]) -> str:
        alive_list = self._alive_list(all_players)
        role_cn = self._role_cn()
        system = prompts.SPEECH_SYSTEM.format(role_name=role_cn, camp=self.camp)
        user = prompts.build_speech_user(
            self.player_id, self.name, role_cn, self.camp,
            day_number, alive_list, night_event, speeches_so_far,
        )
        try:
            resp = call_llm(system, user + "\n\n" + prompts.SPEECH_OUTPUT_EXAMPLE)
            data = _extract_json(resp)
            speech = str(data.get("speech", ""))
        except Exception:
            speech = "我没什么好说的。"

        self._log(f"第{day_number}天-发言", data.get("thought", "") if 'data' in dir() else "",
                  f"发言: {speech}")
        return speech

    # ── 投票 ──────────────────────────────────────────────

    def decide_vote(self, all_players: list, day_number: int,
                    speeches: list[dict]) -> int:
        alive_list = self._alive_list(all_players)
        role_cn = self._role_cn()
        system = prompts.VOTE_SYSTEM.format(role_name=role_cn, camp=self.camp)
        user = prompts.build_vote_user(
            self.player_id, self.name, role_cn, self.camp,
            day_number, alive_list, speeches,
        )
        try:
            resp = call_llm(system, user + "\n\n" + prompts.VOTE_OUTPUT_EXAMPLE)
            data = _extract_json(resp)
            vote = int(data["vote_player_id"])
            valid = {p.player_id for p in all_players
                     if p.is_alive and p.player_id != self.player_id}
            if vote not in valid:
                vote = next(iter(valid))
        except Exception:
            alive_others = [p for p in all_players
                            if p.is_alive and p.player_id != self.player_id]
            vote = alive_others[0].player_id if alive_others else self.player_id

        target_name = next((p.name for p in all_players if p.player_id == vote), str(vote))
        self._log(f"第{day_number}天-投票", data.get("thought", "") if 'data' in dir() else "",
                  f"投票放逐 玩家{vote}({target_name})")
        return vote
