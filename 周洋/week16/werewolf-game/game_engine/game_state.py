"""Game state machine — drives day/night cycles, collects decisions, checks win condition."""

import random
import uuid
from collections import Counter
from typing import Optional

from .configs import get_config, ROLE_NAMES

PHASE_ORDER = [
    "night_wolf",
    "night_seer",
    "night_witch",
    "night_result",
    "day_start",
    "speech",
    "vote",
    "day_end",
]

GOD_ROLES = {"seer", "witch", "hunter"}


class Player:
    def __init__(self, player_id: int, name: str, role: str):
        self.player_id = player_id
        self.name = name
        self.role = role
        self.camp = "evil" if role == "werewolf" else "good"
        self.is_alive = True

        # Witch
        self.has_save_potion = True
        self.has_poison_potion = True

        # Hunter
        self.hunter_can_shoot = True  # False if killed by poison

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "role": self.role,
            "role_name": ROLE_NAMES.get(self.role, self.role),
            "camp": self.camp,
            "is_alive": self.is_alive,
        }


class GameEngine:
    """Drives a single game instance with a phase-based state machine."""

    def __init__(
        self,
        config_name: str = "standard_6",
        player_names: Optional[list[str]] = None,
        shuffle: bool = True,
        use_llm: bool = False,
        agent_manager=None,
    ):
        self.game_id = str(uuid.uuid4())[:8]
        self.config_name = config_name
        self.config = get_config(config_name)

        roles = list(self.config["roles"])
        if shuffle:
            random.shuffle(roles)

        count = len(roles)
        if player_names is None:
            player_names = [f"玩家{i+1}" for i in range(count)]
        elif len(player_names) != count:
            raise ValueError(
                f"Expected {count} player names, got {len(player_names)}"
            )

        self.players = [Player(i, player_names[i], roles[i]) for i in range(count)]
        self.current_phase_index = -1  # not started
        self.day_number = 1
        self.is_game_over = False
        self.winner: Optional[str] = None

        # --- LLM agent integration ---
        self.use_llm = use_llm
        self.agent_manager = agent_manager

        # --- Night bookkeeping ---
        self.wolf_votes: list[dict] = []
        self.wolf_target: Optional[int] = None
        self.seer_target: Optional[int] = None
        self.seer_result: Optional[str] = None
        self.witch_saved = False
        self.witch_poison_target: Optional[int] = None
        self.night_deaths: list[int] = []
        self.night_death_causes: dict[int, str] = {}

        # --- Day bookkeeping ---
        self.speeches: dict[int, str] = {}
        self.votes: dict[int, int] = {}
        self.day_eliminated: Optional[int] = None
        self.day_hunter_shot: Optional[int] = None

        # --- Log ---
        self.dialogues: list[dict] = []
        self.deaths: list[dict] = []

    # ── helpers ──────────────────────────────────────────────

    def _alive(self):
        return [p for p in self.players if p.is_alive]

    def _alive_with_role(self, role: str):
        return [p for p in self.players if p.is_alive and p.role == role]

    def _check_win(self) -> tuple[bool, Optional[str]]:
        alive_wolves = len(self._alive_with_role("werewolf"))
        alive_gods = len(
            [p for p in self._alive() if p.role in GOD_ROLES]
        )
        alive_villagers = len(self._alive_with_role("villager"))

        if alive_wolves == 0:
            return True, "good"
        if alive_gods == 0 or alive_villagers == 0:
            return True, "evil"
        return False, None

    def _maybe_end_game(self):
        over, winner = self._check_win()
        if over:
            self.is_game_over = True
            self.winner = winner

    def _add_death(self, phase: str, player_id: int, cause: str):
        p = self.players[player_id]
        p.is_alive = False
        entry = {
            "phase": phase,
            "player_id": player_id,
            "player_name": p.name,
            "cause": cause,
        }
        self.deaths.append(entry)
        return entry

    def _random_alive_other(self, exclude_id: int):
        others = [p for p in self._alive() if p.player_id != exclude_id]
        return random.choice(others) if others else None

    # ── night phases ─────────────────────────────────────────

    def _exec_wolf(self) -> dict:
        self.wolf_votes = []
        self.wolf_target = None

        if self.use_llm and self.agent_manager:
            self.wolf_votes, self.wolf_target = self.agent_manager.decide_wolf_kill(self.players)
        else:
            wolves = self._alive_with_role("werewolf")
            targets = [p for p in self._alive() if p.role != "werewolf"]
            if wolves and targets:
                for wolf in wolves:
                    self.wolf_votes.append(
                        {"player_id": wolf.player_id, "target": random.choice(targets).player_id}
                    )
                votes = [v["target"] for v in self.wolf_votes]
                cnt = Counter(votes)
                top = max(cnt.values())
                self.wolf_target = random.choice([t for t, c in cnt.items() if c == top])

        return {
            "wolf_votes": self.wolf_votes,
            "final_target": self.wolf_target,
        }

    def _exec_seer(self) -> dict:
        self.seer_target = None
        self.seer_result = None

        if self.use_llm and self.agent_manager:
            self.seer_target, self.seer_result = self.agent_manager.decide_seer_check(self.players)
        else:
            seers = self._alive_with_role("seer")
            if seers:
                target = self._random_alive_other(seers[0].player_id)
                if target:
                    self.seer_target = target.player_id
                    self.seer_result = "wolf" if target.role == "werewolf" else "good"

        return {
            "seer_target": self.seer_target,
            "result": self.seer_result,
        }

    def _exec_witch(self) -> dict:
        self.witch_saved = False
        self.witch_poison_target = None

        if self.use_llm and self.agent_manager:
            self.witch_saved, self.witch_poison_target = \
                self.agent_manager.decide_witch_action(self.players, self.wolf_target)
        else:
            witches = self._alive_with_role("witch")
            if witches:
                witch = witches[0]
                if (
                    self.wolf_target is not None
                    and self.wolf_target != witch.player_id
                    and witch.has_save_potion
                ):
                    self.witch_saved = True
                    witch.has_save_potion = False

                if witch.has_poison_potion and random.random() < 0.3:
                    target = self._random_alive_other(witch.player_id)
                    if target:
                        self.witch_poison_target = target.player_id
                        witch.has_poison_potion = False

        return {
            "saved": self.witch_saved,
            "poison_target": self.witch_poison_target,
        }

    def _exec_night_result(self) -> dict:
        self.night_deaths = []
        self.night_death_causes = {}
        hunter_shot = None

        # Wolf kill
        if self.wolf_target is not None and not self.witch_saved:
            self.night_deaths.append(self.wolf_target)
            self.night_death_causes[self.wolf_target] = "wolf_kill"

        # Witch poison
        if self.witch_poison_target is not None:
            self.night_deaths.append(self.witch_poison_target)
            self.night_death_causes[self.witch_poison_target] = "poison"

        self.night_deaths = list(set(self.night_deaths))

        for pid in self.night_deaths:
            cause = self.night_death_causes.get(pid, "unknown")
            p = self.players[pid]
            if cause == "poison":
                p.hunter_can_shoot = False
            self._add_death("night", pid, cause)

        # Hunter retaliation (night)
        for pid in self.night_deaths:
            p = self.players[pid]
            if p.role == "hunter" and p.hunter_can_shoot:
                target = self._random_alive_other(pid)
                if target:
                    hunter_shot = target.player_id
                    self._add_death("hunter_retaliation", hunter_shot, "hunter_shot")

        self._maybe_end_game()

        return {
            "deaths": self.night_deaths,
            "death_causes": dict(self.night_death_causes),
            "hunter_shot": hunter_shot,
        }

    # ── day phases ───────────────────────────────────────────

    def _exec_day_start(self) -> dict:
        night_dead_ids = {d["player_id"] for d in self.deaths if d["phase"] in ("night", "hunter_retaliation")}
        deceased = [p for p in self.players if not p.is_alive and p.player_id in night_dead_ids]

        if deceased:
            names = "、".join(p.name for p in deceased)
            announcement = f"天亮了，昨晚{names}死了。"
        else:
            announcement = "天亮了，昨晚是平安夜。"

        return {
            "announcement": announcement,
            "deceased": [p.player_id for p in deceased],
        }

    def _build_night_event_summary(self) -> str:
        """Build a short summary of what happened tonight for the speech phase."""
        if not self.night_deaths:
            return "昨晚是平安夜，没有人死亡。"
        names = [self.players[pid].name for pid in self.night_deaths]
        return f"昨晚{'、'.join(names)}死了。"

    def _exec_speech(self) -> dict:
        self.speeches = {}
        night_event = self._build_night_event_summary()

        if self.use_llm and self.agent_manager:
            for player in self._alive():
                speeches_so_far = [
                    {"player_id": pid, "content": text}
                    for pid, text in self.speeches.items()
                ]
                text = self.agent_manager.decide_speech(
                    player.player_id, self.players,
                    self.day_number, night_event, speeches_so_far,
                )
                self.speeches[player.player_id] = text
        else:
            templates = {
                "werewolf": [
                    "我是一只好人，大家不要怀疑我。",
                    "我觉得{x}号很可疑，他的发言有问题。",
                    "昨晚我一直在思考，{x}号的行为不像好人。",
                ],
                "seer": [
                    "我是预言家，昨晚查验了{x}号，他是{rst}。",
                    "我昨晚查了{x}号，{rst}，请大家相信我。",
                ],
                "witch": [
                    "昨晚我救了一个人，具体是谁不方便说。",
                    "我有一些信息，但现在不方便透露太多。",
                ],
                "hunter": [
                    "我是猎人，谁投我谁就是狼人。",
                    "大家冷静分析，不要被狼人带节奏。",
                ],
                "villager": [
                    "我是村民，没什么特殊信息，听大家分析。",
                    "我觉得{x}号发言有问题，可能是狼人。",
                    "大家理性投票，不要跟风。",
                ],
            }
            for player in self._alive():
                tpl = random.choice(templates.get(player.role, templates["villager"]))
                ref = self._random_alive_other(player.player_id)
                if ref is None:
                    ref = player
                rst = "好人" if ref.role != "werewolf" else "狼人"
                self.speeches[player.player_id] = tpl.format(x=ref.player_id, rst=rst)

        return {
            "speeches": [
                {"player_id": pid, "content": text}
                for pid, text in self.speeches.items()
            ]
        }

    def _exec_vote(self) -> dict:
        self.votes = {}
        self.day_eliminated = None
        self.day_hunter_shot = None
        alive_players = self._alive()

        if self.use_llm and self.agent_manager:
            speeches_list = [
                {"player_id": pid, "content": text}
                for pid, text in self.speeches.items()
            ]
            for player in alive_players:
                vote = self.agent_manager.decide_vote(
                    player.player_id, self.players,
                    self.day_number, speeches_list,
                )
                if vote != player.player_id:
                    self.votes[player.player_id] = vote
        else:
            for player in alive_players:
                others = [p for p in alive_players if p.player_id != player.player_id]
                if others:
                    self.votes[player.player_id] = random.choice(others).player_id

        # Tally
        cnt = Counter(self.votes.values())
        if cnt:
            max_votes = max(cnt.values())
            top = [t for t, c in cnt.items() if c == max_votes]
            if len(top) == 1:
                eliminated_id = top[0]
                self.day_eliminated = eliminated_id
                eliminated = self.players[eliminated_id]
                eliminated.is_alive = False
                self.dialogues.append({
                    "type": "vote_elimination",
                    "player_id": eliminated_id,
                    "player_name": eliminated.name,
                    "votes": dict(self.votes),
                })
                self._add_death("vote", eliminated_id, "vote")

                # Hunter retaliation from day elimination
                if eliminated.role == "hunter" and eliminated.hunter_can_shoot:
                    target = self._random_alive_other(eliminated_id)
                    if target:
                        self.day_hunter_shot = target.player_id
                        self._add_death("hunter_retaliation", target.player_id, "hunter_shot")

        return {
            "votes": [{"voter": v, "target": t} for v, t in self.votes.items()],
            "eliminated": self.day_eliminated,
            "tie": self.day_eliminated is None and bool(self.votes),
            "hunter_shot": self.day_hunter_shot,
        }

    def _exec_day_end(self) -> dict:
        self._maybe_end_game()
        return {
            "eliminated_player": self.day_eliminated,
            "is_game_over": self.is_game_over,
            "winner": self.winner,
            "next_day": self.day_number + 1 if not self.is_game_over else None,
        }

    # ── public API ───────────────────────────────────────────

    def get_phase_name(self) -> str:
        if self.is_game_over:
            return "game_over"
        if self.current_phase_index < 0:
            return "not_started"
        return PHASE_ORDER[self.current_phase_index]

    def get_alive_count(self) -> int:
        return len(self._alive())

    def step(self) -> dict:
        """Advance one phase and return results."""
        if self.is_game_over:
            return self._build_response("game_over", {})

        self.current_phase_index += 1
        if self.current_phase_index >= len(PHASE_ORDER):
            self.current_phase_index = 0
            self.day_number += 1

        phase = PHASE_ORDER[self.current_phase_index]

        handlers = {
            "night_wolf": self._exec_wolf,
            "night_seer": self._exec_seer,
            "night_witch": self._exec_witch,
            "night_result": self._exec_night_result,
            "day_start": self._exec_day_start,
            "speech": self._exec_speech,
            "vote": self._exec_vote,
            "day_end": self._exec_day_end,
        }

        step_data = handlers[phase]()
        return self._build_response(phase, step_data)

    def get_state(self) -> dict:
        """Return current state without advancing."""
        phase = (
            PHASE_ORDER[self.current_phase_index]
            if self.current_phase_index >= 0
            else "not_started"
        )
        if self.is_game_over:
            phase = "game_over"

        return {
            "game_id": self.game_id,
            "config_name": self.config_name,
            "phase": phase,
            "day_number": self.day_number,
            "players": [p.to_dict() for p in self.players],
            "alive_count": len(self._alive()),
            "is_game_over": self.is_game_over,
            "winner": self.winner,
            "deaths": list(self.deaths),
            "dialogues": list(self.dialogues[-20:]),
        }

    def _build_response(self, phase: str, step_data: dict) -> dict:
        return {
            "phase": phase,
            "day_number": self.day_number,
            "step_data": step_data,
            "players": [p.to_dict() for p in self.players],
            "dialogues": list(self.dialogues[-10:]),
            "deaths": list(self.deaths[-5:]),
            "is_game_over": self.is_game_over,
            "winner": self.winner,
        }
