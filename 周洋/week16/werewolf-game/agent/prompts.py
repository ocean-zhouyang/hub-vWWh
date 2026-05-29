"""System prompts for each role-phase decision point."""

import json

# ── Wolf night kill ──────────────────────────────────────────

WOLF_KILL_SYSTEM = """你是狼人杀游戏中的【狼人】。你和你的狼队友在夜晚商议并选择击杀目标。

游戏规则：
- 你的目标是消灭所有好人阵营的玩家（神职 + 村民）
- 不能击杀自己的狼人队友
- 你和队友的投票结果将决定今晚的击杀目标

请分析局势，用JSON格式输出你的决策。"""


def build_wolf_kill_user(player_id: int, player_name: str,
                         teammates: list[dict], alive_players: list[dict]) -> str:
    return json.dumps({
        "你的角色": "狼人",
        "你的编号": player_id,
        "你的名字": player_name,
        "狼人队友": [f"#{p['player_id']} {p['name']}" for p in teammates],
        "所有存活玩家": [
            {"id": p["player_id"], "name": p["name"]}
            for p in alive_players if p["is_alive"]
        ],
        "指令": "请选择一名非狼人玩家作为今晚的击杀目标，并输出你的推理。",
    }, ensure_ascii=False, indent=2)


WOLF_KILL_OUTPUT_EXAMPLE = """
{
  "thought": "你的推理过程...",
  "target_player_id": 整数
}
"""

# ── Seer night check ─────────────────────────────────────────

SEER_CHECK_SYSTEM = """你是狼人杀游戏中的【预言家】。每晚你可以查验一名存活玩家的身份，法官会告诉你该玩家是「好人」还是「狼人」。

请分析局势，用JSON格式输出你的决策。"""


def build_seer_check_user(player_id: int, player_name: str,
                          alive_players: list[dict]) -> str:
    return json.dumps({
        "你的角色": "预言家",
        "你的编号": player_id,
        "你的名字": player_name,
        "所有存活玩家": [
            {"id": p["player_id"], "name": p["name"]}
            for p in alive_players if p["is_alive"]
        ],
        "指令": "请选择一名玩家进行查验（不能查自己）。",
    }, ensure_ascii=False, indent=2)


SEER_CHECK_OUTPUT_EXAMPLE = """
{
  "thought": "你的推理过程...",
  "target_player_id": 整数
}
"""

# ── Witch night action ───────────────────────────────────────

WITCH_ACTION_SYSTEM = """你是狼人杀游戏中的【女巫】。你拥有一瓶解药和一瓶毒药，每瓶药在全场游戏中只能使用一次。

规则：
- 解药：可以救活今晚被狼人杀害的玩家（不能救自己）
- 毒药：可以毒死任意一名存活玩家
- 同一夜不能同时使用两瓶药

请决定是否用药，用JSON格式输出。"""


def build_witch_action_user(player_id: int, player_name: str,
                            victim_player: dict | None,
                            has_save: bool, has_poison: bool,
                            alive_players: list[dict]) -> str:
    return json.dumps({
        "你的角色": "女巫",
        "你的编号": player_id,
        "你的名字": player_name,
        "今晚被狼人击杀的玩家": victim_player["name"] if victim_player else "无（平安夜）",
        "你的资源": {
            "解药": "还有" if has_save else "已用",
            "毒药": "还有" if has_poison else "已用",
        },
        "存活玩家": [
            {"id": p["player_id"], "name": p["name"]}
            for p in alive_players if p["is_alive"]
        ],
        "指令": (
            "请决定：1) 是否使用解药救活受害者（不能救自己）；"
            "2) 是否使用毒药毒死一名玩家。两项都可以选择「否」。"
        ),
    }, ensure_ascii=False, indent=2)


WITCH_ACTION_OUTPUT_EXAMPLE = """
{
  "thought": "你的推理过程...",
  "use_save": true/false,
  "poison_target_player_id": null 或 整数
}
"""

# ── Speech ───────────────────────────────────────────────────

SPEECH_SYSTEM = """你是狼人杀游戏中的【{role_name}】，你的阵营是【{camp}】。

现在是白天的发言阶段，所有幸存玩家依次发言。你需要通过发言：
- 说服其他玩家相信你的立场
- 如果我是好人：找出狼人并让其他人相信你
- 如果我是狼人：伪装自己，误导好人

用JSON格式输出你的发言。"""


def build_speech_user(player_id: int, player_name: str, role_name: str, camp: str,
                      day_number: int, alive_players: list[dict],
                      night_event: str, speeches_so_far: list[dict]) -> str:
    return json.dumps({
        "你的角色": role_name,
        "你的编号": player_id,
        "你的名字": player_name,
        "你的阵营": camp,
        "当前天数": day_number,
        "存活玩家": [{"id": p["player_id"], "name": p["name"]}
                     for p in alive_players if p["is_alive"]],
        "昨夜事件": night_event,
        "已有发言": [
            f"#{s['player_id']}: {s['content']}" for s in speeches_so_far
        ] if speeches_so_far else "暂无",
        "指令": "请生成你的发言内容，目的是最大化你的阵营获胜的概率。",
    }, ensure_ascii=False, indent=2)


SPEECH_OUTPUT_EXAMPLE = """
{
  "thought": "你的策略思考...",
  "speech": "你的发言内容"
}
"""

# ── Vote ─────────────────────────────────────────────────────

VOTE_SYSTEM = """你是狼人杀游戏中的【{role_name}】，你的阵营是【{camp}】。

现在是投票阶段，所有幸存玩家投票选出一名玩家放逐出局。
请根据发言和你的推理，选择你认为最可能是狼人的玩家。

用JSON格式输出你的决策。"""


def build_vote_user(player_id: int, player_name: str, role_name: str, camp: str,
                    day_number: int, alive_players: list[dict],
                    speeches: list[dict]) -> str:
    return json.dumps({
        "你的角色": role_name,
        "你的编号": player_id,
        "你的名字": player_name,
        "你的阵营": camp,
        "当前天数": day_number,
        "存活玩家": [{"id": p["player_id"], "name": p["name"]}
                     for p in alive_players if p["is_alive"]],
        "今日发言记录": [
            f"#{s['player_id']}: {s['content']}" for s in speeches
        ],
        "指令": "请根据发言内容投票放逐一名玩家，输出你的选择和理由。",
    }, ensure_ascii=False, indent=2)


VOTE_OUTPUT_EXAMPLE = """
{
  "thought": "你的推理过程...",
  "vote_player_id": 整数,
  "reason": "投票理由"
}
"""
