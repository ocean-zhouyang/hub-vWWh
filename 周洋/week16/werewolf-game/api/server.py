"""FastAPI server for the Werewolf game — HTTP API to create, control, and observe games."""

from fastapi import FastAPI, HTTPException
from .models import CreateGameRequest, GameStateResponse, GameSummary, ConfigInfo
from game_engine import GameManager
from game_engine.configs import get_available_configs
from configs.llm_config import is_llm_available

app = FastAPI(
    title="Werewolf Game API",
    description="多智能体狼人杀游戏服务器 — 创建、控制和观察对局",
    version="1.0.0",
)


@app.get("/")
def health_check():
    """服务健康检查"""
    return {"status": "ok", "service": "werewolf-game-api"}


@app.post("/games")
def create_game(body: CreateGameRequest):
    """创建并初始化一局新游戏"""
    if body.use_llm and not is_llm_available():
        raise HTTPException(
            status_code=400,
            detail="LLM 模式需要设置 DASHSCOPE_API_KEY 环境变量（使用通义千问 qwen-flash）",
        )

    try:
        engine = GameManager.create_game(
            config_name=body.config_name,
            player_names=body.player_names,
            shuffle=body.shuffle,
            use_llm=body.use_llm,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "game_id": engine.game_id,
        "phase": engine.get_phase_name(),
        "day_number": engine.day_number,
        "alive_count": engine.get_alive_count(),
        "is_game_over": engine.is_game_over,
    }


@app.post("/games/{game_id}/step")
def step_game(game_id: str):
    """推进一个游戏阶段"""
    engine = GameManager.get_game(game_id)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Game '{game_id}' not found")

    result = engine.step()
    return result


@app.get("/games/{game_id}")
def get_game_state(game_id: str):
    """获取当前游戏完整状态"""
    engine = GameManager.get_game(game_id)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Game '{game_id}' not found")

    state = engine.get_state()
    return state


@app.get("/games")
def list_all_games():
    """列出所有活跃游戏"""
    return GameManager.list_games()


@app.delete("/games/{game_id}")
def delete_game(game_id: str):
    """删除一局游戏"""
    if not GameManager.delete_game(game_id):
        raise HTTPException(status_code=404, detail=f"Game '{game_id}' not found")
    return {"status": "deleted", "game_id": game_id}


@app.get("/configs")
def list_configs():
    """列出可用角色配置"""
    return get_available_configs()
