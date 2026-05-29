"""Pydantic request/response models for the Werewolf API."""

from typing import Optional
from pydantic import BaseModel, Field


class CreateGameRequest(BaseModel):
    config_name: str = Field("standard_6", description="角色配置名")
    player_names: Optional[list[str]] = Field(None, description="自定义玩家名称列表")
    shuffle: bool = Field(True, description="是否随机打乱角色分配")
    use_llm: bool = Field(True, description="是否使用大模型驱动AI玩家（需要设置 DASHSCOPE_API_KEY）")


class CreateGameResponse(BaseModel):
    game_id: str
    phase: str
    day_number: int
    alive_count: int
    is_game_over: bool


class StepResponse(BaseModel):
    phase: str
    day_number: int
    step_data: dict
    players: list[dict]
    dialogues: list[dict]
    deaths: list[dict]
    is_game_over: bool
    winner: Optional[str]


class GameStateResponse(BaseModel):
    game_id: str
    config_name: str
    phase: str
    day_number: int
    players: list[dict]
    alive_count: int
    is_game_over: bool
    winner: Optional[str]
    deaths: list[dict]
    dialogues: list[dict]


class GameSummary(BaseModel):
    game_id: str
    config_name: str
    phase: str
    day_number: int
    alive_count: int
    is_game_over: bool
    winner: Optional[str]


class ConfigInfo(BaseModel):
    name: str
    description: str
