"""Game role configurations for different player counts."""

ROLE_CONFIGS = {
    "standard_6": {
        "name": "standard_6",
        "description": "标准6人局: 2狼人 + 1预言家 + 1女巫 + 1猎人 + 1村民",
        "roles": ["werewolf", "werewolf", "seer", "witch", "hunter", "villager"],
    },
    "simple_4": {
        "name": "simple_4",
        "description": "简易4人局: 1狼人 + 1预言家 + 1女巫 + 1村民",
        "roles": ["werewolf", "seer", "witch", "villager"],
    },
    "big_9": {
        "name": "big_9",
        "description": "标准9人局: 3狼人 + 1预言家 + 1女巫 + 1猎人 + 3村民",
        "roles": [
            "werewolf", "werewolf", "werewolf",
            "seer", "witch", "hunter",
            "villager", "villager", "villager",
        ],
    },
}

ROLE_NAMES = {
    "werewolf": "狼人",
    "seer": "预言家",
    "witch": "女巫",
    "hunter": "猎人",
    "villager": "村民",
}

ROLE_CAMP = {
    "werewolf": "evil",
    "seer": "good",
    "witch": "good",
    "hunter": "good",
    "villager": "good",
}


def get_config(name: str) -> dict:
    if name not in ROLE_CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Available: {list(ROLE_CONFIGS.keys())}")
    return ROLE_CONFIGS[name]


def get_available_configs() -> list:
    return [
        {"name": name, "description": cfg["description"]}
        for name, cfg in ROLE_CONFIGS.items()
    ]