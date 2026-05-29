"""Integration tests for the Werewolf API endpoints."""

from fastapi.testclient import TestClient
from api.server import app

client = TestClient(app)


class TestHealthCheck:
    def test_health(self):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "werewolf-game-api"


class TestConfigs:
    def test_list_configs(self):
        resp = client.get("/configs")
        assert resp.status_code == 200
        configs = resp.json()
        assert len(configs) >= 3
        names = [c["name"] for c in configs]
        assert "standard_6" in names
        assert "simple_4" in names
        assert "big_9" in names

    def test_config_has_description(self):
        resp = client.get("/configs")
        for cfg in resp.json():
            assert "description" in cfg
            assert len(cfg["description"]) > 0


class TestCreateGame:
    def test_create_standard_6(self):
        resp = client.post("/games", json={"config_name": "standard_6", "use_llm": False})
        assert resp.status_code == 200
        data = resp.json()
        assert "game_id" in data
        assert data["alive_count"] == 6
        assert data["day_number"] == 1
        assert not data["is_game_over"]
        assert data["phase"] == "not_started"

    def test_create_simple_4(self):
        resp = client.post("/games", json={"config_name": "simple_4", "use_llm": False})
        assert resp.status_code == 200
        assert resp.json()["alive_count"] == 4

    def test_create_big_9(self):
        resp = client.post("/games", json={"config_name": "big_9", "use_llm": False})
        assert resp.status_code == 200
        assert resp.json()["alive_count"] == 9

    def test_create_with_custom_names(self):
        names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
        resp = client.post("/games", json={
            "config_name": "standard_6",
            "player_names": names,
            "shuffle": False,
            "use_llm": False,
        })
        assert resp.status_code == 200

    def test_create_without_shuffle(self):
        resp = client.post("/games", json={
            "config_name": "standard_6",
            "shuffle": False,
            "use_llm": False,
        })
        assert resp.status_code == 200

    def test_create_invalid_config(self):
        resp = client.post("/games", json={"config_name": "invalid_config", "use_llm": False})
        assert resp.status_code == 400

    def test_create_with_wrong_name_count(self):
        resp = client.post("/games", json={
            "config_name": "standard_6",
            "player_names": ["a", "b"],
            "use_llm": False,
        })
        assert resp.status_code == 400


class TestStepGame:
    def test_step_advances_phase(self):
        resp = client.post("/games", json={"config_name": "simple_4", "use_llm": False})
        gid = resp.json()["game_id"]

        step1 = client.post(f"/games/{gid}/step")
        assert step1.status_code == 200
        assert step1.json()["phase"] == "night_wolf"

        step2 = client.post(f"/games/{gid}/step")
        assert step2.status_code == 200
        assert step2.json()["phase"] == "night_seer"

    def test_step_returns_all_fields(self):
        resp = client.post("/games", json={"config_name": "simple_4", "use_llm": False})
        gid = resp.json()["game_id"]

        result = client.post(f"/games/{gid}/step").json()
        required = {"phase", "day_number", "step_data", "players", "dialogues", "deaths", "is_game_over", "winner"}
        assert required.issubset(result.keys())

    def test_step_players_contain_role_info(self):
        resp = client.post("/games", json={"config_name": "simple_4", "use_llm": False})
        gid = resp.json()["game_id"]

        result = client.post(f"/games/{gid}/step").json()
        for p in result["players"]:
            assert "player_id" in p
            assert "name" in p
            assert "role" in p
            assert "camp" in p
            assert "is_alive" in p

    def test_step_nonexistent_game(self):
        resp = client.post("/games/nonexistent/step")
        assert resp.status_code == 404

    def test_step_until_game_over(self):
        resp = client.post("/games", json={"config_name": "simple_4", "use_llm": False})
        gid = resp.json()["game_id"]

        for _ in range(50):
            result = client.post(f"/games/{gid}/step").json()
            if result["is_game_over"]:
                assert result["winner"] in ("good", "evil")
                break
        else:
            assert False, "Game did not finish within 50 steps"

    def test_step_after_game_over(self):
        resp = client.post("/games", json={"config_name": "simple_4", "use_llm": False})
        gid = resp.json()["game_id"]

        # Play until game over
        for _ in range(50):
            result = client.post(f"/games/{gid}/step").json()
            if result["is_game_over"]:
                break

        # Should keep returning game_over
        result2 = client.post(f"/games/{gid}/step").json()
        assert result2["phase"] == "game_over"
        assert result2["is_game_over"]
        assert result2["winner"] == result["winner"]


class TestGetGameState:
    def test_get_state(self):
        resp = client.post("/games", json={"config_name": "standard_6", "use_llm": False})
        gid = resp.json()["game_id"]

        state = client.get(f"/games/{gid}").json()
        assert state["game_id"] == gid
        assert state["config_name"] == "standard_6"
        assert len(state["players"]) == 6

    def test_get_state_after_steps(self):
        resp = client.post("/games", json={"config_name": "simple_4", "use_llm": False})
        gid = resp.json()["game_id"]

        client.post(f"/games/{gid}/step")  # night_wolf
        state = client.get(f"/games/{gid}").json()
        assert state["phase"] == "night_wolf"
        assert state["alive_count"] == 4

    def test_get_nonexistent_game(self):
        resp = client.get("/games/nonexistent")
        assert resp.status_code == 404


class TestListGames:
    def test_list_empty(self):
        # Delete all games first
        games = client.get("/games").json()
        for g in games:
            client.delete(f"/games/{g['game_id']}")

        resp = client.get("/games")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_with_games(self):
        client.post("/games", json={"config_name": "simple_4", "use_llm": False})
        client.post("/games", json={"config_name": "standard_6", "use_llm": False})

        resp = client.get("/games")
        data = resp.json()
        assert len(data) >= 2
        for g in data:
            assert "game_id" in g
            assert "config_name" in g
            assert "phase" in g
            assert "alive_count" in g


class TestDeleteGame:
    def test_delete_existing(self):
        resp = client.post("/games", json={"config_name": "simple_4", "use_llm": False})
        gid = resp.json()["game_id"]

        resp = client.delete(f"/games/{gid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Verify game is gone
        assert client.get(f"/games/{gid}").status_code == 404

    def test_delete_nonexistent(self):
        resp = client.delete("/games/nonexistent")
        assert resp.status_code == 404

    def test_delete_and_recreate(self):
        resp = client.post("/games", json={"config_name": "simple_4", "use_llm": False})
        gid = resp.json()["game_id"]

        client.delete(f"/games/{gid}")
        # Same ID should no longer exist
        assert client.get(f"/games/{gid}").status_code == 404
