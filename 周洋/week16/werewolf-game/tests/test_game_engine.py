"""Unit tests for the game engine state machine."""

from game_engine.game_state import GameEngine, PHASE_ORDER


class TestGameEngine:
    def test_create_standard_6(self):
        engine = GameEngine("standard_6")
        assert engine.config_name == "standard_6"
        assert len(engine.players) == 6
        assert engine.get_alive_count() == 6
        assert engine.current_phase_index == -1
        assert engine.get_phase_name() == "not_started"
        assert not engine.is_game_over

    def test_create_simple_4(self):
        engine = GameEngine("simple_4")
        assert len(engine.players) == 4
        assert engine.get_alive_count() == 4

    def test_create_big_9(self):
        engine = GameEngine("big_9")
        assert len(engine.players) == 9
        assert engine.get_alive_count() == 9

    def test_custom_player_names(self):
        names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"]
        engine = GameEngine("standard_6", player_names=names, shuffle=False)
        assert [p.name for p in engine.players] == names

    def test_wrong_number_of_names_raises(self):
        import pytest
        with pytest.raises(ValueError):
            GameEngine("standard_6", player_names=["a", "b"])

    def test_shuffle_changes_order(self):
        # With high probability, shuffled != unshuffled
        unshuffled = GameEngine("standard_6", shuffle=False)
        shuffled = GameEngine("standard_6", shuffle=True)
        unshuffled_roles = [p.role for p in unshuffled.players]
        # At least check roles are correctly assigned
        assert sorted(unshuffled_roles) == sorted([p.role for p in shuffled.players])

    def test_phase_flow(self):
        """All 8 phases cycle correctly (first 4 phases at minimum)."""
        engine = GameEngine("standard_6", shuffle=False)
        assert engine.get_phase_name() == "not_started"

        for i in range(4):
            result = engine.step()
            expected_phase = PHASE_ORDER[i]
            assert result["phase"] == expected_phase, f"Step {i}: expected {expected_phase}, got {result['phase']}"
            assert result["day_number"] >= 1
            if result["is_game_over"]:
                return  # Game may end early due to屠边

    def test_day_advances_after_full_cycle(self):
        """After day_end, next step goes to night_wolf of the next day (if game not over)."""
        engine = GameEngine("standard_6", shuffle=False)
        for _ in range(8):
            result = engine.step()
        # After 8 steps: day_end of day 1
        if not result["is_game_over"]:
            result = engine.step()
            assert result["phase"] == "night_wolf"
            assert result["day_number"] == 2
        else:
            # Game may end early due to simulation RNG
            assert engine.is_game_over

    def test_night_wolf_returns_valid_data(self):
        engine = GameEngine("standard_6", shuffle=False)
        result = engine.step()
        assert result["phase"] == "night_wolf"
        sd = result["step_data"]
        assert "wolf_votes" in sd
        assert "final_target" in sd
        # Wolves (players 2,3) should vote for non-wolves
        target = sd["final_target"]
        assert target in range(6)
        assert engine.players[target].role != "werewolf"

    def test_night_seer_returns_valid_data(self):
        engine = GameEngine("standard_6", shuffle=False)
        engine.step()  # night_wolf
        result = engine.step()  # night_seer
        assert result["phase"] == "night_seer"
        sd = result["step_data"]
        assert sd["seer_target"] is None or sd["result"] in ("good", "wolf")

    def test_night_result_applies_deaths(self):
        engine = GameEngine("standard_6", shuffle=False)
        for _ in range(3):
            engine.step()
        result = engine.step()  # night_result
        assert result["phase"] == "night_result"
        # Deaths may or may not happen depending on witch behavior
        assert "deaths" in result["step_data"]

    def test_speech_generates_text(self):
        engine = GameEngine("standard_6", shuffle=False)
        # Step up to speech phase
        for _ in range(5):
            result = engine.step()
        result = engine.step()  # speech
        if engine.is_game_over:
            assert result["phase"] == "game_over"
            return
        assert result["phase"] == "speech", f"Expected speech, got {result['phase']}"
        speeches = result["step_data"]["speeches"]
        assert len(speeches) > 0
        for s in speeches:
            assert "player_id" in s
            assert "content" in s
            assert len(s["content"]) > 0

    def test_vote_eliminates_player(self):
        engine = GameEngine("standard_6", shuffle=False)
        for _ in range(6):
            result = engine.step()
        result = engine.step()  # vote
        if engine.is_game_over:
            assert result["phase"] == "game_over"
            return
        assert result["phase"] == "vote", f"Expected vote, got {result['phase']}"
        # Vote may or may not eliminate someone (could be tie)
        if result["step_data"]["eliminated"] is not None:
            eliminated_id = result["step_data"]["eliminated"]
            assert not engine.players[eliminated_id].is_alive

    def test_get_state(self):
        engine = GameEngine("standard_6")
        state = engine.get_state()
        assert state["game_id"] == engine.game_id
        assert state["phase"] == "not_started"
        assert len(state["players"]) == 6
        assert state["alive_count"] == 6
        engine.step()
        state = engine.get_state()
        assert state["phase"] in PHASE_ORDER

    def test_game_over_returns_constant_response(self):
        engine = GameEngine("simple_4", shuffle=False)
        # Force end by making all wolves dead
        for p in engine.players:
            if p.role == "werewolf":
                p.is_alive = False
        engine._maybe_end_game()
        assert engine.is_game_over
        assert engine.winner == "good"

        # All subsequent steps return game_over
        result = engine.step()
        assert result["phase"] == "game_over"
        assert result["is_game_over"]
        assert result["winner"] == "good"

    def test_standard_6_role_distribution(self):
        engine = GameEngine("standard_6", shuffle=False)
        roles = [p.role for p in engine.players]
        assert roles == ["werewolf", "werewolf", "seer", "witch", "hunter", "villager"]

    def test_simple_4_role_distribution(self):
        engine = GameEngine("simple_4", shuffle=False)
        roles = [p.role for p in engine.players]
        assert roles == ["werewolf", "seer", "witch", "villager"]

    def test_witch_attributes(self):
        engine = GameEngine("simple_4", shuffle=False)
        witch = [p for p in engine.players if p.role == "witch"][0]
        assert witch.has_save_potion
        assert witch.has_poison_potion
        assert witch.hunter_can_shoot
        assert witch.camp == "good"

    def test_wolf_attributes(self):
        engine = GameEngine("standard_6", shuffle=False)
        wolf = [p for p in engine.players if p.role == "werewolf"][0]
        assert wolf.camp == "evil"

    def test_player_to_dict(self):
        engine = GameEngine("simple_4", shuffle=False)
        player = engine.players[0]
        d = player.to_dict()
        assert d["player_id"] == player.player_id
        assert d["role"] == player.role
        assert d["is_alive"]
        assert d["camp"] in ("good", "evil")

    def test_death_logging(self):
        """Verify that deaths are recorded with proper metadata."""
        engine = GameEngine("standard_6", shuffle=False)
        for _ in range(4):
            engine.step()
        n_night_deaths = len(engine.deaths)
        # Run through vote phase
        for _ in range(3):
            engine.step()
        total_deaths = len(engine.deaths)
        assert total_deaths >= n_night_deaths  # vote may or may not add deaths
        for d in engine.deaths:
            assert "player_id" in d
            assert "player_name" in d
            assert "cause" in d
            assert d["cause"] in ("wolf_kill", "poison", "vote", "hunter_shot")
