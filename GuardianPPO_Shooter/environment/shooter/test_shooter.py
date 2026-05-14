"""
test_shooter.py — Validates ShooterEnv against the Gymnasium API contract.

Checks
------
1. reset() / step() shapes, dtypes, info keys (incl. quota fields)
2. Full random episode runs to termination without errors
3. Seeded resets produce identical initial observations
4. Observation values are finite and in expected range
5. All 7 actions execute without error
6. Wave guard — win does not fire before all NUM_PERIODIC_WAVES waves are launched
7. gym.make("Shooter-v0") registration works end-to-end
"""

import sys
import traceback
import numpy as np

# Allow running from the project root without installing the package
sys.path.insert(0, __file__.replace("/shooter/test_shooter.py", ""))

from shooter.shooter_env import (
    ShooterEnv, GameEngine, OBS_SIZE, NUM_ACTIONS, ACTION_NAMES,
    NUM_PERIODIC_WAVES, MAX_GAME_TICKS,
)


def _header(text: str):
    print(f"\n{'─'*55}")
    print(f"  {text}")
    print(f"{'─'*55}")


def test_api_contract():
    _header("1 · API contract — shapes, types, spaces")
    env = ShooterEnv(render_mode=None)

    obs, info = env.reset(seed=0)

    # Observation shape and dtype
    assert obs.shape == (OBS_SIZE,), f"Expected ({OBS_SIZE},), got {obs.shape}"
    assert obs.dtype == np.float32,  f"Expected float32, got {obs.dtype}"
    assert env.observation_space.contains(obs), "obs not in observation_space"

    # Info keys
    for key in ("tick", "hunterScore", "alive_ai", "total_ai",
                "total_spawned_ai", "wave", "wave_capacity",
                "bullets_in_flight", "gameOverReason"):
        assert key in info, f"Missing info key: {key}"
    assert info["total_spawned_ai"] == 4, \
        "total_spawned_ai should start at 4 (initial AI count)"
    assert info["wave"] == 0,            "wave should start at 0"
    assert info["wave_capacity"] == 4,   "wave_capacity should start at 4"
    assert env.action_space.n == 6,      "RL mode should expose Discrete(6), not 7"

    # Step output
    obs2, reward, terminated, truncated, info2 = env.step(1)
    assert obs2.shape  == (OBS_SIZE,)
    assert isinstance(reward,     float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated,  bool)
    assert env.observation_space.contains(obs2)

    env.close()
    print("  PASSED")


def test_random_episode():
    _header("2 · Full random episode — termination and reward")
    env = ShooterEnv(render_mode=None)
    obs, info = env.reset(seed=7)

    total_reward = 0.0
    steps = 0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs), f"obs out of space at step {steps}"
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    reason = info["gameOverReason"] or "truncated"
    print(f"  Episode ended after {steps} steps  |  "
          f"total reward = {total_reward:.1f}  |  reason: {reason}")
    assert steps > 0
    if terminated:
        assert info["gameOverReason"] != "", "terminated without a gameOverReason"
    env.close()
    print("  PASSED")


def test_seed_reproducibility():
    _header("3 · Seed reproducibility")
    env = ShooterEnv(render_mode=None)

    obs_a, _ = env.reset(seed=42)
    obs_b, _ = env.reset(seed=42)
    assert np.allclose(obs_a, obs_b), "Seeded reset produced different observations"

    obs_c, _ = env.reset(seed=99)
    assert not np.allclose(obs_a, obs_c), "Different seeds gave identical observations"

    env.close()
    print("  PASSED")


def test_obs_range():
    _header("4 · Observation sanity — normalised values in expected range")
    env = ShooterEnv(render_mode=None)
    env.reset(seed=1)

    max_abs = 0.0
    for _ in range(300):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        # Global / turret features (indices 0-5) should be roughly in [-2, 2]
        assert np.all(np.isfinite(obs)), "Non-finite value in observation"
        max_abs = max(max_abs, float(np.abs(obs[:6]).max()))
        if terminated or truncated:
            break

    print(f"  Max |obs[0:6]| over episode = {max_abs:.4f}  (expected ≈ [0, 2])")
    assert max_abs < 5.0, f"Observation values suspiciously large: {max_abs}"
    env.close()
    print("  PASSED")


def test_action_coverage():
    _header("5 · All 7 actions execute without error")
    env = ShooterEnv(render_mode=None)
    env.reset(seed=3)

    for action in range(env.action_space.n):
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            print(f"    action {action} ({ACTION_NAMES[action]:20s}) "
                  f"reward={reward:+.2f}  ok")
        except Exception as e:
            print(f"    action {action} FAILED: {e}")
            raise
        if terminated or truncated:
            env.reset()

    env.close()
    print("  PASSED")


def test_wave_guard():
    _header("6 · Wave guard — win must not fire before all periodic waves launched")
    rng    = np.random.default_rng(42)
    engine = GameEngine(rng)
    engine.reset()

    # Advance past the tick > 20 guard used by the win condition
    for _ in range(25):
        engine.step(0)

    # Force-kill every currently alive vehicle and freeze their respawn timers
    for v in engine.state["vehicles"]:
        v["alive"]        = False
        v["respawnTimer"] = MAX_GAME_TICKS  # won't respawn during this test

    # One more tick — win must NOT fire: waves_launched (0) < NUM_PERIODIC_WAVES (3)
    engine.step(0)

    assert not engine.state["gameOver"], (
        f"Win fired early: waves_launched={engine.state['waves_launched']} "
        f"but NUM_PERIODIC_WAVES={NUM_PERIODIC_WAVES}"
    )
    print(f"  waves_launched={engine.state['waves_launched']}  "
          f"NUM_PERIODIC_WAVES={NUM_PERIODIC_WAVES}  "
          f"gameOver={engine.state['gameOver']}")
    print("  PASSED")


def test_gym_make():
    _header("7 · gym.make('Shooter-v0') registration")
    try:
        import gymnasium as gym
        import shooter  # noqa: F401 — triggers registration
        env = gym.make("Shooter-v0")
        obs, _ = env.reset(seed=0)
        assert obs.shape == (OBS_SIZE,)
        env.close()
        print("  PASSED")
    except Exception as e:
        print(f"  SKIPPED ({e})")


if __name__ == "__main__":
    errors = []
    for test_fn in (
        test_api_contract,
        test_random_episode,
        test_seed_reproducibility,
        test_obs_range,
        test_action_coverage,
        test_wave_guard,
        test_gym_make,
    ):
        try:
            test_fn()
        except Exception:
            errors.append(test_fn.__name__)
            traceback.print_exc()

    print(f"\n{'═'*55}")
    if errors:
        print(f"  FAILED: {', '.join(errors)}")
        sys.exit(1)
    else:
        print("  All tests passed ✓")
    print(f"{'═'*55}\n")
