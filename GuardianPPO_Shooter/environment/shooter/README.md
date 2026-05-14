# ShooterEnv — Deep Reinforcement Learning Environment

A Hunter-only, single-agent 3D game environment implemented as a [Farama Gymnasium](https://gymnasium.farama.org/) environment. A stationary turret at the origin must shoot down AI drones flying at various altitudes across a 200 × 200 unit arena. The environment is rendered with PyOpenGL.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [Environment Overview](#3-environment-overview)
4. [Observation Space](#4-observation-space)
5. [Action Space](#5-action-space)
6. [Reward Function](#6-reward-function)
7. [Game Engine API](#7-game-engine-api)
8. [File Structure](#8-file-structure)

---

## 1. Installation

```bash
pip install gymnasium pygame PyOpenGL PyOpenGL_accelerate
```

PyOpenGL is required for rendering (`render_mode="human"` or `"rgb_array"`). Training with `render_mode=None` requires only `gymnasium` and `numpy`.

---

## 2. Quick Start

```python
import gymnasium as gym
import shooter                      # registers "Shooter-v0"

env = gym.make("Shooter-v0")
obs, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

Run the automated test suite:

```bash
python shooter/test_shooter.py
```

Run the interactive 3D keyboard demo:

```bash
python shooter/play_shooter.py
python shooter/play_shooter.py --seed 7
```

---

## 3. Environment Overview

### Scenario

| Element | Description |
|---|---|
| **Hunter** | Stationary turret fixed at the origin `(0, 0, 0)`. The agent controls yaw and pitch in 3D. |
| **AI Drones** | Spawn at the arena edge and fly toward the Hunter at a fixed random altitude between 4 and 10 units. Respawn 5 s after being killed. Episodes use a 3-wave escalation: 4 drones at start, then +2 at minute 1, +3 at minute 2, +4 at minute 3 — each wave is 50% larger than the previous (capacity: 4 → 6 → 9 → 13). |
| **Trees** | 36 static obstacles scattered across the arena. Vehicles steer around them in XZ; bullets are blocked by them as solid vertical pillars (XZ plane only — no height exemption). |
| **Bullets** | Fired by the Hunter along the 3D aim vector (yaw + pitch). Each bullet destroys the **first** drone whose 3D position is within `HIT_DIST` (3.5 units), then disappears — single target per shot. |

### Why Pitch Matters

Drones fly at varying altitudes (4 – 10 units). The turret barrel is at height 2.3 units. To hit a drone at distance `D` and altitude `Y`, the required pitch angle is:

```
pitch = atan2(Y − 2.3, D)
```

For example, a drone at Y = 8 and XZ distance = 30 requires `pitch ≈ 0.19 rad (≈ 11°)`. At close range (D = 10, Y = 10), `pitch ≈ 0.66 rad (≈ 38°)` is needed. `fire_at_nearest` (action 6, **human mode only**) computes this automatically — RL agents must learn to manage pitch themselves via actions 4 and 5.

### Episode Lifecycle

```
reset(seed)
    │
    ▼
┌──────────────────────────────────────────┐
│  step(action) × N                        │
│  ┌─────────────────────────────────────┐ │
│  │  apply action  (turret aim / fire)  │ │
│  │  advance physics (AI + bullets)     │ │
│  │  check win / lose conditions        │ │
│  │  return obs, reward, done, info     │ │
│  └─────────────────────────────────────┘ │
└──────────────────────────────────────────┘
    │
    ├─ terminated ──► a drone reached the Hunter, OR all waves cleared and field empty
    └─ truncated  ──► 4 800 ticks elapsed (4 minutes)
```

### Key Constants

| Constant | Value | Meaning |
|---|---|---|
| `TICK_RATE` | 20 | Ticks per second (50 ms / tick) |
| `ARENA` | 200 | Side length; XZ coordinates ∈ [−100, 100] |
| `VEHICLE_HEIGHT_MIN` | 4.0 | Minimum drone altitude (units) |
| `VEHICLE_HEIGHT_MAX` | 10.0 | Maximum drone altitude (units) |
| `BULLET_SPEED` | 2.5 | Units per tick |
| `AI_SPEED` | 0.35 | Normal AI approach speed (units / tick) |
| `HIT_DIST` | 3.5 | Bullet → drone 3D collision radius |
| `REACH_DIST` | 6.5 | Drone → Hunter XZ collision radius (game-over) |
| `MAX_GAME_TICKS` | 4 800 | 4-minute hard episode limit |
| `WAVE_INTERVAL` | 1 200 | New wave every 60 s (at minutes 1, 2, 3) |
| `WAVE_GROWTH` | 1.5 | Each wave is 50% larger than the previous |
| `NUM_PERIODIC_WAVES` | 3 | Total periodic waves per episode |
| `INITIAL_WAVE_SIZE` | 4 | Drones at episode start (wave 0) |
| `TREE_CLEAR_RADIUS` | 20 | No trees within this XZ radius of the turret |
| `RESPAWN_TICKS_AI` | 100 | Killed drone respawns after 5 s |

---

## 4. Observation Space

The observation is a **flat `float32` vector of length 169** returned by every `reset()` and `step()` call. All values are normalised to approximately `[−1, 1]`.

```python
observation_space = Box(low=-inf, high=inf, shape=(169,), dtype=float32)
```

### Layout

```
Index        Size   Block
──────────────────────────────────────────────────────────
[0   …   2]    3    Global state
[3   …   5]    3    Turret state
[6   … 125]  120    Vehicle features  (15 slots × 8 features)
[126 … 165]   40    Bullet features   (10 slots × 4 features)
[166 … 168]    3    Nearest-threat summary
──────────────────────────────────────────────────────────
Total          169
```

Use the `OBS` class for named index access:

```python
from shooter.shooter_env import OBS

obs[OBS.YAW]                            # hunter yaw (normalised)
obs[OBS.PITCH]                          # hunter pitch (normalised)
obs[OBS.VEH_START : OBS.BULLET_START]   # all 15 vehicle slots (120 values)
obs[OBS.THREAT_START :]                 # nearest-threat block (3 values)
```

### Block 1 — Global State `[0…2]`

| Index | Feature | Normalisation | Range |
|---|---|---|---|
| 0 | `tick` | `/ 4 800` | [0, 1] |
| 1 | `hunterScore` | `/ 500` | [−∞, ∞] |
| 2 | Alive drone count | `/ 15` (`MAX_VEH_SLOTS`; max alive ≈ 13 at wave 3) | [0, 0.87] |

### Block 2 — Turret State `[3…5]`

| Index | Feature | Normalisation | Range |
|---|---|---|---|
| 3 | `hunterYaw` | `/ π` | [−1, 1] |
| 4 | `hunterPitch` | `/ 0.9` | [−0.33, 1] |
| 5 | `hunterRoll` | `/ 0.25` | [−1, 1] |

### Block 3 — Vehicle Features `[6…125]`

**15 slots × 8 values** — sorted by ascending XZ distance from the Hunter (nearest first). Unused slots are zero-padded.

| Slot offset | Feature | Normalisation |
|---|---|---|
| +0 | Vehicle X position | `/ 100` |
| +1 | Vehicle Z position | `/ 100` |
| +2 | XZ distance from Hunter | `/ 141` (≈ half arena diagonal) |
| +3 | `sin(xz_angle)` toward vehicle | — |
| +4 | `cos(xz_angle)` toward vehicle | — |
| +5 | Vehicle altitude Y | `/ 20` |
| +6 | Alive flag | `1.0` / `0.0` |
| +7 | AI flag | always `1.0` in this environment |

The altitude feature (`+5`) tells the agent how much pitch is needed to hit each drone. Combined with XZ distance (`+2`), the required pitch is `atan2(y * 20 − 2.3, dist * 141)`.

### Block 4 — Bullet Features `[126…165]`

**10 slots × 4 values** — zero-padded.

| Slot offset | Feature | Normalisation |
|---|---|---|
| +0 | Bullet X position | `/ 100` |
| +1 | Bullet Z position | `/ 100` |
| +2 | X velocity component `dx` | unit vector component |
| +3 | Z velocity component `dz` | unit vector component |

### Block 5 — Nearest-Threat Summary `[166…168]`

Pre-computed features for the single closest alive drone (by XZ distance).

| Index | Feature | Normalisation |
|---|---|---|
| 166 | 3D distance to nearest drone | `/ 141` |
| 167 | XZ angle to nearest drone | `/ π` |
| 168 | Approach-speed proxy | dot product of position × heading |

If no drones are alive: `[1.0, 0.0, 0.0]`.

---

## 5. Action Space

The action space depends on `render_mode`:

| `render_mode` | Space | Description |
|---|---|---|
| `None` / `"rgb_array"` | `Discrete(6)` | RL training — actions 0 – 5 only |
| `"human"` | `Discrete(7)` | Interactive play — adds action 6 (auto-aim) |

Action 6 is intentionally excluded from RL training so the agent learns to aim and fire without a shortcut.

| Action ID | Name | Effect | Available |
|---|---|---|---|
| `0` | `do_nothing` | Hold current aim; no shot | always |
| `1` | `fire` | Shoot a bullet along the current 3D aim vector | always |
| `2` | `yaw_left` | Rotate turret left `+0.10 rad` (≈ +5.7°) | always |
| `3` | `yaw_right` | Rotate turret right `−0.10 rad` (≈ −5.7°) | always |
| `4` | `pitch_up` | Tilt barrel upward `+0.02 rad` | always |
| `5` | `pitch_down` | Tilt barrel downward `−0.02 rad` | always |
| `6` | `fire_at_nearest` | Snap yaw **and pitch** to the nearest alive drone, then fire | **human only** |

### Turret Limits

| Axis | Minimum | Maximum | Notes |
|---|---|---|---|
| Yaw | −π rad | +π rad | Wraps continuously |
| Pitch | −0.3 rad (≈ −17°) | +0.9 rad (≈ +52°) | Extended range required to reach drones at close range and max altitude |

### Bullet Behaviour

- Each shot spawns one bullet travelling at 2.5 units / tick in the 3D aim direction `(dx, dy, dz)`.
- A bullet is destroyed on the **first** drone it hits — **single target per shot** — using a full 3D distance check.
- A bullet is also destroyed when it hits a tree (XZ collision, no height exemption), exits the arena boundary, travels beyond 250 units, or its Y position exceeds `[−2, 80]`.
- Action `6` (`fire_at_nearest`, **human mode only**) snaps yaw to `atan2(vz, vx)` and computes the correct pitch via `atan2(vy − 2.3, xz_dist)` before firing. Carries the same `−2` shot cost as action `1`. Not available to RL agents (`render_mode=None` or `"rgb_array"`).

---

## 6. Reward Function

The reward is returned as a Python `float` on every `step()` call. There is no passive reward for standing still — the agent is only rewarded for kills and penalised for proximity danger and wasted shots.

### Per-Step Components

| Condition | Reward |
|---|---|
| AI drone killed this tick | `+20` per kill |
| Shot fired (action `1` or `6`) | `−2` |
| Nearest drone within 20 units (XZ) | `−(20 − dist) × 0.05` (up to `−1.0`) |

### Terminal Components

Added to the final step reward when an episode ends.

| Termination condition | Reward |
|---|---|
| A drone reached the Hunter (`terminated = True`) | `−100` |
| All periodic waves launched and field cleared (`terminated = True`) | `+200` |
| Time limit reached — 4 800 ticks / 4 minutes (`truncated = True`) | `0` (no bonus) |

---

## 7. Game Engine API

The implementation is split into two layers:

```
ShooterEnv (gym.Env)     ← Gymnasium API — used by DRL frameworks
    └── GameEngine        ← Pure Python game logic — no Gym dependency
```

### 7.1 `GameEngine`

`GameEngine` contains all physics and game logic and can be used independently of Gymnasium.

```python
from shooter.shooter_env import GameEngine
import numpy as np

rng    = np.random.default_rng(seed=42)
engine = GameEngine(rng)
engine.reset()

obs                                      = engine.get_obs()
obs, reward, terminated, truncated, info = engine.step(action)
```

#### Methods

| Method | Input | Output | Description |
|---|---|---|---|
| `reset()` | — | `None` | Initialise a fresh episode |
| `step(action)` | `int` | `(obs, reward, terminated, truncated, info)` | Advance one tick |
| `get_obs()` | — | `np.ndarray float32 (169,)` | Build the current observation vector |

#### `step()` — Input / Output Contract

```
INPUT
─────
action : int ∈ {0 … 5}  (RL) or {0 … 6}  (human mode)

OUTPUT
──────
obs        : np.ndarray float32 (169,)   normalised observation of the new state
reward     : float                       shaped reward for this transition
terminated : bool                        True on game-over event
truncated  : bool                        True when tick limit reached
info       : dict                        see Info Dict below
```

#### Info Dict

| Key | Type | Description |
|---|---|---|
| `tick` | `int` | Current simulation tick (0 – 4 800) |
| `hunterScore` | `int` | Cumulative score: `+20` per kill, `−2` per shot |
| `alive_ai` | `int` | Number of alive AI drones right now |
| `total_ai` | `int` | Total AI drone entries in the vehicles list |
| `total_spawned_ai` | `int` | Unique drones spawned so far this episode |
| `wave` | `int` | Periodic waves launched so far (0 – 3) |
| `wave_capacity` | `int` | Max simultaneous vehicles at current wave (4 → 6 → 9 → 13) |
| `bullets_in_flight` | `int` | Active bullet count |
| `gameOverReason` | `str` | `""` until terminal; then the reason string |

---

### 7.2 `ShooterEnv` (Gymnasium wrapper)

```python
from shooter.shooter_env import ShooterEnv

env = ShooterEnv(render_mode=None)   # "human" | "rgb_array" | None
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `render_mode` | `str \| None` | `None` | `"human"` opens a PyOpenGL window; `"rgb_array"` returns frames via `glReadPixels`; `None` disables rendering (recommended for training) |

#### `reset(seed, options)` → `(obs, info)`

| Parameter | Description |
|---|---|
| `seed` | Optional `int` — set for fully reproducible episodes |
| `options` | Not used; reserved for future use |

Calling `reset(seed=N)` twice with the same `N` produces identical initial states. Tree layout, vehicle spawn angles, and drone altitudes are all derived from the seed.

#### `step(action)` → `(obs, reward, terminated, truncated, info)`

| Field | Type | Description |
|---|---|---|
| `obs` | `float32 (169,)` | Observation of the next state |
| `reward` | `float` | Shaped reward for this transition |
| `terminated` | `bool` | `True` on game-over event |
| `truncated` | `bool` | `True` when `MAX_GAME_TICKS` reached |
| `info` | `dict` | Diagnostic values (see Info Dict above) |

#### Spaces

```python
env.observation_space  # Box(low=-inf, high=inf, shape=(169,), dtype=float32)
env.action_space       # Discrete(6) for render_mode=None/"rgb_array"; Discrete(7) for "human"
```

#### `render()` → `np.ndarray | None`

Returns a `(700, 700, 3) uint8` array when `render_mode="rgb_array"` (read via `glReadPixels`, flipped to top-down row order).

#### `close()`

Shuts down the pygame/OpenGL window and releases all resources.

#### `_pre_flip_hook`

Optional `callable()` (no arguments) called just before `pygame.display.flip()` each frame. Intended for compositing stat overlays without causing a double-flip flicker. Use the module-level OpenGL 2D helpers exported from `shooter_env`:

```python
from shooter.shooter_env import gl_begin_2d, gl_end_2d, gl_fill_rect, gl_blit

def my_overlay():
    W = env._WINDOW_SIZE
    gl_begin_2d(W, W)
    gl_fill_rect(10, W - 30, 200, 22, 0.0, 0.0, 0.0, 0.7)
    gl_blit("Custom overlay", 14, W - 28, font, (255, 255, 100))
    gl_end_2d()

env._pre_flip_hook = my_overlay
```

---

### 7.3 Internal Tick Sequence

Each `step(action)` executes in this exact order:

```
step(action)
 ├─ _apply_action(action)
 │    ├─ update hunterYaw (±0.10 rad) / hunterPitch (±0.02 rad)
 │    ├─ clamp pitch to [−0.3, +0.9] rad; wrap yaw to [−π, π]
 │    ├─ if action == 6 (human mode only): snap yaw + pitch to nearest alive drone
 │    │    yaw   = atan2(vz, vx)
 │    │    pitch = atan2(vy − 2.3, xz_dist)
 │    └─ if action ∈ {1, 6}: spawn bullet at barrel tip; hunterScore −= 2
 │
 ├─ _tick()
 │    ├─ increment tick; check MAX_GAME_TICKS (4 800) → truncated
 │    ├─ if tick % WAVE_INTERVAL == 0 and waves_launched < NUM_PERIODIC_WAVES (3):
 │    │    spawn int(wave_capacity × 1.5) − wave_capacity new vehicles; update capacity
 │    ├─ for each drone:
 │    │    ├─ alive  → steer toward origin + obstacle avoidance (XZ only)
 │    │    │           reach check → gameOver if XZ dist < REACH_DIST (6.5)
 │    │    └─ dead   → decrement respawnTimer; respawn at arena edge when 0
 │    └─ for each bullet:
 │         ├─ advance position by (dx, dy, dz) × BULLET_SPEED (2.5)
 │         ├─ cull: dist > 250 / outside arena / Y outside [−2, 80]
 │         ├─ cull: tree collision in XZ plane (solid pillars, no Y check)
 │         └─ 3D hit: first drone within HIT_DIST (3.5) → kill + remove bullet
 │
 ├─ get_obs()          → obs float32 [169]
 ├─ _compute_reward()  → reward float
 └─ _render_frame()    → (only when render_mode is set)
```

---

### 7.4 Rendering

The PyOpenGL renderer (`700 × 700` px) shows the arena from a fixed angled overhead camera at position `(0, 130, 85)` looking toward `(0, 5, 0)`. Height differences between drones are clearly visible.

| Element | Visual |
|---|---|
| Arena floor | Dark green quad with grid at 10-unit intervals |
| Trees | Dark green tall boxes (15 units high — solid visual pillars) |
| AI drones | Coloured flat boxes at their actual altitude; altitude wire + ground shadow; ghost when dead |
| Bullets | Bright yellow tracer line fading to orange at the tail; point at head |
| Hunter (origin) | Green base box + thick barrel line + faint aim ray extending into distance |
| HUD (top-left) | Tick, elapsed time, score, AI alive + wave progress (`wave N/3  cap K`), bullets in flight, yaw / pitch |

```python
# Interactive 3D window
env = ShooterEnv(render_mode="human")

# Capture frames as numpy arrays (via glReadPixels)
env   = ShooterEnv(render_mode="rgb_array")
frame = env.render()   # shape (700, 700, 3) uint8
```

### 7.5 OpenGL 2D Utility Functions

These module-level functions are exported for use in `_pre_flip_hook` callbacks. All require an active OpenGL context (i.e., must be called from within the hook).

| Function | Description |
|---|---|
| `gl_begin_2d(W, H)` | Enter 2D orthographic mode. Y=0 at bottom. |
| `gl_end_2d()` | Exit 2D mode and restore 3D matrices. |
| `gl_fill_rect(x, y, w, h, r, g, b, a)` | Draw a filled rectangle (RGBA 0–1 floats). |
| `gl_blit(text, x, y, font, color)` | Render text at screen position `(x, y)` (bottom of glyph). Returns rendered height in pixels. |

### 7.6 Keyboard Play

```bash
python shooter/play_shooter.py           # default seed 42
python shooter/play_shooter.py --seed 7
```

| Key | Action |
|---|---|
| `←` / `A` | Yaw left |
| `→` / `D` | Yaw right |
| `↑` / `W` | Pitch up |
| `↓` / `S` | Pitch down |
| `Space` | Fire |
| `F` | Auto-aim (yaw + pitch) + fire |
| `R` | Reset episode |
| `Esc` / `Q` | Quit |

---

## 8. File Structure

```
shooter/
├── __init__.py        registers "Shooter-v0" with Gymnasium
├── shooter_env.py     GameEngine + ShooterEnv + OpenGL drawing utilities
├── play_shooter.py    interactive 3D keyboard play with stats overlay
├── test_shooter.py    automated test suite (7 tests)
└── README.md          this file
```

### Test Suite

```bash
python shooter/test_shooter.py
```

| Test | What it checks |
|---|---|
| `test_api_contract` | `reset()` / `step()` shapes `(169,)`, dtypes, `observation_space.contains()`, all info keys, `action_space.n == 6` (RL mode) |
| `test_random_episode` | Full random episode runs to termination without error |
| `test_seed_reproducibility` | Same seed → identical obs; different seeds → different obs |
| `test_obs_range` | All observation values are finite and within expected magnitude |
| `test_action_coverage` | All 6 RL actions execute without error |
| `test_wave_guard` | Win condition does not fire when only the 4 initial drones are dead — all 3 periodic waves not yet launched |
| `test_gym_make` | `gym.make("Shooter-v0")` registration works end-to-end |
