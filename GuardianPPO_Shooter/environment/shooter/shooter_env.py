"""
shooter_env.py
==============
Hunter-only 3D game engine wrapped as a Farama Gymnasium environment.
Rendered with PyOpenGL — a stationary turret at the origin shoots down
AI drones flying at various altitudes across a 200 × 200 arena.

┌────────────────────────────────────────────────────────────────┐
│  INPUTS  (what the DRL agent sends to the environment)        │
│                                                                │
│  action : int ∈ {0 … 5}   →   Discrete(6) action space       │
│                            (render_mode="human" adds action 6)│
│                                                                │
│   0  do_nothing          hold current aim                     │
│   1  fire                shoot along current aim vector       │
│   2  yaw_left            rotate turret left   (+0.10 rad)     │
│   3  yaw_right           rotate turret right  (−0.10 rad)     │
│   4  pitch_up            tilt barrel up       (+0.02 rad)     │
│   5  pitch_down          tilt barrel down     (−0.02 rad)     │
│   6  fire_at_nearest     auto-aim + fire  [human mode only]   │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  OUTPUTS  (what the environment returns to the agent)         │
│                                                                │
│  obs        float32[169]  normalised observation vector       │
│  reward     float         shaped scalar reward signal         │
│  terminated bool          game-over event occurred            │
│  truncated  bool          MAX_GAME_TICKS reached              │
│  info       dict          raw diagnostics for logging         │
└────────────────────────────────────────────────────────────────┘

Observation layout  (169 features, all float32)
───────────────────────────────────────────────
  [0]        tick / 4 800
  [1]        hunterScore / 500
  [2]        alive vehicle count / 15  (MAX_VEH_SLOTS; max alive ≈ 13 at wave 3)
  [3]        hunterYaw   / π
  [4]        hunterPitch / 0.9
  [5]        hunterRoll  / 0.25
  [6 …125]   15 vehicle slots × 8 features (sorted nearest-first, zero-padded)
               x/100, z/100, dist/141, sin(angle_xz), cos(angle_xz), y/20, alive, isAI
  [126…165]  10 bullet  slots × 4 features (zero-padded)
               x/100, z/100, dx, dz
  [166…168]  nearest-threat summary
               3d_dist/141, xz_angle/π, approach-speed proxy

Reward components
─────────────────
  +20   × kills this tick
  −2    per shot (actions 1 and 6)
  −(20 − dist) × 0.05   when nearest vehicle is within 20 units (XZ)
  −100  game-over: a vehicle reached the Hunter
  +200  game-over: all waves cleared AND all AI eliminated before time limit
        (no bonus for simply surviving to the tick limit)

Quick start
───────────
    import gymnasium as gym
    import shooter                    # registers Shooter-v0
    env = gym.make("Shooter-v0")
    obs, info = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
"""

import math
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    _OPENGL_OK = True
except ImportError:
    _OPENGL_OK = False


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — GAME CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

TICK_RATE          = 20
ARENA              = 200
HALF               = ARENA // 2
TREE_R             = 2.5
VEH_R              = 2.0
BULLET_SPEED       = 2.5
AI_SPEED           = 0.35
RESPAWN_TICKS_AI   = 5 * TICK_RATE
HIT_DIST           = 3.5
REACH_DIST         = 6.5
BULLET_MAX_DIST    = 250
MAX_GAME_TICKS     = 4 * 60 * TICK_RATE    # 4800 ticks = 4 minutes
WAVE_INTERVAL      = 60 * TICK_RATE        # new wave every 60 s  (ticks: 1200, 2400, 3600)
WAVE_GROWTH        = 1.5                   # each wave is 50% larger than the previous
NUM_PERIODIC_WAVES = (MAX_GAME_TICKS // WAVE_INTERVAL) - 1   # = 3  (minutes 1, 2, 3)
INITIAL_WAVE_SIZE  = 4                     # vehicles at episode start  (= len(_AI_COLORS))
TREE_CLEAR_RADIUS  = 20                    # no trees within this XZ radius of the turret
VEHICLE_HEIGHT_MIN = 4.0                   # drone altitude range (units)
VEHICLE_HEIGHT_MAX = 10.0

_PITCH_MIN, _PITCH_MAX = -0.3, 0.9        # extended range for 3D aiming


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — ACTION SPACE
# ═══════════════════════════════════════════════════════════════════════════

NUM_ACTIONS  = 7
ACTION_NAMES = {
    0: "do_nothing",
    1: "fire",
    2: "yaw_left",          # Δyaw   = +0.10 rad
    3: "yaw_right",         # Δyaw   = −0.10 rad
    4: "pitch_up",          # Δpitch = +0.02 rad
    5: "pitch_down",        # Δpitch = −0.02 rad
    6: "fire_at_nearest",   # auto-aim (yaw + pitch) + fire  [human mode only]
}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — OBSERVATION SPACE  (169 float32 values)
# ═══════════════════════════════════════════════════════════════════════════

MAX_VEH_SLOTS    = 15
MAX_BULLET_SLOTS = 10
OBS_SIZE = (
    3                        # global state
    + 3                      # turret state
    + MAX_VEH_SLOTS    * 8   # 15 × 8 = 120  (added y feature)
    + MAX_BULLET_SLOTS * 4   # 10 × 4 =  40
    + 3                      # nearest-threat summary
)  # = 169


class OBS:
    """Named start indices for each block in the observation vector."""
    TICK         = 0
    SCORE        = 1
    ALIVE_COUNT  = 2
    YAW          = 3
    PITCH        = 4
    ROLL         = 5
    VEH_START    = 6
    BULLET_START = 6 + MAX_VEH_SLOTS * 8                          # 126
    THREAT_START = 6 + MAX_VEH_SLOTS * 8 + MAX_BULLET_SLOTS * 4   # 166


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — PURE GAME-ENGINE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

_AI_COLORS    = ["red", "lime", "cyan", "orange"]
_EXTRA_COLORS = ["magenta", "yellow", "deepskyblue", "tomato", "chartreuse",
                 "gold", "hotpink", "turquoise", "coral", "violet"]


def _dist(ax: float, az: float, bx: float, bz: float) -> float:
    return math.sqrt((ax - bx) ** 2 + (az - bz) ** 2)


def _dist3(ax: float, ay: float, az: float,
           bx: float, by: float, bz: float) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def _wrap_yaw(y: float) -> float:
    return y - 2 * math.pi * math.floor((y + math.pi) / (2 * math.pi))


def _gen_trees(rng: np.random.Generator) -> list:
    """genTrees() — guarantees exactly 36 trees placed via seeded RNG."""
    t = []

    def _rp():
        return float(rng.uniform(-(HALF - 15), HALF - 15))

    for _ in range(6):
        cx, cz = _rp(), _rp()
        for _ in range(4):
            x = cx + float(rng.uniform(-6, 6))
            z = cz + float(rng.uniform(-6, 6))
            if math.sqrt(x * x + z * z) > TREE_CLEAR_RADIUS:
                t.append({"x": x, "z": z, "r": TREE_R})

    for _ in range(12):
        x, z = _rp(), _rp()
        if math.sqrt(x * x + z * z) > TREE_CLEAR_RADIUS:
            t.append({"x": x, "z": z, "r": TREE_R})

    while len(t) < 36:
        x, z = _rp(), _rp()
        if math.sqrt(x * x + z * z) > TREE_CLEAR_RADIUS:
            t.append({"x": x, "z": z, "r": TREE_R})

    return t[:36]


def _tree_blocked_fast(tree_xs: np.ndarray, tree_zs: np.ndarray,
                       x: float, z: float, r: float) -> bool:
    """Vectorised tree collision check (squared-distance, no sqrt)."""
    threshold_sq = (r + TREE_R) ** 2
    dx = tree_xs - x
    dz = tree_zs - z
    return bool(np.any(dx * dx + dz * dz < threshold_sq))


def _vehicle_blocked(vehicles: list, x: float, z: float,
                     r: float, self_id: str) -> bool:
    for v in vehicles:
        if not v["alive"] or v["id"] == self_id:
            continue
        if _dist(x, z, v["x"], v["z"]) < r + VEH_R:
            return True
    return False


def _clamp_arena(x: float, z: float, r: float):
    lim = HALF - r
    return max(-lim, min(lim, x)), max(-lim, min(lim, z))


def _check_stuck(v: dict, tree_xs: np.ndarray, tree_zs: np.ndarray,
                 rng: np.random.Generator) -> None:
    if v["stuckCounter"] % 40 == 0 and v["stuckCounter"] > 0:
        if _dist(v["x"], v["z"], v["lastX"], v["lastZ"]) < 1.0:
            att = 0
            while att < 20:
                a  = float(rng.uniform(0, 2 * math.pi))
                tx = math.cos(a) * (HALF - 15)
                tz = math.sin(a) * (HALF - 15)
                if not _tree_blocked_fast(tree_xs, tree_zs, tx, tz, VEH_R):
                    break
                att += 1
            v["x"], v["z"] = tx, tz
            v["angle"]      = math.atan2(-v["z"], -v["x"])
            v["stuckCounter"] = 0
            return
    if v["stuckCounter"] % 40 == 0:
        v["lastX"] = v["x"]
        v["lastZ"] = v["z"]
    v["stuckCounter"] += 1


def _try_move(v: dict, nx: float, nz: float, ignore_trees: bool,
              tree_xs: np.ndarray, tree_zs: np.ndarray, vehicles: list) -> bool:
    blocked = (
        (not ignore_trees and _tree_blocked_fast(tree_xs, tree_zs, nx, nz, VEH_R))
        or _vehicle_blocked(vehicles, nx, nz, VEH_R, v["id"])
    )
    if not blocked:
        v["x"], v["z"] = _clamp_arena(nx, nz, VEH_R)
        return True
    return False


def _steer_ai(v: dict, tree_xs: np.ndarray, tree_zs: np.ndarray,
              vehicles: list, rng: np.random.Generator) -> None:
    if not v["alive"]:
        return
    d = _dist(v["x"], v["z"], 0.0, 0.0)
    _check_stuck(v, tree_xs, tree_zs, rng)

    desired    = math.atan2(-v["z"], -v["x"])
    random_amt = 0.05 if d < 20 else 0.25
    desired   += float(rng.uniform(-random_amt / 2, random_amt / 2))

    if d > 15:
        look_dist = 8
        ax = v["x"] + math.cos(desired) * look_dist
        az = v["z"] + math.sin(desired) * look_dist
        if _tree_blocked_fast(tree_xs, tree_zs, ax, az, VEH_R):
            found, offset = False, 0.4
            while offset < math.pi:
                for sign in (1, -1):
                    tx = v["x"] + math.cos(desired + offset * sign) * look_dist
                    tz = v["z"] + math.sin(desired + offset * sign) * look_dist
                    if not _tree_blocked_fast(tree_xs, tree_zs, tx, tz, VEH_R):
                        desired += offset * sign
                        found = True
                        break
                if found:
                    break
                offset += 0.25

    diff = desired - v["angle"]
    while diff >  math.pi: diff -= 2 * math.pi
    while diff < -math.pi: diff += 2 * math.pi
    turn_rate   = 0.18 if d < 15 else 0.08
    v["angle"] += math.copysign(min(abs(diff), turn_rate), diff)

    speed = AI_SPEED * 1.4 if d < 20 else AI_SPEED
    nx    = v["x"] + math.cos(v["angle"]) * speed
    nz    = v["z"] + math.sin(v["angle"]) * speed

    ignore = d < 15
    if not _try_move(v, nx, nz, ignore, tree_xs, tree_zs, vehicles):
        for off in (0.3, -0.3, 0.6, -0.6, 1.0, -1.0):
            if _try_move(v,
                         v["x"] + math.cos(v["angle"] + off) * speed,
                         v["z"] + math.sin(v["angle"] + off) * speed,
                         ignore, tree_xs, tree_zs, vehicles):
                break


# ═══════════════════════════════════════════════════════════════════════════
# SECTION GL — OPENGL DRAWING UTILITIES
# Module-level; importable by play_shooter.py for overlay rendering.
# All functions must be called with a valid OpenGL context active.
# ═══════════════════════════════════════════════════════════════════════════

def gl_begin_2d(W: int, H: int) -> None:
    """Enter 2D screen-space rendering.  Y=0 at bottom, Y=H at top."""
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, W, 0, H, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)


def gl_end_2d() -> None:
    """Exit 2D screen-space rendering and restore 3D matrices."""
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()


def gl_fill_rect(x: int, y: int, w: int, h: int,
                 r: float, g: float, b: float, a: float = 0.75) -> None:
    """Draw a filled axis-aligned rectangle in the current 2D mode."""
    glColor4f(r, g, b, a)
    glBegin(GL_QUADS)
    glVertex2f(x,     y    )
    glVertex2f(x + w, y    )
    glVertex2f(x + w, y + h)
    glVertex2f(x,     y + h)
    glEnd()


def gl_blit(text: str, x: int, y: int, font, color=(210, 210, 210)) -> int:
    """
    Render a text string at screen position (x, y) using a pygame font.
    y is the BOTTOM edge of the text (Y=0 at screen bottom).
    Returns the rendered text height in pixels.
    Requires gl_begin_2d() to be active.
    """
    import pygame
    surf = font.render(text, True, color)
    tw, th = surf.get_size()
    # True = flip vertically so row-0 is at the bottom (OpenGL convention)
    raw = pygame.image.tostring(surf, "RGBA", True)

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tw, th, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, raw)

    glEnable(GL_TEXTURE_2D)
    glColor4f(1.0, 1.0, 1.0, 1.0)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(x,      y     )
    glTexCoord2f(1, 0); glVertex2f(x + tw, y     )
    glTexCoord2f(1, 1); glVertex2f(x + tw, y + th)
    glTexCoord2f(0, 1); glVertex2f(x,      y + th)
    glEnd()
    glDisable(GL_TEXTURE_2D)
    glDeleteTextures([tex])
    return th


# ── Private 3D scene drawing helpers ──────────────────────────────────────

def _gl_box(w: float, h: float, d: float,
            r: float, g: float, b: float, a: float = 1.0) -> None:
    """
    Draw a face-shaded solid box centred at the origin.
    w/h/d = full dimensions along X/Y/Z.
    r,g,b,a in [0,1].  Top face is brightest; bottom is darkest.
    Pass a < 1.0 for transparent boxes (blending must be enabled).
    """
    hx, hy, hz = w * 0.5, h * 0.5, d * 0.5

    glBegin(GL_QUADS)

    # Top (+Y) — brightest
    glColor4f(min(r * 1.25, 1.0), min(g * 1.25, 1.0), min(b * 1.25, 1.0), a)
    glVertex3f(-hx, hy, -hz); glVertex3f(hx, hy, -hz)
    glVertex3f(hx, hy,  hz); glVertex3f(-hx, hy,  hz)

    # Front (−Z)
    glColor4f(r, g, b, a)
    glVertex3f(-hx, -hy, -hz); glVertex3f(hx, -hy, -hz)
    glVertex3f(hx,  hy, -hz); glVertex3f(-hx,  hy, -hz)

    # Back (+Z)
    glColor4f(r * 0.80, g * 0.80, b * 0.80, a)
    glVertex3f( hx, -hy, hz); glVertex3f(-hx, -hy, hz)
    glVertex3f(-hx,  hy, hz); glVertex3f( hx,  hy, hz)

    # Right (+X)
    glColor4f(r * 0.90, g * 0.90, b * 0.90, a)
    glVertex3f(hx, -hy, -hz); glVertex3f(hx, -hy, hz)
    glVertex3f(hx,  hy,  hz); glVertex3f(hx,  hy, -hz)

    # Left (−X)
    glColor4f(r * 0.70, g * 0.70, b * 0.70, a)
    glVertex3f(-hx, -hy,  hz); glVertex3f(-hx, -hy, -hz)
    glVertex3f(-hx,  hy, -hz); glVertex3f(-hx,  hy,  hz)

    # Bottom (−Y) — darkest
    glColor4f(r * 0.45, g * 0.45, b * 0.45, a)
    glVertex3f(-hx, -hy,  hz); glVertex3f( hx, -hy,  hz)
    glVertex3f( hx, -hy, -hz); glVertex3f(-hx, -hy, -hz)

    glEnd()


def _gl_draw_floor() -> None:
    # Ground quad
    glColor3f(0.05, 0.12, 0.05)
    glBegin(GL_QUADS)
    glVertex3f(-HALF, 0.0,  -HALF); glVertex3f(HALF, 0.0, -HALF)
    glVertex3f( HALF, 0.0,   HALF); glVertex3f(-HALF, 0.0,  HALF)
    glEnd()

    # Grid lines
    glColor3f(0.10, 0.20, 0.10)
    glBegin(GL_LINES)
    for i in range(-10, 11):
        v = float(i * 10)
        glVertex3f(v,    0.05, -HALF); glVertex3f(v,    0.05, HALF)
        glVertex3f(-HALF, 0.05, v   ); glVertex3f(HALF, 0.05, v   )
    glEnd()

    # Arena border
    glLineWidth(2.0)
    glColor3f(0.25, 0.50, 0.25)
    glBegin(GL_LINE_LOOP)
    glVertex3f(-HALF, 0.1, -HALF); glVertex3f(HALF, 0.1, -HALF)
    glVertex3f( HALF, 0.1,  HALF); glVertex3f(-HALF, 0.1,  HALF)
    glEnd()
    glLineWidth(1.0)


def _gl_draw_trees(trees: list) -> None:
    for t in trees:
        glPushMatrix()
        glTranslatef(t["x"], 7.5, t["z"])   # tree height 15 → centre at 7.5
        _gl_box(TREE_R * 2, 15.0, TREE_R * 2, 0.08, 0.32, 0.08)
        glPopMatrix()


_VEH_COL_GL = {
    "red":         (0.86, 0.20, 0.20),  "lime":        (0.20, 0.86, 0.20),
    "cyan":        (0.20, 0.78, 0.86),  "orange":      (0.86, 0.55, 0.20),
    "magenta":     (0.86, 0.20, 0.86),  "yellow":      (0.86, 0.86, 0.20),
    "deepskyblue": (0.00, 0.71, 1.00),  "tomato":      (1.00, 0.39, 0.28),
    "chartreuse":  (0.50, 1.00, 0.00),  "gold":        (1.00, 0.84, 0.00),
    "hotpink":     (1.00, 0.41, 0.71),  "turquoise":   (0.25, 0.88, 0.82),
    "coral":       (1.00, 0.50, 0.31),  "violet":      (0.93, 0.51, 0.93),
}
_DEFAULT_COL_GL = (0.60, 0.60, 0.60)


def _gl_draw_vehicles(vehicles: list) -> None:
    for v in vehicles:
        r, g, b = _VEH_COL_GL.get(v["color"], _DEFAULT_COL_GL)
        if v["alive"]:
            # Altitude shadow on ground
            glPushMatrix()
            glTranslatef(v["x"], 0.08, v["z"])
            glColor4f(0.0, 0.0, 0.0, 0.35)
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, 0, 0)
            for k in range(13):
                a = 2 * math.pi * k / 12
                glVertex3f(math.cos(a) * 2.0, 0, math.sin(a) * 2.0)
            glEnd()
            glPopMatrix()

            # Altitude wire (thin vertical line from ground to vehicle)
            glLineWidth(1.0)
            glColor4f(r, g, b, 0.35)
            glBegin(GL_LINES)
            glVertex3f(v["x"], 0.1,    v["z"])
            glVertex3f(v["x"], v["y"], v["z"])
            glEnd()

            # Vehicle body — drone-like flat box oriented to heading
            glPushMatrix()
            glTranslatef(v["x"], v["y"], v["z"])
            glRotatef(math.degrees(v["angle"]) + 90.0, 0, 1, 0)
            _gl_box(4.0, 1.2, 3.0, r, g, b)
            # Front marker
            glColor3f(min(r * 1.6, 1.0), min(g * 1.6, 1.0), min(b * 1.6, 1.0))
            glBegin(GL_QUADS)
            glVertex3f(-0.4, 0.62, -1.7); glVertex3f(0.4, 0.62, -1.7)
            glVertex3f( 0.4, 0.62, -1.2); glVertex3f(-0.4, 0.62, -1.2)
            glEnd()
            glPopMatrix()

        else:
            # Ghost of dead/respawning vehicle — semi-transparent dark box
            glPushMatrix()
            glTranslatef(v["x"], v["y"], v["z"])
            _gl_box(4.0, 1.2, 3.0, r * 0.30, g * 0.30, b * 0.30, a=0.30)
            glPopMatrix()


def _gl_draw_bullets(bullets: list) -> None:
    # Tracer line (fades from bright to transparent)
    glLineWidth(2.0)
    glBegin(GL_LINES)
    for b in bullets:
        glColor4f(1.0, 0.85, 0.15, 1.0)
        glVertex3f(b["x"], b["y"], b["z"])
        glColor4f(1.0, 0.40, 0.05, 0.0)
        trail = 5.0
        glVertex3f(b["x"] - b["dx"] * trail,
                   b["y"] - b["dy"] * trail,
                   b["z"] - b["dz"] * trail)
    glEnd()
    glLineWidth(1.0)

    # Bright point at bullet head
    glPointSize(5.0)
    glBegin(GL_POINTS)
    for b in bullets:
        glColor3f(1.0, 1.0, 0.4)
        glVertex3f(b["x"], b["y"], b["z"])
    glEnd()
    glPointSize(1.0)


def _gl_draw_hunter(yaw: float, pitch: float) -> None:
    # Turret base (sits on floor, centre at y=1.0 → occupies 0..2)
    glPushMatrix()
    glTranslatef(0.0, 1.0, 0.0)
    _gl_box(5.0, 2.0, 5.0, 0.10, 0.70, 0.20)
    glPopMatrix()

    # Barrel direction vector
    cp = math.cos(pitch)
    dx = math.cos(yaw) * cp
    dy = math.sin(pitch)
    dz = math.sin(yaw) * cp
    blen = 18.0

    # Barrel line
    glLineWidth(4.0)
    glColor3f(0.20, 1.00, 0.40)
    glBegin(GL_LINES)
    glVertex3f(0.0, 2.3, 0.0)
    glVertex3f(dx * blen, 2.3 + dy * blen, dz * blen)
    glEnd()

    # Faint aim ray extending into the distance (disable depth test so it
    # is never occluded by terrain or vehicles — it should always be visible)
    glDisable(GL_DEPTH_TEST)
    glLineWidth(1.0)
    glColor4f(0.20, 1.00, 0.40, 0.12)
    glBegin(GL_LINES)
    glVertex3f(0.0, 2.3, 0.0)
    glVertex3f(dx * 120.0, 2.3 + dy * 120.0, dz * 120.0)
    glEnd()
    glEnable(GL_DEPTH_TEST)


def _gl_draw_hud(s: dict, hud_font, W: int) -> None:
    """Built-in stats HUD drawn as a 2D overlay in the top-left corner."""
    yaw      = s["hunterYaw"]
    pitch    = s["hunterPitch"]
    alive_ai = sum(1 for v in s["vehicles"] if v["isAI"] and v["alive"])

    lines = [
        (f"Tick  : {s['tick']:5d} / {MAX_GAME_TICKS}", (160, 190, 160)),
        (f"Time  : {s['tick'] / TICK_RATE:.1f} s",     (160, 190, 160)),
        (f"Score : {s['hunterScore']}",                  (110, 255, 110)),
        (f"AI    : {alive_ai} alive  "
         f"wave {s['waves_launched']}/{NUM_PERIODIC_WAVES}  "
         f"cap {s['wave_capacity']}",                    (160, 190, 160)),
        (f"Shots : {len(s['bullets'])} in flight",       (160, 190, 160)),
        (f"Yaw {math.degrees(yaw):+.1f}°  Pitch {math.degrees(pitch):+.1f}°",
         (200, 200, 140)),
    ]
    if s["gameOver"]:
        lines.insert(0, (f">>> {s['gameOverReason']}", (255, 80, 80)))

    LINE_H  = 18
    PAD     = 6
    panel_w = 290
    panel_h = len(lines) * LINE_H + PAD

    gl_begin_2d(W, W)
    gl_fill_rect(0, W - panel_h, panel_w, panel_h, 0.02, 0.02, 0.02, 0.65)
    for i, (text, color) in enumerate(lines):
        y = W - PAD - (i + 1) * LINE_H
        gl_blit(text, PAD, y, hud_font, color)
    gl_end_2d()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — GAME ENGINE  (pure Python — no Gymnasium dependency)
#
#  INPUT  → action : int  (one of NUM_ACTIONS discrete choices)
#  OUTPUT → (obs, reward, terminated, truncated, info)
# ═══════════════════════════════════════════════════════════════════════════

class GameEngine:
    """
    Self-contained 3D game engine.  Gymnasium-agnostic.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded random-number generator (pass env.np_random from ShooterEnv).

    Typical usage
    -------------
        engine = GameEngine(rng)
        engine.reset()
        obs = engine.get_obs()                    # float32[169]
        obs, reward, terminated, truncated, info = engine.step(action)
    """

    def __init__(self, rng: np.random.Generator):
        self._rng  = rng
        self.state: dict = {}

    # ── Public interface ───────────────────────────────────────────────────

    def reset(self) -> None:
        self.state    = self._build_initial_state()
        trees         = self.state["trees"]
        self._tree_xs = np.array([t["x"] for t in trees], dtype=np.float32)
        self._tree_zs = np.array([t["z"] for t in trees], dtype=np.float32)

    def step(self, action: int):
        """
        Advance the simulation by one tick.

        INPUT
        -----
        action : int ∈ {0 … NUM_ACTIONS−1}

        OUTPUT
        ------
        obs        : np.ndarray float32 shape (OBS_SIZE,)
        reward     : float
        terminated : bool
        truncated  : bool
        info       : dict  — tick, hunterScore, alive_ai, total_ai,
                             bullets_in_flight, gameOverReason
        """
        prev_alive = sum(1 for v in self.state["vehicles"] if v["alive"])

        self._apply_action(action)
        self._tick()

        curr_alive = sum(1 for v in self.state["vehicles"] if v["alive"])
        obs        = self.get_obs()
        reward     = self._compute_reward(action, prev_alive, curr_alive)
        terminated = bool(self.state["gameOver"])
        truncated  = self.state["tick"] >= MAX_GAME_TICKS and not terminated
        info       = self._build_info()

        return obs, reward, terminated, truncated, info

    def get_obs(self) -> np.ndarray:
        """
        Build the 169-element normalised observation vector.

        Feature blocks
        ──────────────
        [0  … 2 ]  global   : tick/4800, score/500, alive_count/15 (MAX_VEH_SLOTS)
        [3  … 5 ]  turret   : yaw/π, pitch/0.9, roll/0.25
        [6  …125]  vehicles : 15 slots × 8 values (nearest-xz first, zero-padded)
                              x/100, z/100, dist_xz/141, sinθ, cosθ, y/20, alive, isAI
        [126…165]  bullets  : 10 slots × 4 values (zero-padded)
                              x/100, z/100, dx, dz
        [166…168]  threat   : 3d_dist/141, xz_angle/π, approach-speed proxy
        """
        s   = self.state
        obs = np.zeros(OBS_SIZE, dtype=np.float32)

        # — Global state (indices 0–2) ────────────────────────────────────
        obs[0] = s["tick"]        / MAX_GAME_TICKS
        obs[1] = s["hunterScore"] / 500.0
        obs[2] = sum(1 for v in s["vehicles"] if v["alive"]) / MAX_VEH_SLOTS

        # — Turret state (indices 3–5) ────────────────────────────────────
        obs[3] = s["hunterYaw"]   / math.pi
        obs[4] = s["hunterPitch"] / 0.9
        obs[5] = s["hunterRoll"]  / 0.25

        # — Vehicle features (indices 6–125) — sort by squared XZ distance ─
        vehs = sorted(s["vehicles"], key=lambda v: v["x"] ** 2 + v["z"] ** 2)
        for i in range(min(len(vehs), MAX_VEH_SLOTS)):
            v     = vehs[i]
            d_xz  = math.sqrt(v["x"] ** 2 + v["z"] ** 2)
            angle = math.atan2(v["z"], v["x"])
            j = OBS.VEH_START + i * 8
            obs[j]     = v["x"] / 100.0
            obs[j + 1] = v["z"] / 100.0
            obs[j + 2] = d_xz   / 141.0
            obs[j + 3] = math.sin(angle)
            obs[j + 4] = math.cos(angle)
            obs[j + 5] = v["y"] / 20.0
            obs[j + 6] = 1.0 if v["alive"] else 0.0
            obs[j + 7] = 1.0 if v["isAI"]  else 0.0

        # — Bullet features (indices 126–165) ─────────────────────────────
        for i in range(min(len(s["bullets"]), MAX_BULLET_SLOTS)):
            b = s["bullets"][i]
            j = OBS.BULLET_START + i * 4
            obs[j]     = b["x"]  / 100.0
            obs[j + 1] = b["z"]  / 100.0
            obs[j + 2] = b["dx"]
            obs[j + 3] = b["dz"]

        # — Nearest-threat summary (indices 166–168) ──────────────────────
        alive = [v for v in s["vehicles"] if v["alive"]]
        base  = OBS.THREAT_START
        if alive:
            n        = min(alive, key=lambda v: v["x"] ** 2 + v["z"] ** 2)
            nd3      = _dist3(n["x"], n["y"], n["z"], 0.0, 2.3, 0.0)
            approach = n["x"] * math.cos(n["angle"]) + n["z"] * math.sin(n["angle"])
            obs[base]     = nd3 / 141.0
            obs[base + 1] = math.atan2(n["z"], n["x"]) / math.pi
            obs[base + 2] = approach
        else:
            obs[base] = 1.0

        return obs

    # ── State initialisation ───────────────────────────────────────────────

    def _build_initial_state(self) -> dict:
        trees        = _gen_trees(self._rng)
        vehicles     = []
        angle_offset = float(self._rng.uniform(0, 2 * math.pi))

        for i, color in enumerate(_AI_COLORS):
            angle = (i / 4) * math.pi * 2 + angle_offset
            vehicles.append({
                "id":           f"ai_{i}",
                "color":        color,
                "x":            math.cos(angle) * (HALF - 10),
                "z":            math.sin(angle) * (HALF - 10),
                "y":            float(self._rng.uniform(VEHICLE_HEIGHT_MIN,
                                                        VEHICLE_HEIGHT_MAX)),
                "angle":        math.atan2(-math.sin(angle), -math.cos(angle)),
                "alive":        True,
                "respawnTimer": 0,
                "isAI":         True,
                "stuckCounter": 0,
                "lastX":        0.0,
                "lastZ":        0.0,
            })

        return {
            "tick":             0,
            "gameOver":         False,
            "gameOverReason":   "",
            "trees":            trees,
            "vehicles":         vehicles,
            "bullets":          [],
            "hunterYaw":        0.0,
            "hunterPitch":      0.0,
            "hunterRoll":       0.0,
            "hunterScore":      0,
            "nextAIIndex":      len(_AI_COLORS),
            "total_spawned_ai": len(_AI_COLORS),  # unique vehicles spawned this episode
            "wave_capacity":    INITIAL_WAVE_SIZE, # current-wave vehicle count (grows each wave)
            "waves_launched":   0,                 # periodic waves fired so far (max NUM_PERIODIC_WAVES)
        }

    # ── Action application ─────────────────────────────────────────────────

    def _apply_action(self, action: int) -> None:
        s = self.state

        if   action == 2: s["hunterYaw"]  += 0.10
        elif action == 3: s["hunterYaw"]  -= 0.10
        elif action == 4: s["hunterPitch"] = min(_PITCH_MAX, s["hunterPitch"] + 0.02)
        elif action == 5: s["hunterPitch"] = max(_PITCH_MIN, s["hunterPitch"] - 0.02)

        s["hunterYaw"] = _wrap_yaw(s["hunterYaw"])

        # Auto-aim: snap yaw AND pitch to the nearest alive vehicle.
        # _fire is set True only when a target exists, so fire_at_nearest
        # does nothing (no bullet, no score penalty) when all drones are down.
        _fire = (action == 1)
        if action == 6:
            alive = [v for v in s["vehicles"] if v["alive"]]
            if alive:
                n      = min(alive, key=lambda v: _dist(v["x"], v["z"], 0.0, 0.0))
                xz_d   = max(_dist(n["x"], n["z"], 0.0, 0.0), 0.5)
                s["hunterYaw"]   = math.atan2(n["z"], n["x"])
                raw_pitch        = math.atan2(n["y"] - 2.3, xz_d)
                s["hunterPitch"] = max(_PITCH_MIN, min(_PITCH_MAX, raw_pitch))
                _fire = True

        # Fire: spawn bullet from barrel tip in 3D aim direction
        if _fire:
            s["hunterScore"] -= 2
            yaw, pitch = s["hunterYaw"], s["hunterPitch"]
            cp = math.cos(pitch)
            dx = math.cos(yaw) * cp
            dy = math.sin(pitch)
            dz = math.sin(yaw) * cp
            s["bullets"].append({
                "x": dx * 3.0,  "y": 2.3 + dy * 3.0,  "z": dz * 3.0,
                "dx": dx,       "dy": dy,               "dz": dz,
                "dist": 0.0,
            })

    # ── Game tick ──────────────────────────────────────────────────────────

    def _tick(self) -> None:
        s = self.state
        if s["gameOver"]:
            return

        s["tick"] += 1

        if s["tick"] >= MAX_GAME_TICKS:
            s["gameOver"]       = True
            s["gameOverReason"] = "Time is up! Hunter survived 4 minutes!"
            return

        # Wave spawning — every WAVE_INTERVAL ticks, spawn int(capacity × WAVE_GROWTH)
        # additional vehicles so the field grows by ~50% each minute.
        # Waves fire at ticks 1200, 2400, 3600  (minutes 1, 2, 3).
        if (s["tick"] % WAVE_INTERVAL == 0
                and s["waves_launched"] < NUM_PERIODIC_WAVES):
            new_capacity = int(s["wave_capacity"] * WAVE_GROWTH)
            to_spawn     = new_capacity - s["wave_capacity"]
            for _ in range(to_spawn):
                ci = (s["nextAIIndex"] - len(_AI_COLORS)) % len(_EXTRA_COLORS)
                a  = float(self._rng.uniform(0, 2 * math.pi))
                s["vehicles"].append({
                    "id":           f"ai_{s['nextAIIndex']}",
                    "color":        _EXTRA_COLORS[ci],
                    "x":            math.cos(a) * (HALF - 10),
                    "z":            math.sin(a) * (HALF - 10),
                    "y":            float(self._rng.uniform(VEHICLE_HEIGHT_MIN,
                                                            VEHICLE_HEIGHT_MAX)),
                    "angle":        math.atan2(-math.sin(a), -math.cos(a)),
                    "alive":        True,
                    "respawnTimer": 0,
                    "isAI":         True,
                    "stuckCounter": 0,
                    "lastX":        0.0,
                    "lastZ":        0.0,
                })
                s["nextAIIndex"]      += 1
                s["total_spawned_ai"] += 1
            s["wave_capacity"]  = new_capacity
            s["waves_launched"] += 1

        # AI movement and reach check
        for v in s["vehicles"]:
            if s["gameOver"]:
                break
            if v["alive"]:
                _steer_ai(v, self._tree_xs, self._tree_zs, s["vehicles"], self._rng)
                if _dist(v["x"], v["z"], 0.0, 0.0) < REACH_DIST:
                    s["gameOver"]       = True
                    s["gameOverReason"] = (
                        f"{v['color'].upper()} vehicle reached the Hunter!"
                    )
            else:
                v["respawnTimer"] -= 1
                if v["respawnTimer"] <= 0:
                    a          = float(self._rng.uniform(0, 2 * math.pi))
                    v["x"]     = math.cos(a) * (HALF - 10)
                    v["z"]     = math.sin(a) * (HALF - 10)
                    v["angle"] = math.atan2(-v["z"], -v["x"])
                    v["alive"] = True
                    v["stuckCounter"] = 0
                    # Y is preserved — vehicle keeps its altitude through respawn

        if s["gameOver"]:
            return

        # Bullet physics + 3D hit detection
        surviving = []
        for b in s["bullets"]:
            b["x"]    += b["dx"] * BULLET_SPEED
            b["y"]    += b["dy"] * BULLET_SPEED
            b["z"]    += b["dz"] * BULLET_SPEED
            b["dist"] += BULLET_SPEED
            if b["dist"] > BULLET_MAX_DIST:               continue
            if abs(b["x"]) > HALF or abs(b["z"]) > HALF: continue
            if b["y"] < -2.0 or b["y"] > 80.0:           continue
            # Trees are solid XZ pillars — no height exemption.
            if _tree_blocked_fast(self._tree_xs, self._tree_zs, b["x"], b["z"], 0.3): continue
            # 3D hit detection — single target per shot.
            hit = False
            for v in s["vehicles"]:
                if not v["alive"]:
                    continue
                if _dist3(b["x"], b["y"], b["z"], v["x"], v["y"], v["z"]) < HIT_DIST:
                    v["alive"]        = False
                    v["respawnTimer"] = RESPAWN_TICKS_AI
                    s["hunterScore"] += 20
                    hit = True
                    break
            if not hit:
                surviving.append(b)
        s["bullets"] = surviving

        # Win: all periodic waves launched AND all AI simultaneously down.
        # Prevents an early win from the initial 4 drones; agent must hold until
        # all 3 periodic waves have fired (minute 3) and clear the field.
        all_ai = [v for v in s["vehicles"] if v["isAI"]]
        if (all_ai
                and all(not v["alive"] for v in all_ai)
                and s["waves_launched"] >= NUM_PERIODIC_WAVES
                and s["tick"] > 20):
            s["gameOver"]       = True
            s["gameOverReason"] = "All AI vehicles eliminated! Hunter wins!"

    # ── Reward ─────────────────────────────────────────────────────────────

    def _compute_reward(self, action: int,
                        prev_alive: int, curr_alive: int) -> float:
        s      = self.state
        reward = 0.0

        reward += max(0, prev_alive - curr_alive) * 20.0

        if action in (1, 6):
            reward -= 2.0

        alive = [v for v in s["vehicles"] if v["alive"]]
        if alive:
            nd = min(math.sqrt(v["x"] ** 2 + v["z"] ** 2) for v in alive)
            if nd < 20:
                reward -= (20.0 - nd) * 0.05

        if s["gameOver"]:
            if "reached" in s["gameOverReason"]:
                reward -= 100.0
            elif "eliminated" in s["gameOverReason"]:
                reward += 200.0
            # time-limit survival: no bonus

        return reward

    # ── Info dict ──────────────────────────────────────────────────────────

    def _build_info(self) -> dict:
        s = self.state
        return {
            "tick":              s["tick"],
            "hunterScore":       s["hunterScore"],
            "alive_ai":          sum(1 for v in s["vehicles"]
                                     if v["isAI"] and v["alive"]),
            "total_ai":          sum(1 for v in s["vehicles"] if v["isAI"]),
            "total_spawned_ai":  s["total_spawned_ai"],
            "wave":              s["waves_launched"],
            "wave_capacity":     s["wave_capacity"],
            "bullets_in_flight": len(s["bullets"]),
            "gameOverReason":    s["gameOverReason"],
        }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — GYMNASIUM WRAPPER  (thin shell around GameEngine)
# ═══════════════════════════════════════════════════════════════════════════

class ShooterEnv(gym.Env):
    """
    Farama Gymnasium environment — thin wrapper around GameEngine.

    Inputs (agent → environment)
    ─────────────────────────────
    reset(seed, options) → (obs, info)
    step(action)         → (obs, reward, terminated, truncated, info)
      action ∈ Discrete(7)   see ACTION_NAMES

    Outputs (environment → agent)
    ──────────────────────────────
    obs        Box(169,) float32  — see OBS class for feature layout
    reward     float
    terminated bool               game-over event
    truncated  bool               tick limit reached
    info       dict               tick, hunterScore, alive_ai, total_ai,
                                  bullets_in_flight, gameOverReason

    Parameters
    ──────────
    render_mode : "human" | "rgb_array" | None

    Rendering
    ─────────
    Requires PyOpenGL when render_mode is not None:
        pip install PyOpenGL PyOpenGL_accelerate
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": TICK_RATE}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )
        # RL agents get Discrete(6); human play adds action 6 (auto-aim)
        n_actions = NUM_ACTIONS if render_mode == "human" else NUM_ACTIONS - 1
        self.action_space = spaces.Discrete(n_actions)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._engine: Optional[GameEngine] = None
        self._window  = None
        self._clock   = None
        self._hud_font = None
        self._WINDOW_SIZE = 700
        # Optional callable() inserted just before pygame.display.flip().
        # Called with no arguments — use OpenGL calls or gl_begin_2d/gl_blit
        # helpers from this module to draw overlays inside the hook.
        self._pre_flip_hook = None

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None,
              options: Optional[dict] = None):
        super().reset(seed=seed)
        self._engine = GameEngine(self.np_random)
        self._engine.reset()
        if self.render_mode == "human":
            self._render_frame()
        return self._engine.get_obs(), self._engine._build_info()

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._engine.step(int(action))
        if self.render_mode == "human":
            self._render_frame()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self._window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
        self._window   = None
        self._clock    = None
        self._hud_font = None
        self._engine   = None

    # ── PyOpenGL 3D renderer ───────────────────────────────────────────────

    def _render_frame(self):
        import pygame

        if self.render_mode is not None and not _OPENGL_OK:
            raise ImportError(
                "PyOpenGL is required for rendering.\n"
                "  pip install PyOpenGL PyOpenGL_accelerate"
            )

        W = self._WINDOW_SIZE
        s = self._engine.state

        # Initialise window + OpenGL state once
        if self._window is None:
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode(
                (W, W), pygame.DOUBLEBUF | pygame.OPENGL
            )
            pygame.display.set_caption("Shooter-v0  |  Hunter RL Environment")
            self._clock    = pygame.time.Clock()
            self._hud_font = pygame.font.SysFont("monospace", 14)

            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(50.0, 1.0, 0.5, 1500.0)
            glMatrixMode(GL_MODELVIEW)

        # Clear frame
        glClearColor(0.04, 0.06, 0.12, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Fixed overhead-angle camera
        glLoadIdentity()
        gluLookAt(0.0, 130.0, 85.0,   # eye position
                  0.0,   5.0,  0.0,   # look-at target
                  0.0,   1.0,  0.0)   # up vector

        # 3D scene
        _gl_draw_floor()
        _gl_draw_trees(s["trees"])
        _gl_draw_vehicles(s["vehicles"])
        _gl_draw_bullets(s["bullets"])
        _gl_draw_hunter(s["hunterYaw"], s["hunterPitch"])

        # Built-in HUD overlay
        _gl_draw_hud(s, self._hud_font, W)

        # External overlay hook (e.g. play_shooter.py controls panel)
        if self._pre_flip_hook is not None:
            self._pre_flip_hook()

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
        else:
            # rgb_array: drain the event queue so the OS doesn't mark the
            # window as unresponsive when stepping headlessly at speed
            pygame.event.pump()
            # rgb_array: read back the OpenGL framebuffer
            glReadBuffer(GL_BACK)
            data = glReadPixels(0, 0, W, W, GL_RGB, GL_UNSIGNED_BYTE)
            img  = np.frombuffer(data, dtype=np.uint8).reshape(W, W, 3)
            return np.flipud(img)
