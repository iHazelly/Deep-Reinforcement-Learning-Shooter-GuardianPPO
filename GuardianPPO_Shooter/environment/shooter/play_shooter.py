"""
play_shooter.py — Interactive keyboard play for the 3D ShooterEnv.

Launches the PyOpenGL window, lets you drive the Hunter turret with the
keyboard, and composites a semi-transparent controls + stats panel onto
the rendered frame via env._pre_flip_hook — called once per tick, just
before pygame.display.flip(), so there is no double-flip flicker.

Controls
--------
  ← / → (or A / D)   yaw left / right
  ↑ / ↓ (or W / S)   pitch up / down
  SPACE               fire
  F                   fire at nearest vehicle (auto-aim + fire)
  R                   reset episode
  ESC / Q             quit

Usage
-----
  python shooter/play_shooter.py [--seed SEED]
"""

import sys
import argparse
import pygame

sys.path.insert(0, __file__.replace("/shooter/play_shooter.py", ""))

from shooter.shooter_env import (
    ShooterEnv, ACTION_NAMES, NUM_PERIODIC_WAVES,
    gl_begin_2d, gl_end_2d, gl_fill_rect, gl_blit,
)


# ── Key → discrete-action mapping ─────────────────────────────────────────

_KEY_ACTION = [
    (pygame.K_SPACE,  1),
    (pygame.K_f,      6),
    (pygame.K_LEFT,   2), (pygame.K_a, 2),
    (pygame.K_RIGHT,  3), (pygame.K_d, 3),
    (pygame.K_UP,     4), (pygame.K_w, 4),
    (pygame.K_DOWN,   5), (pygame.K_s, 5),
]

_KEY_HELP = [
    ("← / A",   "yaw left"),
    ("→ / D",   "yaw right"),
    ("↑ / W",   "pitch up"),
    ("↓ / S",   "pitch down"),
    ("SPACE",   "fire"),
    ("F",       "auto-aim + fire"),
    ("R",       "reset episode"),
    ("ESC / Q", "quit"),
]

_COL_KEY  = 80
_COL_DESC = 105
_PANEL_W  = _COL_KEY + _COL_DESC + 16   # 201 px
_LINE_H   = 17
_PAD      = 7


def _read_action(keys) -> int:
    for key, action in _KEY_ACTION:
        if keys[key]:
            return action
    return 0


def _draw_overlay(font, hud: dict, W: int) -> None:
    """
    Render the controls + live-stats panel using OpenGL 2D helpers.
    Must be called inside env._pre_flip_hook (OpenGL context active,
    gl_begin_2d / gl_end_2d managed here).
    """
    action_name = ACTION_NAMES[hud["action"]]
    reward_col  = (100, 255, 130) if hud["reward"] >= 0 else (255, 100, 100)
    total_col   = (100, 255, 130) if hud["total"]  >= 0 else (255, 100, 100)

    # Build rows: (key_text, desc_text, key_color, desc_color)
    ctrl_rows = [
        (k, d, (230, 230, 100), (180, 180, 180)) for k, d in _KEY_HELP
    ]
    stat_rows = [
        ("Episode",  str(hud["episode"]),              (160, 210, 255), (160, 210, 255)),
        ("Action",   action_name,                      (150, 150, 150), (255, 200, 100)),
        ("Reward",   f"{hud['reward']:+.2f}",          (150, 150, 150), reward_col),
        ("Total R",  f"{hud['total']:+.1f}",           (150, 150, 150), total_col),
        ("Score",    str(hud["score"]),                                        (150, 150, 150), (200, 200, 200)),
        ("AI alive", str(hud["alive_ai"]),                                     (150, 150, 150), (200, 200, 200)),
        ("Wave",     f"{hud['wave']}/{NUM_PERIODIC_WAVES}  cap {hud['wave_capacity']}", (150, 150, 150), (200, 200, 200)),
    ]

    # Total rows: header + controls + divider-gap + stats
    n_rows  = 1 + len(ctrl_rows) + 1 + len(stat_rows)
    panel_h = _PAD + n_rows * _LINE_H + _PAD
    px      = W - _PANEL_W - 6
    py      = W - panel_h - 6   # bottom of panel in Y-up coords

    gl_begin_2d(W, W)

    # Background
    gl_fill_rect(px, py, _PANEL_W, panel_h, 0.03, 0.03, 0.03, 0.80)

    row = 0

    # ── Controls header ───────────────────────────────────────────────────
    y = py + panel_h - _PAD - (row + 1) * _LINE_H
    gl_blit("CONTROLS", px + _PAD, y, font, (255, 215, 60))
    row += 1

    for key_str, desc, kc, dc in ctrl_rows:
        y = py + panel_h - _PAD - (row + 1) * _LINE_H
        gl_blit(key_str, px + _PAD,            y, font, kc)
        gl_blit(desc,    px + _PAD + _COL_KEY, y, font, dc)
        row += 1

    # ── Divider gap ───────────────────────────────────────────────────────
    div_y = py + panel_h - _PAD - (row + 1) * _LINE_H + _LINE_H // 2
    # Use gl_fill_rect as a 1-pixel-high divider line
    gl_fill_rect(px + _PAD, div_y, _PANEL_W - 2 * _PAD, 2,
                 0.27, 0.27, 0.27, 1.0)
    row += 1

    # ── Live stats ────────────────────────────────────────────────────────
    for label, value, lc, vc in stat_rows:
        y = py + panel_h - _PAD - (row + 1) * _LINE_H
        gl_blit(label, px + _PAD,            y, font, lc)
        gl_blit(value, px + _PAD + _COL_KEY, y, font, vc)
        row += 1

    gl_end_2d()


def main():
    parser = argparse.ArgumentParser(description="Interactive ShooterEnv play")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = ShooterEnv(render_mode="human")
    obs, info = env.reset(seed=args.seed)

    # pygame is initialised inside the first _render_frame call above
    font = pygame.font.SysFont("monospace", 13)
    W    = env._WINDOW_SIZE

    hud = {
        "action":        0,
        "reward":        0.0,
        "total":         0.0,
        "episode":       1,
        "score":         info["hunterScore"],
        "alive_ai":      info["alive_ai"],
        "wave":          info["wave"],
        "wave_capacity": info["wave_capacity"],
    }

    def _pre_flip() -> None:
        _draw_overlay(font, hud, W)

    env._pre_flip_hook = _pre_flip

    running = True

    print("═" * 55)
    print("  Shooter-v0  |  3D Interactive Play")
    print("  ← → ↑ ↓ / WASD   SPACE=fire   F=auto-aim+fire")
    print("  R=reset           ESC/Q=quit")
    print("═" * 55)

    while running:

        # ── 1. Events ─────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    hud.update(action=0, reward=0.0, total=0.0, score=0,
                               alive_ai=info["alive_ai"],
                               wave=info["wave"],
                               wave_capacity=info["wave_capacity"])
                    hud["episode"] += 1
                    print(f"\n[Episode {hud['episode']}]  Manual reset")

        if not running:
            break

        # ── 2. Action ─────────────────────────────────────────────────────
        action       = _read_action(pygame.key.get_pressed())
        hud["action"] = action

        # ── 3. Step ───────────────────────────────────────────────────────
        obs, reward, terminated, truncated, info = env.step(action)

        # ── 4. Update HUD ─────────────────────────────────────────────────
        hud["reward"]        = reward
        hud["total"]        += reward
        hud["score"]         = info["hunterScore"]
        hud["alive_ai"]      = info["alive_ai"]
        hud["wave"]          = info["wave"]
        hud["wave_capacity"] = info["wave_capacity"]

        # ── 5. Episode end ────────────────────────────────────────────────
        if terminated or truncated:
            reason = info["gameOverReason"] or "truncated"
            print(
                f"[Episode {hud['episode']}]  {reason}\n"
                f"  ticks={info['tick']:5d}  score={info['hunterScore']:4d}"
                f"  total_reward={hud['total']:+.1f}"
            )
            obs, info = env.reset()
            hud.update(action=0, reward=0.0, total=0.0, score=0,
                       alive_ai=info["alive_ai"],
                       wave=info["wave"],
                       wave_capacity=info["wave_capacity"])
            hud["episode"] += 1

    env.close()
    print("\nSession ended.")


if __name__ == "__main__":
    main()
