# Deep-Reinforcement-Learning-Shooter-GuardianPPO
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.0-green.svg)](https://gymnasium.farama.org/)

Deep Reinforcement Learning for 3D Shooter-v0 Environment – Implemented and compared multiple DRL approaches: Vanilla PPO, PPO with Reward Shaping, and Behavioral Cloning + PPO. The project evolved through 30+ versions.

This project is part of our Master's studies at the Asian Institute of Technology (AIT).  
We built a 3D turret-shooter environment called **Shooter-v0** and trained four DRL agents: Vanilla PPO, PPO with Reward Shaping, BC + PPO, and our proposed **Guardian PPO** (BC + failure penalty + self-imitation learning).

Vanilla PPO gets 0 kills. Guardian PPO achieves ~24 kills per episode and a max score of 750 (update version is 1500 max score in plain environment).

---

## 🎮 Environment: Shooter-v0

- Stationary turret at origin, controls yaw & pitch in 3D
- Drones fly at altitudes 4–10 units, spawn in waves (4 → 6 → 9 → 13)
- 36 trees block bullets (solid XZ pillars)
- Observation: `float32[169]` (turret angles, 15 nearest drones, 10 bullets, threat summary)
- Actions: Discrete(6): `no-op, fire, yaw left/right, pitch up/down`

---

## AI Use Declaration
During the development of this project, we used AI tools for:

Language translation and sentence refinement

Code suggestions, debugging, and structural guidance

Writing assistance for the report and this README

Brainstorming and "vibe coding" support

However, all core model design, data collection, result interpretation, and final technical decisions were made by the authors (Paradorn & Jirapat), and all outputs have been manually verified


##  Citation
If you use this code or ideas, please cite:
Khanongsuwan, P., Datephanyawat, J. (2025).
BC-Guided Self-Imitation PPO for Shooter-v0.
GitHub. https://github.com/iHazelly/Deep-Reinforcement-Learning-Shooter-GuardianPPO



## Installation

**Python 3.10 or 3.11 recommended**

```bash
pip install -r requirements.txt

# Or install manually:
pip install gymnasium numpy torch stable-baselines3 pygame PyOpenGL tensorboard matplotlib

## Run the environment manually (keyboard control)
python environment/shooter/play_shooter.py

## Load and run Guardian PPO

from stable_baselines3 import PPO
import gymnasium as gym
import shooter

model = PPO.load("models/guardian_ppo.zip")
env = gym.make("Shooter-v0", render_mode="human")

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
env.close()

