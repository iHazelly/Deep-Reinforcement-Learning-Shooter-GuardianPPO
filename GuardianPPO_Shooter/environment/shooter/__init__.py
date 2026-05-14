from gymnasium.envs.registration import register

register(
    id="Shooter-v0",
    entry_point="shooter.shooter_env:ShooterEnv",
)
