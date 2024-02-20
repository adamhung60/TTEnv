from gymnasium.envs.registration import register

register(
     id="TT",
     entry_point="tt_env.envs:TTEnv",
)