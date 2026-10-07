from gymnasium.envs.registration import register

register(
    id='online-ant-v0',
    entry_point='ogbench.online_locomotion.ant:AntEnv',
    max_episode_steps=1000,
)
register(
    id='online-humanoid-v0',
    entry_point='ogbench.online_locomotion.humanoid:HumanoidEnv',
    max_episode_steps=1000,
)
