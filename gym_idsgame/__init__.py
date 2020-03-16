from gym.envs.registration import register

register(
    id='idsgame-v1',
    entry_point='gym_idsgame.envs:IdsGameEnv',
)