"""
Register OpenAI Envs
"""
from gym.envs.registration import register

# [AttackEnv] 1 server per layer, 10 attack-defense-values, random defender
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Version] 0
register(
    id='idsgame-random_defense-v0',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV0Env',
)

# [DefenseEnv] 1 server per layer, 10 attack-defense-values, random attacker
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Version] 0
register(
    id='idsgame-random_attack-v0',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV0Env',
)

# [AttackDefenseEnv] 1 server per layer, 10 attack-defense-values
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Version] 0
register(
    id='idsgame-v0',
    entry_point='gym_idsgame.envs:IdsGameV0Env',
)

#
# register(
#     id='idsgame-attack_only-2l-1s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-3l-1s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-4l-1s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-5l-1s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )


# 2 servers per layer, 10 attack-defense-values
# register(
#     id='idsgame-attack_only-1l-2s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-2l-2s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-3l-2s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-4l-2s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-5l-2s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )

# 3 servers per layer, 10 attack-defense-values
# register(
#     id='idsgame-attack_only-1l-3s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-2l-3s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-3l-3s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-4l-3s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-5l-3s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )

# 4 servers per layer, 10 attack-defense-values
# register(
#     id='idsgame-attack_only-1l-4s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-2l-4s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-3l-4s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-4l-4s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-5l-4s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )

# 5 servers per layer, 10 attack-defense-values
# register(
#     id='idsgame-attack_only-1l-5s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-2l-5s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-3l-5s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-4l-5s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-5l-5s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )

# 6 servers per layer, 10 attack-defense-values
# register(
#     id='idsgame-attack_only-1l-6s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-2l-6s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-3l-6s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-4l-6s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-5l-6s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )

# 7 servers per layer, 10 attack-defense-values
# register(
#     id='idsgame-attack_only-1l-7s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-2l-7s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-3l-7s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-4l-7s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )
#
# register(
#     id='idsgame-attack_only-5l-7s-10ad-v0',
#     entry_point='gym_idsgame.envs:IdsGameEnv',
# )