"""
Register OpenAI Envs
"""
from gym.envs.registration import register

# -------- Version 0 ------------

# [AttackerEnv] 1 server per layer, 10 attack-defense-values, random defender
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Version] 0
register(
    id='idsgame-random_defense-v0',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV0Env',
)

# [AttackerEnv] 1 server per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Version] 0
register(
    id='idsgame-minimal_defense-v0',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV0Env',
)

# [DefenseEnv] 1 server per layer, 10 attack-defense-values, random attacker
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Version] 0
register(
    id='idsgame-random_attack-v0',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV0Env',
)

# [DefenseEnv] 1 server per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Version] 0
register(
    id='idsgame-maximal_attack-v0',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV0Env',
)

# [AttackDefenseEnv] 1 server per layer, 10 attack-defense-values
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Version] 0
register(
    id='idsgame-v0',
    entry_point='gym_idsgame.envs:IdsGameV0Env',
)

# -------- Version 1 ------------

# [AttackerEnv] 1 server per layer, 10 attack-defense-values, random defender
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Version] 1
register(
    id='idsgame-random_defense-v1',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV1Env',
)

# [AttackerEnv] 1 server per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Version] 1
register(
    id='idsgame-minimal_defense-v1',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV1Env',
)

# [DefenseEnv] 1 server per layer, 10 attack-defense-values, random attacker
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Version] 1
register(
    id='idsgame-random_attack-v1',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV1Env',
)

# [DefenseEnv] 1 server per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Version] 1
register(
    id='idsgame-maximal_attack-v1',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV1Env',
)

# [AttackDefenseEnv] 1 server per layer, 10 attack-defense-values
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Version] 1
register(
    id='idsgame-v1',
    entry_point='gym_idsgame.envs:IdsGameV1Env',
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