"""
Register OpenAI Envs
"""
from gym.envs.registration import register

# -------- Version 0 ------------

# [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random defender
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Version] 0
register(
    id='idsgame-random_defense-v0',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV0Env',
)

# [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Version] 0
register(
    id='idsgame-minimal_defense-v0',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV0Env',
)

# [DefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random attacker
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Version] 0
register(
    id='idsgame-random_attack-v0',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV0Env',
)

# [DefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Version] 0
register(
    id='idsgame-maximal_attack-v0',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV0Env',
)

# [AttackDefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Version] 0
register(
    id='idsgame-v0',
    entry_point='gym_idsgame.envs:IdsGameV0Env',
)

# -------- Version 1 ------------

# [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random defender
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Version] 1
register(
    id='idsgame-random_defense-v1',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV1Env',
)

# [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Version] 1
register(
    id='idsgame-minimal_defense-v1',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV1Env',
)

# [DefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random attacker
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Version] 1
register(
    id='idsgame-random_attack-v1',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV1Env',
)

# [DefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Version] 1
register(
    id='idsgame-maximal_attack-v1',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV1Env',
)

# [AttackDefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Version] 1
register(
    id='idsgame-v1',
    entry_point='gym_idsgame.envs:IdsGameV1Env',
)

# -------- Version 2 ------------

# [AttackerEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, random defender
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Version] 1
register(
    id='idsgame-random_defense-v2',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV2Env',
)

# [AttackerEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Version] 1
register(
    id='idsgame-minimal_defense-v2',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV2Env',
)

# [DefenseEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, random attacker
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Version] 1
register(
    id='idsgame-random_attack-v2',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV2Env',
)

# [DefenseEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Version] 1
register(
    id='idsgame-maximal_attack-v2',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV2Env',
)

# [AttackDefenseEnv] 1 layer, 2 servers per layer, 10 attack-defense-values
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Version] 1
register(
    id='idsgame-v2',
    entry_point='gym_idsgame.envs:IdsGameV2Env',
)

# -------- Version 3 ------------

# [AttackerEnv] 2 layers, 3 servers per layer, 10 attack-defense-values, random defender
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Version] 1
register(
    id='idsgame-random_defense-v3',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV3Env',
)

# [AttackerEnv] 2 layer, 3 servers per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Version] 1
register(
    id='idsgame-minimal_defense-v3',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV3Env',
)

# [DefenseEnv] 2 layers, 3 servers per layer, 10 attack-defense-values, random attacker
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Version] 1
register(
    id='idsgame-random_attack-v3',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV3Env',
)

# [DefenseEnv] 2 layers, 3 servers per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Version] 1
register(
    id='idsgame-maximal_attack-v3',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV3Env',
)

# [AttackDefenseEnv] 2 layers, 3 servers per layer, 10 attack-defense-values
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Version] 1
register(
    id='idsgame-v3',
    entry_point='gym_idsgame.envs:IdsGameV3Env',
)