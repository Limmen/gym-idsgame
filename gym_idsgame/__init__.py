"""
Register OpenAI Envs
"""
from gym.envs.registration import register

# -------- Version 0 ------------

# [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random defender
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 0
register(
    id='idsgame-random_defense-v0',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV0Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 0
register(
    id='idsgame-minimal_defense-v0',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV0Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random attacker
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 0
register(
    id='idsgame-random_attack-v0',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV0Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 0
register(
    id='idsgame-maximal_attack-v0',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV0Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 0
register(
    id='idsgame-v0',
    entry_point='gym_idsgame.envs:IdsGameV0Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# -------- Version 1 ------------

# [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random defender
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 1
register(
    id='idsgame-random_defense-v1',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV1Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 1
register(
    id='idsgame-minimal_defense-v1',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV1Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random attacker
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 1
register(
    id='idsgame-random_attack-v1',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV1Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 1
register(
    id='idsgame-maximal_attack-v1',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV1Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 1
register(
    id='idsgame-v1',
    entry_point='gym_idsgame.envs:IdsGameV1Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# -------- Version 2 ------------

# [AttackerEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, random defender
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 2
register(
    id='idsgame-random_defense-v2',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV2Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 2
register(
    id='idsgame-minimal_defense-v2',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV2Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, random attacker
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 2
register(
    id='idsgame-random_attack-v2',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV2Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 2
register(
    id='idsgame-maximal_attack-v2',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV2Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 1 layer, 2 servers per layer, 10 attack-defense-values
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 2
register(
    id='idsgame-v2',
    entry_point='gym_idsgame.envs:IdsGameV2Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# -------- Version 3 ------------

# [AttackerEnv] 2 layers, 3 servers per layer, 10 attack-defense-values, random defender
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 3
register(
    id='idsgame-random_defense-v3',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV3Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 2 layer, 3 servers per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 3
register(
    id='idsgame-minimal_defense-v3',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV3Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 2 layers, 3 servers per layer, 10 attack-defense-values, random attacker
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 3
register(
    id='idsgame-random_attack-v3',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV3Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 2 layers, 3 servers per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 3
register(
    id='idsgame-maximal_attack-v3',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV3Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 2 layers, 3 servers per layer, 10 attack-defense-values
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 3
register(
    id='idsgame-v3',
    entry_point='gym_idsgame.envs:IdsGameV3Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# -------- Version 4 ------------

# [AttackerEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, random defender
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 4
register(
    id='idsgame-random_defense-v4',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV4Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 4 layer, 5 servers per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 4
register(
    id='idsgame-minimal_defense-v4',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV4Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, random attacker
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 4
register(
    id='idsgame-random_attack-v4',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV4Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 4
register(
    id='idsgame-maximal_attack-v4',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV4Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 4 layers, 5 servers per layer, 10 attack-defense-values
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 4
register(
    id='idsgame-v4',
    entry_point='gym_idsgame.envs:IdsGameV4Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# -------- Version 5 ------------

# [AttackerEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, random defender, connected layers
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 5
register(
    id='idsgame-random_defense-v5',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV5Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 4 layer, 5 servers per layer, 10 attack-defense-values,
# defender following the "defend minimal strategy", connected layers
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 5
register(
    id='idsgame-minimal_defense-v5',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV5Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, random attacker, connected layers
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 5
register(
    id='idsgame-random_attack-v5',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV5Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 4 layers, 5 servers per layer, 10 attack-defense-values,
# attacker following the "attack maximal strategy", connected layers
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 5
register(
    id='idsgame-maximal_attack-v5',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV5Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, connected layers
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 5
register(
    id='idsgame-v5',
    entry_point='gym_idsgame.envs:IdsGameV5Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# -------- Version 6 ------------

# [AttackerEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, random defender, connected layers
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Dense
# [Version] 6
register(
    id='idsgame-random_defense-v6',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV6Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 4 layer, 5 servers per layer, 10 attack-defense-values,
# defender following the "defend minimal strategy", connected layers
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Dense
# [Version] 6
register(
    id='idsgame-minimal_defense-v6',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV6Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, random attacker, connected layers
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Dense
# [Version] 6
register(
    id='idsgame-random_attack-v6',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV6Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 4 layers, 5 servers per layer, 10 attack-defense-values,
# attacker following the "attack maximal strategy", connected layers
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Dense
# [Version] 6
register(
    id='idsgame-maximal_attack-v6',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV6Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, connected layers
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Dense
# [Version] 5
register(
    id='idsgame-v6',
    entry_point='gym_idsgame.envs:IdsGameV6Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)