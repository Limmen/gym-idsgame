"""
Register OpenAI Envs
"""
from gym.envs.registration import register

# -------- Version 0 ------------

# [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random defender
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 0
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_defense-v0',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV0Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 0
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-minimal_defense-v0',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV0Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random attacker
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 0
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_attack-v0',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV0Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 0
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-maximal_attack-v0',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV0Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 0
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
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
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_defense-v1',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV1Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 1
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-minimal_defense-v1',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV1Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random attacker
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 1
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_attack-v1',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV1Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 1
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-maximal_attack-v1',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV1Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 1
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
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
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_defense-v2',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV2Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 2
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-minimal_defense-v2',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV2Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, random attacker
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 2
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_attack-v2',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV2Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 2
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-maximal_attack-v2',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV2Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 1 layer, 2 servers per layer, 10 attack-defense-values
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 2
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
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
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_defense-v3',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV3Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 2 layer, 3 servers per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 3
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-minimal_defense-v3',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV3Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 2 layers, 3 servers per layer, 10 attack-defense-values, random attacker
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 3
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_attack-v3',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV3Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 2 layers, 3 servers per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 3
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-maximal_attack-v3',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV3Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 2 layers, 3 servers per layer, 10 attack-defense-values
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 3
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
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
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_defense-v4',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV4Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 4 layer, 5 servers per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 4
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-minimal_defense-v4',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV4Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, random attacker
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 4
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_attack-v4',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV4Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 4
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-maximal_attack-v4',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV4Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 4 layers, 5 servers per layer, 10 attack-defense-values
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 4
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
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
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_defense-v5',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV5Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 4 layers, 5 servers per layer, 10 attack-defense-values,
# defender following the "defend minimal strategy", connected layers
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 5
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-minimal_defense-v5',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV5Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, random attacker, connected layers
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 5
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
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
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-maximal_attack-v5',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV5Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, connected layers
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Sparse
# [Version] 5
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
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
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
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
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-minimal_defense-v6',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV6Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, random attacker, connected layers
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Dense
# [Version] 6
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
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
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-maximal_attack-v6',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV6Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 4 layers, 5 servers per layer, 10 attack-defense-values, connected layers
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Dense
# [Version] 5
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-v6',
    entry_point='gym_idsgame.envs:IdsGameV6Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# -------- Version 7 ------------

# [AttackerEnv] 2 layers, 3 servers per layer, 10 attack-defense-values, random defender
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Dense
# [Version] 7
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_defense-v7',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV7Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 2 layer, 3 servers per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Dense
# [Version] 7
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-minimal_defense-v7',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV7Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 2 layers, 3 servers per layer, 10 attack-defense-values, random attacker
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Dense
# [Version] 7
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_attack-v7',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV7Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 2 layers, 3 servers per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Dense
# [Version] 7
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-maximal_attack-v7',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV7Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 2 layers, 3 servers per layer, 10 attack-defense-values
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Dense
# [Version] 7
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-v7',
    entry_point='gym_idsgame.envs:IdsGameV7Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# -------- Version 8 ------------

# [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random defender
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Rewards] Dense
# [Version] 8
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_defense-v8',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV8Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Rewards] Dense
# [Version] 8
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-minimal_defense-v8',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV8Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random attacker
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Rewards] Dense
# [Version] 8
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_attack-v8',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV8Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Rewards] Dense
# [Version] 8
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-maximal_attack-v8',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV8Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Rewards] Dense
# [Version] 8
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-v8',
    entry_point='gym_idsgame.envs:IdsGameV8Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# -------- Version 9 ------------

# [AttackerEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, random defender
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Dense
# [Version] 9
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_defense-v9',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV9Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Dense
# [Version] 9
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-minimal_defense-v9',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV9Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, random attacker
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Dense
# [Version] 9
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_attack-v9',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV9Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 2 servers per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Dense
# [Version] 9
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-maximal_attack-v9',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV9Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 1 layer, 2 servers per layer, 10 attack-defense-values
# [Initial State] Defense: 4, Attack:0, Num vulnerabilities: 4, Det: 3, Vulnerability value: 0
# [Rewards] Dense
# [Version] 9
# [Observations] partially observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-v9',
    entry_point='gym_idsgame.envs:IdsGameV9Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)


# -------- Version 10 ------------

# [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random defender
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Rewards] Dense
# [Version] 10
# [Observations] fully observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_defense-v10',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV10Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Rewards] Dense
# [Version] 10
# [Observations] fully observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-minimal_defense-v10',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV10Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values, random attacker
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Rewards] Dense
# [Version] 10
# [Observations] fully observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_attack-v10',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV10Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Rewards] Dense
# [Version] 10
# [Observations] fully observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-maximal_attack-v10',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV10Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 1 layer, 1 server per layer, 10 attack-defense-values
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
# [Rewards] Dense
# [Version] 10
# [Observations] fully observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-v10',
    entry_point='gym_idsgame.envs:IdsGameV10Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# -------- Version 11 ------------

# [AttackerEnv] 0 layers, 1 server per layer, 2 attack-defense-values
# [Initial State] Defense: 0, Attack:0, Num vulnerabilities: 0, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 11
# [Observations] fully observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_defense-v11',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV11Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 0 layers, 1 server per layer, 2 attack-defense-values
# [Initial State] Defense: 0, Attack:0, Num vulnerabilities: 0, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 11
# [Observations] fully observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-minimal_defense-v11',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV11Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 0 layers, 1 server per layer, 2 attack-defense-values
# [Initial State] Defense: 0, Attack:0, Num vulnerabilities: 0, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 11
# [Observations] fully observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_attack-v11',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV11Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 0 layers, 1 server per layer, 2 attack-defense-values
# [Initial State] Defense: 0, Attack:0, Num vulnerabilities: 0, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 11
# [Observations] fully observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-maximal_attack-v11',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV11Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 0 layers, 1 server per layer, 2 attack-defense-values
# [Initial State] Defense: 0, Attack:0, Num vulnerabilities: 0, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 11
# [Observations] fully observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-v11',
    entry_point='gym_idsgame.envs:IdsGameV11Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# -------- Version 12 ------------

# [AttackerEnv] 0 layers, 1 server per layer, 2 attack-defense-values
# [Initial State] Defense: (0-1) (random), Attack: (0-1) (random), Num vulnerabilities: 0, Det: (0-1) (random),
# Vulnerability value: 0
# [Rewards] Dense
# [Version] 12
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_defense-v12',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV12Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 0 layers, 1 server per layer, 2 attack-defense-values
# [Initial State] Defense: (0-1) (random), Attack: (0-1) (random), Num vulnerabilities: 0, Det: (0-1) (random),
# Vulnerability value: 0
# [Rewards] Dense
# [Version] 12
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-minimal_defense-v12',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV12Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 0 layers, 1 server per layer, 2 attack-defense-values
# [Initial State] Defense: (0-1) (random), Attack: (0-1) (random), Num vulnerabilities: 0, Det: (0-1) (random),
# Vulnerability value: 0
# [Rewards] Dense
# [Version] 12
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_attack-v12',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV12Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 0 layers, 1 server per layer, 2 attack-defense-values
# [Initial State] Defense: (0-1) (random), Attack: (0-1) (random), Num vulnerabilities: 0, Det: (0-1) (random),
# Vulnerability value: 0
# [Rewards] Dense
# [Version] 12
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-maximal_attack-v12',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV12Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 0 layers, 1 server per layer, 2 attack-defense-values
# [Initial State] Defense: (0-1) (random), Attack: (0-1) (random), Num vulnerabilities: 0, Det: (0-1) (random),
# Vulnerability value: 0
# [Rewards] Dense
# [Version] 12
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-v12',
    entry_point='gym_idsgame.envs:IdsGameV12Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# -------- Version 13 ------------

# [AttackerEnv] 0 layers, 1 server per layer, 2 attack-defense-values
# [Initial State] Defense: 0, Attack:0, Num vulnerabilities: 0, Det: 10, Vulnerability value: 0
# [Rewards] Dense
# [Version] 13
# [Observations] fully observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_defense-v13',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV13Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 0 layers, 1 server per layer, 2 attack-defense-values
# [Initial State] Defense: 0, Attack:0, Num vulnerabilities: 0, Det: 10, Vulnerability value: 0
# [Rewards] Dense
# [Version] 13
# [Observations] fully observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-minimal_defense-v13',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV13Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 0 layers, 1 server per layer, 2 attack-defense-values
# [Initial State] Defense: 0, Attack:0, Num vulnerabilities: 0, Det: 10, Vulnerability value: 0
# [Rewards] Dense
# [Version] 13
# [Observations] fully observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_attack-v13',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV13Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 0 layers, 1 server per layer, 2 attack-defense-values
# [Initial State] Defense: 0, Attack:0, Num vulnerabilities: 0, Det: 10, Vulnerability value: 0
# [Rewards] Dense
# [Version] 13
# [Observations] fully observed
# [Environment] Deterministic
# [Attacker Starting Position] Start node
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-maximal_attack-v13',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV13Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 0 layers, 1 server per layer, 2 attack-defense-values
# [Initial State] Defense: 0, Attack:0, Num vulnerabilities: 0, Det: 10, Vulnerability value: 0
# [Rewards] Dense
# [Version] 13
# [Observations] fully observed
# [Attacker Starting Position] Start node
# [Environment] Deterministic
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-v13',
    entry_point='gym_idsgame.envs:IdsGameV13Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# -------- Version 14 ------------

# [AttackerEnv] 1 layer, 1 server per layer, 4 attack-defense-values, random defender
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 14
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Start node
# [Local View] Yes
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_defense-v14',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV14Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 1 layer, 1 server per layer, 4 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 14
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Start node
# [Local View] Yes
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-minimal_defense-v14',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV14Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 4 attack-defense-values, random attacker
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 14
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Start node
# [Local View] Yes
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_attack-v14',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV14Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 4 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 14
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Start node
# [Local View] Yes
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-maximal_attack-v14',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV14Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 1 layer, 1 server per layer, 4 attack-defense-values
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 14
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Start node
# [Local View] Yes
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-v14',
    entry_point='gym_idsgame.envs:IdsGameV14Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# -------- Version 15 ------------

# [AttackEnv] 1 layer, 1 server per layer, 4 attack-defense-values
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 15
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Start node
# [Local View] Yes
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_defense-v15',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV15Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackEnv] 1 layer, 1 server per layer, 4 attack-defense-values
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 15
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Start node
# [Local View] Yes
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-minimal_defense-v15',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV15Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 4 attack-defense-values
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 15
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Start node
# [Local View] Yes
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_attack-v15',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV15Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 4 attack-defense-values
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 15
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Start node
# [Local View] Yes
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-maximal_attack-v15',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV15Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 1 layer, 1 server per layer, 4 attack-defense-values
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 15
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Start node
# [Local View] Yes
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-v15',
    entry_point='gym_idsgame.envs:IdsGameV15Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)


# -------- Version 16 ------------

# [AttackerEnv] 1 layer, 1 server per layer, 4 attack-defense-values, random defender
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 16
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] No
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_defense-v16',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV16Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 1 layer, 1 server per layer, 4 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 16
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] No
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-minimal_defense-v16',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV16Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 4 attack-defense-values, random attacker
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 16
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] No
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_attack-v16',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV16Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 4 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 16
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] No
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-maximal_attack-v16',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV16Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 1 layer, 1 server per layer, 4 attack-defense-values
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 16
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] No
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-v16',
    entry_point='gym_idsgame.envs:IdsGameV16Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# -------- Version 17 ------------

# [AttackerEnv] 1 layer, 1 server per layer, 4 attack-defense-values, random defender
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 17
# [Observations] partially observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] No
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_defense-v17',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV17Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 1 layer, 1 server per layer, 4 attack-defense-values, defender following the "defend minimal strategy"
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 17
# [Observations] partially observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] No
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-minimal_defense-v17',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV17Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 4 attack-defense-values, random attacker
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 17
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] No
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-random_attack-v17',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV17Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 4 attack-defense-values, attacker following the "attack maximal strategy"
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 17
# [Observations] partially observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] No
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-maximal_attack-v17',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV17Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 1 layer, 1 server per layer, 4 attack-defense-values
# [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 0
# [Rewards] Dense
# [Version] 17
# [Observations] partially observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] No
# [Reconnaissance activities] disabled
# [Reconnaissance bool features] No
register(
    id='idsgame-v17',
    entry_point='gym_idsgame.envs:IdsGameV17Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# -------- Version 18 ------------

# [AttackerEnv] 1 layer, 1 server per layer, 7 attack-defense-values, random defender, random defender
# [Initial State] Defense: 7, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 1
# [Rewards] Dense
# [Version] 18
# [Observations] partially observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] Yes
# [Reconnaissance activities] enabled
# [Reconnaissance bool features] Yes
register(
    id='idsgame-random_defense-v18',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV18Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 1 layer, 1 server per layer, 7 attack-defense-values, random defender, defender following the "defend minimal strategy"
# [Initial State] Defense: 7, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 1
# [Rewards] Dense
# [Version] 18
# [Observations] partially observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] Yes
# [Reconnaissance activities] enabled
# [Reconnaissance bool features] Yes
register(
    id='idsgame-minimal_defense-v18',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV18Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 7 attack-defense-values, random defender, random attacker
# [Initial State] Defense: 7, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 1
# [Rewards] Dense
# [Version] 18
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] Yes
# [Reconnaissance activities] enabled
# [Reconnaissance bool features] Yes
register(
    id='idsgame-random_attack-v18',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV18Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 7 attack-defense-values, random defender, attacker following the "attack maximal strategy"
# [Initial State] Defense: 7, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 1
# [Rewards] Dense
# [Version] 18
# [Observations] partially observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] Yes
# [Reconnaissance activities] enabled
# [Reconnaissance bool features] Yes
register(
    id='idsgame-maximal_attack-v18',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV18Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 1 layer, 1 server per layer, 7 attack-defense-values, random defender
# [Initial State] Defense: 7, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 1
# [Rewards] Dense
# [Version] 18
# [Observations] partially observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] Yes
# [Reconnaissance activities] enabled
# [Reconnaissance bool features] Yes
register(
    id='idsgame-v18',
    entry_point='gym_idsgame.envs:IdsGameV18Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# -------- Version 19 ------------

# [AttackerEnv] 1 layer, 1 server per layer, 7 attack-defense-values, random defender, random defender
# [Initial State] Defense: 7, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 1
# [Rewards] Dense
# [Version] 19
# [Observations] partially observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] Yes
# [Reconnaissance activities] enabled
# [Reconnaissance bool features] Yes
register(
    id='idsgame-random_defense-v19',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV19Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 1 layer, 1 server per layer, 7 attack-defense-values, random defender, defender following the "defend minimal strategy"
# [Initial State] Defense: 7, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 1
# [Rewards] Dense
# [Version] 19
# [Observations] partially observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] Yes
# [Reconnaissance activities] enabled
# [Reconnaissance bool features] Yes
register(
    id='idsgame-minimal_defense-v19',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV19Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 7 attack-defense-values, random defender, random attacker
# [Initial State] Defense: 7, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 1
# [Rewards] Dense
# [Version] 19
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] Yes
# [Reconnaissance activities] enabled
# [Reconnaissance bool features] Yes
register(
    id='idsgame-random_attack-v19',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV19Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 7 attack-defense-values, random defender, attacker following the "attack maximal strategy"
# [Initial State] Defense: 7, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 1
# [Rewards] Dense
# [Version] 19
# [Observations] partially observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] Yes
# [Reconnaissance activities] enabled
# [Reconnaissance bool features] Yes
register(
    id='idsgame-maximal_attack-v19',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV19Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 1 layer, 1 server per layer, 7 attack-defense-values, random defender
# [Initial State] Defense: 7, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 1
# [Rewards] Dense
# [Version] 19
# [Observations] partially observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] Yes
# [Reconnaissance activities] enabled
# [Reconnaissance bool features] Yes
register(
    id='idsgame-v19',
    entry_point='gym_idsgame.envs:IdsGameV19Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)


# -------- Version 20 ------------

# [AttackerEnv] 1 layer, 1 server per layer, 7 attack-defense-values, random defender, random defender
# [Initial State] Defense: 7, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 1
# [Rewards] Dense
# [Version] 20
# [Observations] partially observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] Yes
# [Reconnaissance activities] enabled
# [Reconnaissance bool features] Yes
register(
    id='idsgame-random_defense-v20',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV20Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 1 layer, 1 server per layer, 7 attack-defense-values, random defender, defender following the "defend minimal strategy"
# [Initial State] Defense: 7, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 1
# [Rewards] Dense
# [Version] 20
# [Observations] partially observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] Yes
# [Reconnaissance activities] enabled
# [Reconnaissance bool features] Yes
register(
    id='idsgame-minimal_defense-v20',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV20Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 7 attack-defense-values, random defender, random attacker
# [Initial State] Defense: 7, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 1
# [Rewards] Dense
# [Version] 20
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] Yes
# [Reconnaissance activities] enabled
# [Reconnaissance bool features] Yes
register(
    id='idsgame-random_attack-v20',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV20Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 7 attack-defense-values, random defender, attacker following the "attack maximal strategy"
# [Initial State] Defense: 7, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 1
# [Rewards] Dense
# [Version] 20
# [Observations] partially observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] Yes
# [Reconnaissance activities] enabled
# [Reconnaissance bool features] Yes
register(
    id='idsgame-maximal_attack-v20',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV20Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 1 layer, 1 server per layer, 7 attack-defense-values, random defender
# [Initial State] Defense: 7, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 1
# [Rewards] Dense
# [Version] 20
# [Observations] partially observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] Yes
# [Reconnaissance activities] enabled
# [Reconnaissance bool features] Yes
register(
    id='idsgame-v20',
    entry_point='gym_idsgame.envs:IdsGameV20Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# -------- Version 21 ------------

# [AttackerEnv] 1 layer, 1 server per layer, 7 attack-defense-values, random defender, random defender
# [Initial State] Defense: 7, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 1
# [Rewards] Dense
# [Version] 21
# [Observations] partially observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] Yes
# [Reconnaissance activities] enabled
# [Reconnaissance bool features] Yes
register(
    id='idsgame-random_defense-v21',
    entry_point='gym_idsgame.envs:IdsGameRandomDefenseV21Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackerEnv] 1 layer, 1 server per layer, 7 attack-defense-values, random defender, defender following the "defend minimal strategy"
# [Initial State] Defense: 7, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 1
# [Rewards] Dense
# [Version] 21
# [Observations] partially observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] Yes
# [Reconnaissance activities] enabled
# [Reconnaissance bool features] Yes
register(
    id='idsgame-minimal_defense-v21',
    entry_point='gym_idsgame.envs:IdsGameMinimalDefenseV21Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 7 attack-defense-values, random defender, random attacker
# [Initial State] Defense: 7, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 1
# [Rewards] Dense
# [Version] 21
# [Observations] fully observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] Yes
# [Reconnaissance activities] enabled
# [Reconnaissance bool features] Yes
register(
    id='idsgame-random_attack-v21',
    entry_point='gym_idsgame.envs:IdsGameRandomAttackV21Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [DefenseEnv] 1 layer, 1 server per layer, 7 attack-defense-values, random defender, attacker following the "attack maximal strategy"
# [Initial State] Defense: 7, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 1
# [Rewards] Dense
# [Version] 21
# [Observations] partially observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] Yes
# [Reconnaissance activities] enabled
# [Reconnaissance bool features] Yes
register(
    id='idsgame-maximal_attack-v21',
    entry_point='gym_idsgame.envs:IdsGameMaximalAttackV21Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)

# [AttackDefenseEnv] 1 layer, 1 server per layer, 7 attack-defense-values, random defender
# [Initial State] Defense: 7, Attack:0, Num vulnerabilities: 1, Det: 1, Vulnerability value: 1
# [Rewards] Dense
# [Version] 21
# [Observations] partially observed
# [Environment] Random
# [Attacker Starting Position] Random
# [Local View] Yes
# [Reconnaissance activities] enabled
# [Reconnaissance bool features] Yes
register(
    id='idsgame-v21',
    entry_point='gym_idsgame.envs:IdsGameV21Env',
    kwargs={'idsgame_config': None, 'save_dir': None, 'initial_state_path': None}
)