# Experiments

## Experiments in Version 0 Environments 

### Training Experiments
- [tabular_q_learning_vs_random_defense-v0](training/v0/random_defense/tabular_q_learning/)
   * This is an experiment in the `random_defense-v0` environment. 
   An environment where the defender is following a random defense policy. 
   This experiment trains an attacker agent using tabular q-learning to act optimally in the given
   environment and defeat the random defender.
   
- [tabular_q_learning_vs_minimal_defense-v0](training/v0/minimal_defense/tabular_q_learning/)
   * This is an experiment in the `minimal_defense-v0` environment.  
   An environment where the defender is following the `defend_minimal` defense policy. 
   The `defend_minimal` policy entails that the defender will always 
   defend the attribute with the minimal value out of all of its neighbors.
   This experiment trains an attacker agent using tabular q-learning to act optimally in the given 
   environment and defeat the defender.       
   
- [random_attack_vs_tabular_q_learning-v0](training/v0/random_attack/tabular_q_learning/)
   * This is an experiment in the `random_attack-v0` environment.  
   An environment where the attack is following a random attack policy.  
   This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and defeat the random attacker.
   
- [maximal_attack_vs_tabular_q_learning-v0](training/v0/maximal_attack/tabular_q_learning/)
   * This is an experiment in the `maximal_attack-v0` environment.
   An environment where the attack is following the `attack_maximal` attack policy.
   The `attack_maximal` policy entails that the attacker will always attack the attribute with
   the maximum value out of all of its neighbors. The defender is implemented with a
   random defense policy.
   This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and detect the attacker.       

### Simulation Experiments   
- [random_vs_random-v0](simulations/v0/random_vs_random/)
   * This is an experiment in the `idsgame-v0` environment.  
   An environment where neither the attacker nor defender is part of the environment, i.e.
   it is intended for 2-agent simulations or RL training.
   In this experiment, the attacker is implemented with a random attack policy.
   Similarly, the defender is implemented with a random defense policy. 
   
- [random_vs_defend_minimal-v0](simulations/v0/random_vs_defend_minimal/)
   * This is an experiment in the `idsgame-v0` environment. 
   An environment where neither the attacker nor defender is part of the environment, i.e.
   it is intended for 2-agent simulations or RL training. 
   In this experiment, the attacker is implemented with a random attack policy.
   The defender is implemented with the  policy `defend_minimal`. 
   The `defend_minimal` policy entails that the defender will always
   defend the attribute with the minimal value out of all of its neighbors.   
   
- [attack_maximal_vs_defend_minimal-v0](simulations/v0/attack_maximal_vs_defend_minimal/)
   * This is an experiment in the `idsgame-v0` environment.  
   An environment where neither the attacker nor defender is part of the environment, i.e. 
   it is intended for 2-agent simulations or RL training. 
   In this experiment, the attacker is implemented with the policy `attack_maximal`.
   The `attack_maximal` policy entails that the attacker will always attack the attribute with
   the maximum value out of all of its neighbors. Similarly, the defender is implemented with the
   policy `defend_minimal`. The `defend_minimal` policy entails that the defender will always
   defend the attribute with the minimal value out of all of its neighbors.   
   
- [attack_maximal_vs_random-v0](simulations/v0/attack_maximal_vs_random/)
   * This is an experiment in the `idsgame-v0` environment.  
   An environment where neither the attacker nor defender is part of the environment, i.e.
   it is intended for 2-agent simulations or RL training.
   In this experiment, the attacker is implemented with the policy `attack_maximal`.
   The `attack_maximal` policy entails that the attacker will always attack the attribute with
   the maximum value out of all of its neighbors. The defender is implemented with a
   random defense policy.
      
- [tabular_q_agent_vs_random-v0](simulations/v0/tabular_q_agent_vs_random/)
   * This is an experiment in the `idsgame-v0` environment. 
   An environment where neither the attacker nor defender is part of the environment, i.e.
   it is intended for 2-agent simulations or RL training.
   In this experiment, the attacker is implemented with a greedy policy 
   based on a save Q-table.The defender is implemented with a random defense policy.
   **NOTE** By default the pre-configured environments will create vulnerabilities at random
   locations in the network in order to make sure that a pre-programmed policy cannot be the
   optimal. Therefore, when you run simulation with a pre-trained Q-agent and with a pre-configured
   environment, it is likely that the environment will be different (have vulnerabilities at different locations)
   than the environment for which the Q-agent was trained to act optimally in. Thus,
   if you want to simulate a trained Q-agent with the exact same environment as when it was
   trained you'll need to manually use the configuration APIs to make the environment 
   be identical. 
   
- [random_vs_tabular_q_agent-v0](simulations/v0/random_vs_tabular_q_agent/)
   * This is an experiment in the `idsgame-v0` environment. 
   An environment where neither the attacker nor defender is part of the environment, i.e.
   it is intended for 2-agent simulations or RL training.
   
   In this experiment, the defender is implemented with a greedy policy 
   based on a save Q-table. The attacker is implemented with a random attack policy.
   
   **NOTE** By default the pre-configured environments will create vulnerabilities at random
   locations in the network in order to make sure that a pre-programmed policy cannot be the
   optimal. Therefore, when you run simulation with a pre-trained Q-agent and with a pre-configured
   environment, it is likely that the environment will be different (have vulnerabilities at different locations)
   than the environment for which the Q-agent was trained to act optimally in. Thus,
   if you want to simulate a trained Q-agent with the exact same environment as when it was
   trained you'll need to manually use the configuration APIs to make the environment 
   be identical.
   
## Experiments in Version 1 Environments 

### Training Experiments
- [tabular_q_learning_vs_random_defense-v1](training/v1/random_defense/tabular_q_learning/)
   * This is an experiment in the `random_defense-v1` environment. 
   An environment where the defender is following a random defense policy. 
   This experiment trains an attacker agent using tabular q-learning to act optimally in the given
   environment and defeat the random defender.
   
- [tabular_q_learning_vs_minimal_defense-v1](training/v1/minimal_defense/tabular_q_learning/)
   * This is an experiment in the `minimal_defense-v1` environment.  
   An environment where the defender is following the `defend_minimal` defense policy. 
   The `defend_minimal` policy entails that the defender will always 
   defend the attribute with the minimal value out of all of its neighbors.
   This experiment trains an attacker agent using tabular q-learning to act optimally in the given 
   environment and defeat the defender.              
   
- [random_attack_vs_tabular_q_learning-v1](training/v1/random_attack/tabular_q_learning/)
   * This is an experiment in the `random_attack-v1` environment.  
   An environment where the attack is following a random attack policy.  
   This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and defeat the random attacker.
   
- [maximal_attack_vs_tabular_q_learning-v1](training/v1/maximal_attack/tabular_q_learning/)
   * This is an experiment in the `maximal_attack-v1` environment.
   An environment where the attack is following the `attack_maximal` attack policy.
   The `attack_maximal` policy entails that the attacker will always attack the attribute with
   the maximum value out of all of its neighbors. The defender is implemented with a
   random defense policy.
   This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and detect the attacker.   

### Simulation Experiments   
- [random_vs_random-v1](simulations/v1/random_vs_random/)
   * This is an experiment in the `idsgame-v1` environment.  
   An environment where neither the attacker nor defender is part of the environment, i.e.
   it is intended for 2-agent simulations or RL training.
   In this experiment, the attacker is implemented with a random attack policy.
   Similarly, the defender is implemented with a random defense policy. 
   
- [random_vs_defend_minimal-v1](simulations/v1/random_vs_defend_minimal/)
   * This is an experiment in the `idsgame-v1` environment. 
   An environment where neither the attacker nor defender is part of the environment, i.e.
   it is intended for 2-agent simulations or RL training. 
   In this experiment, the attacker is implemented with a random attack policy.
   The defender is implemented with the  policy `defend_minimal`. 
   The `defend_minimal` policy entails that the defender will always
   defend the attribute with the minimal value out of all of its neighbors.   
   
- [attack_maximal_vs_defend_minimal-v1](simulations/v1/attack_maximal_vs_defend_minimal/)
   * This is an experiment in the `idsgame-v1` environment.  
   An environment where neither the attacker nor defender is part of the environment, i.e. 
   it is intended for 2-agent simulations or RL training. 
   In this experiment, the attacker is implemented with the policy `attack_maximal`.
   The `attack_maximal` policy entails that the attacker will always attack the attribute with
   the maximum value out of all of its neighbors. Similarly, the defender is implemented with the
   policy `defend_minimal`. The `defend_minimal` policy entails that the defender will always
   defend the attribute with the minimal value out of all of its neighbors.   
   
- [attack_maximal_vs_random-v1](simulations/v1/attack_maximal_vs_random/)
   * This is an experiment in the `idsgame-v1` environment.  
   An environment where neither the attacker nor defender is part of the environment, i.e.
   it is intended for 2-agent simulations or RL training.
   In this experiment, the attacker is implemented with the policy `attack_maximal`.
   The `attack_maximal` policy entails that the attacker will always attack the attribute with
   the maximum value out of all of its neighbors. The defender is implemented with a
   random defense policy.

## Experiments in Version 2 Environments 

### Training Experiments
- [tabular_q_learning_vs_random_defense-v2](training/v2/random_defense/tabular_q_learning/)
   * This is an experiment in the `random_defense-v2` environment. 
   An environment where the defender is following a random defense policy. 
   This experiment trains an attacker agent using tabular q-learning to act optimally in the given
   environment and defeat the random defender.