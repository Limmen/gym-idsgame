# Experiments

Below are code to experiment with different scenarios of the environment. 
This code also makes it possible to reproduce any results that may have been reported  

## Experiments in Version 0 Environments 

Experiment in version 0 environments. That is, evironments with the following network topology:

```
                   Start
		     |
		     |
		     v
		   Server
		     |
		     |
		     v
		   Data
```
Nodes are initialized with the following state (index of the defense values to set to zero is selected randomly):
```
attack values: [0,0,0,0,0,0,0,0,0,0]
defense values: [2,2,0,2,2,2,2,2,2,2]
det: 2
```

The environment has sparse rewards (+1,-1 rewards are given at the terminal state of each episode)

### Training Experiments

Experiments where one or two of the agents are using some learning algorithm to update their policy.

#### Random Defense
Experiments in the `random_defense-v0` environment. 
An environment where the defender is following a random defense policy.

- [tabular_q_learning_vs_random_defense-v0](training/v0/random_defense/tabular_q_learning/)
   *  This experiment trains an attacker agent using tabular q-learning to act optimally in the given
   environment and defeat the random defender.
   
- [dqn_vs_random_defense-v0](training/v0/random_defense/dqn/)
   *  This experiment trains an attacker agent using DQN to act optimally in the given
   environment and defeat the random defender.   

#### Minimal Defense
   
Experiments in the `minimal_defense-v0` environment.  
An environment where the defender is following the `defend_minimal` defense policy.
The `defend_minimal` policy entails that the defender will always 
defend the attribute with the minimal value out of all of its neighbors.   
   
- [tabular_q_learning_vs_minimal_defense-v0](training/v0/minimal_defense/tabular_q_learning/)
   * This experiment trains an attacker agent using tabular q-learning to act optimally in the given 
   environment and defeat the defender.
   
- [dqn_vs_minimal_defense-v0](training/v0/minimal_defense/dqn/)
   * This experiment trains an attacker agent using DQN to act optimally in the given
   environment and defeat the random defender.             

#### Random Attack   

This is an experiment in the `random_attack-v0` environment.  
An environment where the attack is following a random attack policy.

- [random_attack_vs_tabular_q_learning-v0](training/v0/random_attack/tabular_q_learning/)
   * This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and defeat the random attacker.

#### Maximal Attack

Experiments in the `maximal_attack-v0` environment. 
An environment where the attack is following the `attack_maximal` attack policy.
The `attack_maximal` policy entails that the attacker will always attack the attribute with
the maximum value out of all of its neighbors. 
      
- [maximal_attack_vs_tabular_q_learning-v0](training/v0/maximal_attack/tabular_q_learning/)
   * This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and detect the attacker.

#### Two Agents
Experiments in the `idsgame-v0` environment. 
An environment where neither the attacker nor defender is part of the environment, i.e.
it is intended for 2-agent simulations or RL training.
   
- [tabular_q_learning_vs_tabular_q_learning-v0](training/v0/two_agents/tabular_q_learning/)
   * This experiment trains both an attacker and a defender agent simultaneously against each other 
   using tabular q-learning.     

### Simulation Experiments   

Experiments with pre-defined policies (no training)

#### Two Agents

Experiments in the `idsgame-v0` environment.  
An environment where neither the attacker nor defender is part of the environment, i.e.
it is intended for 2-agent simulations or RL training.

- [random_vs_random-v0](simulations/v0/random_vs_random/)
   * In this experiment, the attacker is implemented with a random attack policy.
   Similarly, the defender is implemented with a random defense policy. 
   
- [random_vs_defend_minimal-v0](simulations/v0/random_vs_defend_minimal/)
   * In this experiment, the attacker is implemented with a random attack policy.
   The defender is implemented with the  policy `defend_minimal`. 
   The `defend_minimal` policy entails that the defender will always
   defend the attribute with the minimal value out of all of its neighbors.   
   
- [attack_maximal_vs_defend_minimal-v0](simulations/v0/attack_maximal_vs_defend_minimal/)
   *  In this experiment, the attacker is implemented with the policy `attack_maximal`.
   The `attack_maximal` policy entails that the attacker will always attack the attribute with
   the maximum value out of all of its neighbors. Similarly, the defender is implemented with the
   policy `defend_minimal`. The `defend_minimal` policy entails that the defender will always
   defend the attribute with the minimal value out of all of its neighbors.   
   
- [attack_maximal_vs_random-v0](simulations/v0/attack_maximal_vs_random/)
   * In this experiment, the attacker is implemented with the policy `attack_maximal`.
   The `attack_maximal` policy entails that the attacker will always attack the attribute with
   the maximum value out of all of its neighbors. The defender is implemented with a
   random defense policy.
      
- [tabular_q_agent_vs_random-v0](simulations/v0/tabular_q_agent_vs_random/)
   * In this experiment, the attacker is implemented with a greedy policy 
   based on a save Q-table.The defender is implemented with a random defense policy.    
   
- [random_vs_tabular_q_agent-v0](simulations/v0/random_vs_tabular_q_agent/)
   * In this experiment, the defender is implemented with a greedy policy 
   based on a save Q-table. The attacker is implemented with a random attack policy.   
   
## Experiments in Version 1 Environments 

Experiment in version 1 environments. That is, evironments with the following network topology:

```
                   Start
		     |
		     |
		     v
		   Server
		     |
		     |
		     v
		   Data
```
Nodes are initialized with the following state (index of the defense values to set to zero is selected randomly):

```
attack values: [0,0,0,0,0,0,0,0,0,0]
defense values: [4,0,0,4,4,0,4,4,0,4]
det: 3
```
The environment has sparse rewards (+1,-1 rewards are given at the terminal state of each episode)

### Training Experiments

Experiments where one or two of the agents are using some learning algorithm to update their policy.

#### Random Defense

Experiments in the `random_defense-v1` environment. 
An environment where the defender is following a random defense policy.

- [tabular_q_learning_vs_random_defense-v1](training/v1/random_defense/tabular_q_learning/)
   * This experiment trains an attacker agent using tabular q-learning to act optimally in the given
   environment and defeat the random defender.

- [dqn_vs_random_defense-v1](training/v1/random_defense/dqn/)
   * This experiment trains an attacker agent using DQN to act optimally in the given
   environment and defeat the random defender. 

#### Minimal Defense
   
This is an experiment in the `minimal_defense-v1` environment.  
An environment where the defender is following the `defend_minimal` defense policy. 
The `defend_minimal` policy entails that the defender will always 
defend the attribute with the minimal value out of all of its neighbors.   
   
- [tabular_q_learning_vs_minimal_defense-v1](training/v1/minimal_defense/tabular_q_learning/)
   * This experiment trains an attacker agent using tabular q-learning to act optimally in the given 
   environment and defeat the defender.
   
- [dqn_vs_minimal_defense-v1](training/v1/minimal_defense/dqn/)
   * This experiment trains an attacker agent using DQN to act optimally in the given
   environment and defeat the random defender.                 

#### Random Attack   
This is an experiment in the `random_attack-v1` environment.  
An environment where the attack is following a random attack policy.   
   
- [random_attack_vs_tabular_q_learning-v1](training/v1/random_attack/tabular_q_learning/)
   * This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and defeat the random attacker.

#### Maximal Attack
This is an experiment in the `maximal_attack-v1` environment.
An environment where the attack is following the `attack_maximal` attack policy.
The `attack_maximal` policy entails that the attacker will always attack the attribute with
the maximum value out of all of its neighbors.
      
- [maximal_attack_vs_tabular_q_learning-v1](training/v1/maximal_attack/tabular_q_learning/)
   * This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and detect the attacker.  

#### Two Agents

Experiments in the `idsgame-v1` environment. 
An environment where neither the attacker nor defender is part of the environment, i.e.
it is intended for 2-agent simulations or RL training.
   
- [tabular_q_learning_vs_tabular_q_learning-v1](training/v1/two_agents/tabular_q_learning/)
   * This experiment trains both an attacker and a defender agent simultaneously against each other 
   using tabular q-learning.   

### Simulation Experiments

Experiments with pre-defined policies (no training)

#### Two Agents

Experiments in the `idsgame-v1` environment.  
An environment where neither the attacker nor defender is part of the environment, i.e.
it is intended for 2-agent simulations or RL training.
      
- [random_vs_random-v1](simulations/v1/random_vs_random/)
   * In this experiment, the attacker is implemented with a random attack policy.
   Similarly, the defender is implemented with a random defense policy. 
   
- [random_vs_defend_minimal-v1](simulations/v1/random_vs_defend_minimal/)
   *  In this experiment, the attacker is implemented with a random attack policy.
   The defender is implemented with the  policy `defend_minimal`. 
   The `defend_minimal` policy entails that the defender will always
   defend the attribute with the minimal value out of all of its neighbors.   
   
- [attack_maximal_vs_defend_minimal-v1](simulations/v1/attack_maximal_vs_defend_minimal/)
   * In this experiment, the attacker is implemented with the policy `attack_maximal`.
   The `attack_maximal` policy entails that the attacker will always attack the attribute with
   the maximum value out of all of its neighbors. Similarly, the defender is implemented with the
   policy `defend_minimal`. The `defend_minimal` policy entails that the defender will always
   defend the attribute with the minimal value out of all of its neighbors.   
   
- [attack_maximal_vs_random-v1](simulations/v1/attack_maximal_vs_random/)
   * In this experiment, the attacker is implemented with the policy `attack_maximal`.
   The `attack_maximal` policy entails that the attacker will always attack the attribute with
   the maximum value out of all of its neighbors. The defender is implemented with a
   random defense policy.

## Experiments in Version 2 Environments 

Experiment in version 2 environments. That is, evironments with the following network topology:

```
				 Start
				   |
			  +--------+--------+
			  |		    |
			Server            Server
			  |		    |
			  +--------+--------+
				   |
				  Data
```

This is the standard network from Elderman et al. Nodes are initialized with the following state (index of the defense values to set to zero is selected randomly):

```
attack values: [0,0,0,0,0,0,0,0,0,0]
defense values: [2,2,0,2,2,2,2,2,2,2]
det: 2
```
The environment has sparse rewards (+1,-1 rewards are given at the terminal state of each episode)

### Training Experiments

Experiments where one or two of the agents are using some learning algorithm to update their policy.

#### Random Defense
This is an experiment in the `random_defense-v2` environment. 
An environment where the defender is following a random defense policy.

- [tabular_q_learning_vs_random_defense-v2](training/v2/random_defense/tabular_q_learning/)
   * This experiment trains an attacker agent using tabular q-learning to act optimally in the given
   environment and defeat the random defender.
   
- [dqn_vs_random_defense-v2](training/v2/random_defense/dqn/)
   * This experiment trains an attacker agent using DQN to act optimally in the given
   environment and defeat the random defender.   

#### Minimal Defense

This is an experiment in the `minimal_defense-v2` environment.  
An environment where the defender is following the `defend_minimal` defense policy. 
The `defend_minimal` policy entails that the defender will always 
defend the attribute with the minimal value out of all of its neighbors.
   
- [tabular_q_learning_vs_minimal_defense-v2](training/v2/minimal_defense/tabular_q_learning/)
   * This experiment trains an attacker agent using tabular q-learning to act optimally in the given 
   environment and defeat the defender.   

- [dqn_vs_minimal_defense-v2](training/v2/minimal_defense/dqn/)
   * This experiment trains an attacker agent using DQN to act optimally in the given
   environment and defeat the random defender.

#### Random Attack   

This is an experiment in the `random_attack-v2` environment.  
An environment where the attack is following a random attack policy.
   
- [random_attack_vs_tabular_q_learning-v2](training/v2/random_attack/tabular_q_learning/)
   * This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and defeat the random attacker.   

#### Maximal Attack

This is an experiment in the `maximal_attack-v2` environment.
An environment where the attack is following the `attack_maximal` attack policy.
The `attack_maximal` policy entails that the attacker will always attack the attribute with
the maximum value out of all of its neighbors.

- [maximal_attack_vs_tabular_q_learning-v2](training/v2/maximal_attack/tabular_q_learning/)
   * This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and detect the attacker.    

#### Two Agents
This is an experiment in the `idsgame-v2` environment. 
An environment where neither the attacker nor defender is part of the environment, i.e.
it is intended for 2-agent simulations or RL training.
      
- [tabular_q_learning_vs_tabular_q_learning-v2](training/v2/two_agents/tabular_q_learning/)
   * This experiment trains both an attacker and a defender agent simultaneously against each other 
   using tabular q-learning.   
   
### Simulation Experiments

Experiments with pre-defined policies (no training)

#### Two Agents

Experiments in the `idsgame-v2` environment.  
An environment where neither the attacker nor defender is part of the environment, i.e.
it is intended for 2-agent simulations or RL training.
      
- [random_vs_random-v2](simulations/v2/random_vs_random/)
   * In this experiment, the attacker is implemented with a random attack policy.
   Similarly, the defender is implemented with a random defense policy. 
   
- [random_vs_defend_minimal-v2](simulations/v2/random_vs_defend_minimal/)
   * In this experiment, the attacker is implemented with a random attack policy.
   The defender is implemented with the  policy `defend_minimal`. 
   The `defend_minimal` policy entails that the defender will always
   defend the attribute with the minimal value out of all of its neighbors.   
   
- [attack_maximal_vs_defend_minimal-v2](simulations/v2/attack_maximal_vs_defend_minimal/)
   * In this experiment, the attacker is implemented with the policy `attack_maximal`.
   The `attack_maximal` policy entails that the attacker will always attack the attribute with
   the maximum value out of all of its neighbors. Similarly, the defender is implemented with the
   policy `defend_minimal`. The `defend_minimal` policy entails that the defender will always
   defend the attribute with the minimal value out of all of its neighbors.   
   
- [attack_maximal_vs_random-v2](simulations/v2/attack_maximal_vs_random/)
   * In this experiment, the attacker is implemented with the policy `attack_maximal`.
   The `attack_maximal` policy entails that the attacker will always attack the attribute with
   the maximum value out of all of its neighbors. The defender is implemented with a
   random defense policy.
   
## Experiments in Version 3 Environments

Experiment in version 3 environments. That is, evironments with the following network topology:

```
				 Start
				  |
				  |
		       +-------------------+
		       |	  |	   |
		       v	  v	   v
		     Server     Server   Server
		       |	  |	   |
		       |	  |	   |
		       v	  v	   v
		     Server  	Server   Server
		       |	  |	   |
		       |	  |	   |
		       +----------+--------+
				  |
				  v
				 Data
```

Nodes are initialized with the following state (index of the defense values to set to zero is selected randomly):

```
attack values: [0,0,0,0,0,0,0,0,0,0]
defense values: [2,2,0,2,2,2,2,2,2,2]
det: 2
```

The environment has sparse rewards (+1,-1 rewards are given at the terminal state of each episode)

### Training Experiments

Experiments where one or two of the agents are using some learning algorithm to update their policy.

#### Random Defense
This is an experiment in the `random_defense-v3` environment. 
An environment where the defender is following a random defense policy.

- [tabular_q_learning_vs_random_defense-v3](training/v3/random_defense/tabular_q_learning/)
   * This experiment trains an attacker agent using tabular q-learning to act optimally in the given
   environment and defeat the random defender.    

- [dqn_vs_random_defense-v3](training/v3/random_defense/dqn/)
   * This experiment trains an attacker agent using DQN to act optimally in the given
   environment and defeat the random defender.

#### Minimal Defense
This is an experiment in the `minimal_defense-v3` environment.  
An environment where the defender is following the `defend_minimal` defense policy. 
The `defend_minimal` policy entails that the defender will always 
defend the attribute with the minimal value out of all of its neighbors.
   
- [tabular_q_learning_vs_minimal_defense-v3](training/v3/minimal_defense/tabular_q_learning/)
   * This experiment trains an attacker agent using tabular q-learning to act optimally in the given 
   environment and defeat the defender.   
   
- [dqn_vs_minimal_defense-v3](training/v3/minimal_defense/dqn/)
   * This experiment trains an attacker agent using DQN to act optimally in the given
   environment and defeat the random defender.  

#### Random Attack

Experiments in the `random_attack-v3` environment.  
An environment where the attack is following a random attack policy.
         
- [random_attack_vs_tabular_q_learning-v3](training/v3/random_attack/tabular_q_learning/)
   * This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and defeat the random attacker.   

#### Maximal Attack

Experiments in the `maximal_attack-v3` environment.
An environment where the attack is following the `attack_maximal` attack policy.
The `attack_maximal` policy entails that the attacker will always attack the attribute with
the maximum value out of all of its neighbors. 
      
- [maximal_attack_vs_tabular_q_learning-v3](training/v3/maximal_attack/tabular_q_learning/)
   * This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and detect the attacker. 

#### Two Agents

Experiments in the `idsgame-v3` environment. 
An environment where neither the attacker nor defender is part of the environment, i.e.
it is intended for 2-agent simulations or RL training.
   
- [tabular_q_learning_vs_tabular_q_learning-v3](training/v3/two_agents/tabular_q_learning/)
   * This experiment trains both an attacker and a defender agent simultaneously against each other 
   using tabular q-learning.       
   
### Simulation Experiments

Experiments with pre-defined policies (no training)

#### Two Agents
Experiments in the `idsgame-v3` environment.  
An environment where neither the attacker nor defender is part of the environment, i.e.
it is intended for 2-agent simulations or RL training.

- [random_vs_random-v3](simulations/v3/random_vs_random/)
   * In this experiment, the attacker is implemented with a random attack policy.
   Similarly, the defender is implemented with a random defense policy. 
   
- [random_vs_defend_minimal-v3](simulations/v3/random_vs_defend_minimal/)
   * In this experiment, the attacker is implemented with a random attack policy.
   The defender is implemented with the  policy `defend_minimal`. 
   The `defend_minimal` policy entails that the defender will always
   defend the attribute with the minimal value out of all of its neighbors.   
   
- [attack_maximal_vs_defend_minimal-v3](simulations/v3/attack_maximal_vs_defend_minimal/)
   * In this experiment, the attacker is implemented with the policy `attack_maximal`.
   The `attack_maximal` policy entails that the attacker will always attack the attribute with
   the maximum value out of all of its neighbors. Similarly, the defender is implemented with the
   policy `defend_minimal`. The `defend_minimal` policy entails that the defender will always
   defend the attribute with the minimal value out of all of its neighbors.   
   
- [attack_maximal_vs_random-v3](simulations/v3/attack_maximal_vs_random/)
   * In this experiment, the attacker is implemented with the policy `attack_maximal`.
   The `attack_maximal` policy entails that the attacker will always attack the attribute with
   the maximum value out of all of its neighbors. The defender is implemented with a
   random defense policy. 
   
- [tabular_q_agent_vs_tabular_q_agent-v3](simulations/v3/tabular_q_agent_vs_tabular_q_agent/)
   *  In this experiment, both the attacker and defender are implemented with a greedy policy 
   based on saved Q-table from previous Q-training. 
   
   
## Experiments in Version 4 Environments

Experiment in version 4 environments. That is, evironments with the following network topology:

```

												 Start
												   |
							                          		   |
				       +-----------------------------+-----------------------------+-------------------------+-------------------------+
				       | 			     |			           |			     |			       |
				       | 			     |				   |			     |			       |
				       v			     v				   v			     v			       v
				     Server			   Server     			 Server      		   Server    		     Server
				       |			     |				   |			     |			       |
				       |			     |				   |			     |			       |
				       |			     |				   |			     |			       |
				       v			     v				   v			     v			       v
				     Server			   Server			 Server		           Server		     Server
				       |			     |				   |			     |			       |
				       |			     |				   |			     |			       |
				       |			     |				   |			     |			       |
				       v			     v				   v			     v			       v
				     Server			   Server			 Server		           Server		     Server
				       |			     |				   |			     |			       |
				       |			     |				   |			     |			       |
				       |			     |				   |			     |			       |
				       v			     v				   v			     v			       v
				     Server			   Server			 Server 		   Server		     Server
				       |			     |				   |			     |			       |
				       |			     |				   |			     |			       |
				       +-----------------------------+-----------------------------+-------------------------+-------------------------+
				       								   |
												   |
												   v
												  Data
```

Nodes are initialized with the following state (index of the defense values to set to zero is selected randomly):

```
attack values: [0,0,0,0,0,0,0,0,0,0]
defense values: [2,2,0,2,2,2,2,2,2,2]
det: 2
```

The environment has sparse rewards (+1,-1 rewards are given at the terminal state of each episode)

### Training Experiments

Experiments where one or two of the agents are using some learning algorithm to update their policy.

#### Random Defense

Experiments in the `random_defense-v4` environment. 
An environment where the defender is following a random defense policy.

- [tabular_q_learning_vs_random_defense-v4](training/v4/random_defense/tabular_q_learning/)
   * This experiment trains an attacker agent using tabular q-learning to act optimally in the given
   environment and defeat the random defender.

- [dqn_vs_random_defense-v4](training/v4/random_defense/dqn/)
   * This experiment trains an attacker agent using DQN to act optimally in the given
   environment and defeat the random defender.
   
#### Minimal Defense
Experiments in the `minimal_defense-v4` environment.  
An environment where the defender is following the `defend_minimal` defense policy. 
The `defend_minimal` policy entails that the defender will always 
defend the attribute with the minimal value out of all of its neighbors.

- [tabular_q_learning_vs_minimal_defense-v4](training/v4/minimal_defense/tabular_q_learning/)
   * This experiment trains an attacker agent using tabular q-learning to act optimally in the given 
   environment and defeat the defender.   

- [dqn_vs_minimal_defense-v4](training/v4/minimal_defense/dqn/)
   * This experiment trains an attacker agent using DQN to act optimally in the given
   environment and defeat the random defender.  
   
#### Random Attack
Experiments in the `random_attack-v4` environment.  
An environment where the attack is following a random attack policy.
   
- [random_attack_vs_tabular_q_learning-v4](training/v4/random_attack/tabular_q_learning/)
   * This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and defeat the random attacker.   

#### Maximal Attack
Experiments in the `maximal_attack-v4` environment.
An environment where the attack is following the `attack_maximal` attack policy.
      
- [maximal_attack_vs_tabular_q_learning-v4](training/v4/maximal_attack/tabular_q_learning/)
   * The `attack_maximal` policy entails that the attacker will always attack the attribute with
   the maximum value out of all of its neighbors. The defender is implemented with a
   random defense policy. This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and detect the attacker.

#### Two Agents
Experiments in the `idsgame-v4` environment. 
An environment where neither the attacker nor defender is part of the environment, i.e.
it is intended for 2-agent simulations or RL training.
      
- [tabular_q_learning_vs_tabular_q_learning-v4](training/v4/two_agents/tabular_q_learning/)
   * This experiment trains both an attacker and a defender agent simultaneously against each other 
   using tabular q-learning.                
   
### Simulation Experiments

Experiments with pre-defined policies (no training)

#### Two Agents
Experiments in the `idsgame-v4` environment.  
An environment where neither the attacker nor defender is part of the environment, i.e.
it is intended for 2-agent simulations or RL training.
      
- [random_vs_random-v4](simulations/v4/random_vs_random/)
   * In this experiment, the attacker is implemented with a random attack policy.
   Similarly, the defender is implemented with a random defense policy. 
   
- [random_vs_defend_minimal-v4](simulations/v4/random_vs_defend_minimal/)
   * In this experiment, the attacker is implemented with a random attack policy.
   The defender is implemented with the  policy `defend_minimal`. 
   The `defend_minimal` policy entails that the defender will always
   defend the attribute with the minimal value out of all of its neighbors.   
   
- [attack_maximal_vs_defend_minimal-v4](simulations/v4/attack_maximal_vs_defend_minimal/)
   * In this experiment, the attacker is implemented with the policy `attack_maximal`.
   The `attack_maximal` policy entails that the attacker will always attack the attribute with
   the maximum value out of all of its neighbors. Similarly, the defender is implemented with the
   policy `defend_minimal`. The `defend_minimal` policy entails that the defender will always
   defend the attribute with the minimal value out of all of its neighbors.   
   
- [attack_maximal_vs_random-v4](simulations/v4/attack_maximal_vs_random/)
   * In this experiment, the attacker is implemented with the policy `attack_maximal`.
   The `attack_maximal` policy entails that the attacker will always attack the attribute with
   the maximum value out of all of its neighbors. The defender is implemented with a
   random defense policy.
   
- [tabular_q_agent_vs_defend_minimal-v4](simulations/v4/tabular_q_agent_vs_defend_minimal/)
   * In this experiment, the attacker is implemented with a greedy policy 
   based on a save Q-table. The defender is implemented with the
   policy `defend_minimal`. The `defend_minimal` policy entails that the defender will always
   defend the attribute with the minimal value out of all of its neighbors.       
   
## Experiments in Version 5 Environments
Experiment in version 5 environments. That is, evironments with the following network topology:

```

												 Start
												   |
							                          		   |
				       +-----------------------------+-----------------------------+-------------------------+-------------------------+
				       | 			     |			           |			     |			       |
				       | 			     |				   |			     |			       |
				       v			     v				   v			     v			       v
				     Server------------------------Server------------------------Server--------------------Server--------------------Server
				       |                	     |		             	   |			     |			       |
				       |			     |				   |			     |			       |
				       |			     |				   |			     |			       |
				       v			     v				   v			     v			       v
				     Server------------------------Server------------------------Server--------------------Server--------------------Server
				       |	         	     |		                   |	        	     |			       |
				       |			     |				   |			     |			       |
				       |			     |				   |			     |			       |
				       v			     v				   v			     v			       v
				     Server------------------------Server------------------------Server--------------------Server--------------------Server
				       |	         	     |				   |	                     |			       |
				       |			     |				   |			     |			       |
				       |			     |				   |			     |			       |
				       v			     v				   v			     v			       v
				     Server------------------------Server------------------------Server--------------------Server--------------------Server
				       |			     |				   |			     |			       |
				       |			     |				   |			     |			       |
				       +-----------------------------+-----------------------------+-------------------------+-------------------------+
				       								   |
												   |
												   v
												  Data

```

Nodes are initialized with the following state (index of the defense values to set to zero is selected randomly):

```
attack values: [0,0,0,0,0,0,0,0,0,0]
defense values: [2,2,0,2,2,2,2,2,2,2]
det: 2
```
Moreover, only two nodes per layer has a vulnerability (defense value set to 0, all other nodes in the layer have defense value initialized to 2 on all attributes)

The environment has sparse rewards (+1,-1 rewards are given at the terminal state of each episode)

### Training Experiments

Experiments where one or two of the agents are using some learning algorithm to update their policy.

#### Random Defense
Experiments in the `random_defense-v5` environment. 
An environment where the defender is following a random defense policy. 
   
- [tabular_q_learning_vs_random_defense-v5](training/v5/random_defense/tabular_q_learning/)
   * This experiment trains an attacker agent using tabular q-learning to act optimally in the given
   environment and defeat the random defender.

- [dqn_vs_random_defense-v5](training/v5/random_defense/dqn/)
   * This experiment trains an attacker agent using DQN to act optimally in the given
   environment and defeat the random defender.
   
#### Minimal Defense
Experiments in the `minimal_defense-v5` environment.  
An environment where the defender is following the `defend_minimal` defense policy.
The `defend_minimal` policy entails that the defender will always 
defend the attribute with the minimal value out of all of its neighbors.
      
- [tabular_q_learning_vs_minimal_defense-v5](training/v5/minimal_defense/tabular_q_learning/)
   * This experiment trains an attacker agent using tabular q-learning to act optimally in the given 
   environment and defeat the defender.
   
- [dqn_vs_minimal_defense-v5](training/v5/minimal_defense/dqn/)
   * This experiment trains an attacker agent using dqn to act optimally in the given 
   environment and defeat the defender.         

#### Random Attack
Experiments in the `random_attack-v5` environment.  
An environment where the attack is following a random attack policy.
   
- [random_attack_vs_tabular_q_learning-v5](training/v5/random_attack/tabular_q_learning/)
   * This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and defeat the random attacker.   

#### Maximal Attack
This is an experiment in the `maximal_attack-v5` environment.
An environment where the attack is following the `attack_maximal` attack policy.
      
- [maximal_attack_vs_tabular_q_learning-v5](training/v5/maximal_attack/tabular_q_learning/)
   * The `attack_maximal` policy entails that the attacker will always attack the attribute with
   the maximum value out of all of its neighbors. The defender is implemented with a
   random defense policy. This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and detect the attacker. 

#### Two Agents
This is an experiment in the `idsgame-v5` environment. 
An environment where neither the attacker nor defender is part of the environment, i.e.
it is intended for 2-agent simulations or RL training.
   
- [tabular_q_learning_vs_tabular_q_learning-v5](training/v5/two_agents/tabular_q_learning/)
   * This experiment trains both an attacker and a defender agent simultaneously against each other 
   using tabular q-learning.    
         
### Simulation Experiments

Experiments with pre-defined policies (no training)

#### Two Agents
This is an experiment in the `idsgame-v5` environment.  
An environment where neither the attacker nor defender is part of the environment, i.e.
it is intended for 2-agent simulations or RL training.
   
- [random_vs_random-v5](simulations/v5/random_vs_random/)
   * In this experiment, the attacker is implemented with a random attack policy.
   Similarly, the defender is implemented with a random defense policy. 
   
- [random_vs_defend_minimal-v5](simulations/v5/random_vs_defend_minimal/)
   *  In this experiment, the attacker is implemented with a random attack policy.
   The defender is implemented with the  policy `defend_minimal`. 
   The `defend_minimal` policy entails that the defender will always
   defend the attribute with the minimal value out of all of its neighbors.   
   
- [attack_maximal_vs_defend_minimal-v5](simulations/v5/attack_maximal_vs_defend_minimal/)
   * In this experiment, the attacker is implemented with the policy `attack_maximal`.
   The `attack_maximal` policy entails that the attacker will always attack the attribute with
   the maximum value out of all of its neighbors. Similarly, the defender is implemented with the
   policy `defend_minimal`. The `defend_minimal` policy entails that the defender will always
   defend the attribute with the minimal value out of all of its neighbors.   
   
- [attack_maximal_vs_random-v5](simulations/v5/attack_maximal_vs_random/)
   * In this experiment, the attacker is implemented with the policy `attack_maximal`.
   The `attack_maximal` policy entails that the attacker will always attack the attribute with
   the maximum value out of all of its neighbors. The defender is implemented with a
   random defense policy.
   
## Experiments in Version 7 Environments

Experiment in version 7 environments. That is, evironments with the following network topology:

```
				 Start
				  |
				  |
		       +-------------------+
		       |	  |	   |
		       v	  v	   v
		     Server     Server   Server
		       |	  |	   |
		       |	  |	   |
		       v	  v	   v
		     Server  	Server   Server
		       |	  |	   |
		       |	  |	   |
		       +----------+--------+
				  |
				  v
				 Data
```

Nodes are initialized with the following state (index of the defense values to set to zero is selected randomly):

```
attack values: [0,0,0,0,0,0,0,0,0,0]
defense values: [2,2,0,2,2,2,2,2,2,2]
det: 2
```

The environment has dense rewards (+1,-1 given whenever the attacker reaches a new level in the network)

### Training Experiments

Experiments where one or two of the agents are using some learning algorithm to update their policy.

#### Random Defense
This is an experiment in the `random_defense-v7` environment. 
An environment where the defender is following a random defense policy.

- [tabular_q_learning_vs_random_defense-v7](training/v7/random_defense/tabular_q_learning/)
   * This experiment trains an attacker agent using tabular q-learning to act optimally in the given
   environment and defeat the random defender.    

- [dqn_vs_random_defense-v7](training/v7/random_defense/dqn/)
   * This experiment trains an attacker agent using DQN to act optimally in the given
   environment and defeat the random defender.

#### Minimal Defense
This is an experiment in the `minimal_defense-v7` environment.  
An environment where the defender is following the `defend_minimal` defense policy. 
The `defend_minimal` policy entails that the defender will always 
defend the attribute with the minimal value out of all of its neighbors.
   
- [tabular_q_learning_vs_minimal_defense-v7](training/v7/minimal_defense/tabular_q_learning/)
   * This experiment trains an attacker agent using tabular q-learning to act optimally in the given 
   environment and defeat the defender.   
   
- [dqn_vs_minimal_defense-v7](training/v7/minimal_defense/dqn/)
   * This experiment trains an attacker agent using DQN to act optimally in the given
   environment and defeat the random defender.  

#### Random Attack

Experiments in the `random_attack-v7` environment.  
An environment where the attack is following a random attack policy.
         
- [random_attack_vs_tabular_q_learning-v7](training/v7/random_attack/tabular_q_learning/)
   * This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and defeat the random attacker. 
   
- [random_attack_vs_dqn-v7](training/v7/random_attack/dqn/)
   * This experiment trains a defender agent using DQN to act optimally in the given
   environment and defeat the random attacker.   

#### Maximal Attack

Experiments in the `maximal_attack-v7` environment.
An environment where the attack is following the `attack_maximal` attack policy.
The `attack_maximal` policy entails that the attacker will always attack the attribute with
the maximum value out of all of its neighbors. 
      
- [maximal_attack_vs_tabular_q_learning-v7](training/v7/maximal_attack/tabular_q_learning/)
   * This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and detect the attacker. 

- [maximal_attack_vs_dqn-v7](training/v7/maximal_attack/dqn/)
   * This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and detect the attacker. 

#### Two Agents

Experiments in the `idsgame-v7` environment. 
An environment where neither the attacker nor defender is part of the environment, i.e.
it is intended for 2-agent simulations or RL training.
   
- [tabular_q_learning_vs_tabular_q_learning-v7](training/v7/two_agents/tabular_q_learning/)
   * This experiment trains both an attacker and a defender agent simultaneously against each other 
   using tabular q-learning.
   
- [dqn_vs_dqn-v7](training/v7/two_agents/dqn/)
   * This experiment trains both an attacker and a defender agent simultaneously against each other 
   using tabular q-learning.       
      
## Experiments in Version 8 Environments 

Experiment in version 8 environments. That is, evironments with the following network topology:

```
                   Start
		     |
		     |
		     v
		   Server
		     |
		     |
		     v
		   Data
```
Nodes are initialized with the following state (index of the defense values to set to zero is selected randomly):
```
attack values: [0,0,0,0,0,0,0,0,0,0]
defense values: [2,2,0,2,2,2,2,2,2,2]
det: 2
```

The environment has dense rewards (+1,-1 given whenever the attacker reaches a new level in the network)

### Training Experiments

Experiments where one or two of the agents are using some learning algorithm to update their policy.

#### Random Defense
Experiments in the `random_defense-v8` environment. 
An environment where the defender is following a random defense policy.

- [tabular_q_learning_vs_random_defense-v8](training/v8/random_defense/tabular_q_learning/)
   *  This experiment trains an attacker agent using tabular q-learning to act optimally in the given
   environment and defeat the random defender.
   
- [dqn_vs_random_defense-v8](training/v8/random_defense/dqn/)
   *  This experiment trains an attacker agent using DQN to act optimally in the given
   environment and defeat the random defender.   

#### Minimal Defense
   
Experiments in the `minimal_defense-v8` environment.  
An environment where the defender is following the `defend_minimal` defense policy.
The `defend_minimal` policy entails that the defender will always 
defend the attribute with the minimal value out of all of its neighbors.   
   
- [tabular_q_learning_vs_minimal_defense-v8](training/v8/minimal_defense/tabular_q_learning/)
   * This experiment trains an attacker agent using tabular q-learning to act optimally in the given 
   environment and defeat the defender.
   
- [dqn_vs_minimal_defense-v8](training/v8/minimal_defense/dqn/)
   * This experiment trains an attacker agent using DQN to act optimally in the given
   environment and defeat the random defender.
   
- [reinforce_vs_minimal_defense-v8](training/v8/minimal_defense/reinforce/)
   * This experiment trains an attacker agent using REINFORCE to act optimally in the given
   environment and defeat the random defender.
   
- [actor_critic_vs_minimal_defense-v8](training/v8/minimal_defense/actor_critic/)
   * This experiment trains an attacker agent using Actor-Critic to act optimally in the given
   environment and defeat the random defender.                   

#### Random Attack   

This is an experiment in the `random_attack-v8` environment.  
An environment where the attack is following a random attack policy.

- [random_attack_vs_tabular_q_learning-v8](training/v8/random_attack/tabular_q_learning/)
   * This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and defeat the random attacker.
   
- [random_attack_vs_dqn-v8](training/v8/random_attack/dqn/)
   * This experiment trains a defender agent using DQN to act optimally in the given
   environment and defeat the random attacker.   

#### Maximal Attack

Experiments in the `maximal_attack-v8` environment. 
An environment where the attack is following the `attack_maximal` attack policy.
The `attack_maximal` policy entails that the attacker will always attack the attribute with
the maximum value out of all of its neighbors. 
      
- [maximal_attack_vs_tabular_q_learning-v8](training/v8/maximal_attack/tabular_q_learning/)
   * This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and detect the attacker.
   
- [maximal_attack_vs_dqn-v8](training/v8/maximal_attack/dqn/)
   * This experiment trains a defender agent using DQN to act optimally in the given
   environment and detect the attacker.   

#### Two Agents
Experiments in the `idsgame-v8` environment. 
An environment where neither the attacker nor defender is part of the environment, i.e.
it is intended for 2-agent simulations or RL training.
   
- [tabular_q_learning_vs_tabular_q_learning-v8](training/v8/two_agents/tabular_q_learning/)
   * This experiment trains both an attacker and a defender agent simultaneously against each other 
   using tabular q-learning.
   
- [dqn_vs_dqn-v8](training/v8/two_agents/dqn/)
   * This experiment trains both an attacker and a defender agent simultaneously against each other 
   using DQN.
   
- [reinforce_vs_reinforce-v8](training/v8/two_agents/dqn/)
   * This experiment trains both an attacker and a defender agent simultaneously against each other 
   using REINFORCE.     
   
## Experiments in Version 9 Environments 

Experiment in version 9 environments. That is, evironments with the following network topology:

```
				 Start
				   |
			  +--------+--------+
			  |		    |
			Server            Server
			  |		    |
			  +--------+--------+
				   |
				  Data
```

This is the standard network from Elderman et al. Nodes are initialized with the following state (index of the defense values to set to zero is selected randomly):

```
attack values: [0,0,0,0,0,0,0,0,0,0]
defense values: [2,2,0,2,2,2,2,2,2,2]
det: 2
```
The environment has dense rewards (+1,-1 given whenever the attacker reaches a new level in the network)

### Training Experiments

Experiments where one or two of the agents are using some learning algorithm to update their policy.

#### Random Defense
This is an experiment in the `random_defense-v9` environment. 
An environment where the defender is following a random defense policy.

- [tabular_q_learning_vs_random_defense-v9](training/v9/random_defense/tabular_q_learning/)
   * This experiment trains an attacker agent using tabular q-learning to act optimally in the given
   environment and defeat the random defender.
   
- [dqn_vs_random_defense-v9](training/v9/random_defense/dqn/)
   * This experiment trains an attacker agent using DQN to act optimally in the given
   environment and defeat the random defender.   

#### Minimal Defense

This is an experiment in the `minimal_defense-v9` environment.  
An environment where the defender is following the `defend_minimal` defense policy. 
The `defend_minimal` policy entails that the defender will always 
defend the attribute with the minimal value out of all of its neighbors.
   
- [tabular_q_learning_vs_minimal_defense-v9](training/v9/minimal_defense/tabular_q_learning/)
   * This experiment trains an attacker agent using tabular q-learning to act optimally in the given 
   environment and defeat the defender.   

- [dqn_vs_minimal_defense-v9](training/v9/minimal_defense/dqn/)
   * This experiment trains an attacker agent using DQN to act optimally in the given
   environment and defeat the random defender.

#### Random Attack   

This is an experiment in the `random_attack-v9` environment.  
An environment where the attack is following a random attack policy.
   
- [random_attack_vs_tabular_q_learning-v9](training/v9/random_attack/tabular_q_learning/)
   * This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and defeat the random attacker.
   
- [random_attack_vs_dqn-v9](training/v9/random_attack/dqn/)
   * This experiment trains a defender agent using dqn to act optimally in the given
   environment and defeat the random attacker.      

#### Maximal Attack

This is an experiment in the `maximal_attack-v9` environment.
An environment where the attack is following the `attack_maximal` attack policy.
The `attack_maximal` policy entails that the attacker will always attack the attribute with
the maximum value out of all of its neighbors.

- [maximal_attack_vs_tabular_q_learning-v9](training/v9/maximal_attack/tabular_q_learning/)
   * This experiment trains a defender agent using tabular q-learning to act optimally in the given
   environment and detect the attacker.
   
- [maximal_attack_vs_dqn-v9](training/v9/maximal_attack/dqn/)
   * This experiment trains a defender agent using dqn to act optimally in the given
   environment and detect the attacker.       

#### Two Agents
This is an experiment in the `idsgame-v9` environment. 
An environment where neither the attacker nor defender is part of the environment, i.e.
it is intended for 2-agent simulations or RL training.
      
- [tabular_q_learning_vs_tabular_q_learning-v9](training/v9/two_agents/tabular_q_learning/)
   * This experiment trains both an attacker and a defender agent simultaneously against each other 
   using tabular q-learning.
   
- [dqn_vs_dqn-v9](training/v9/two_agents/dqn/)
   * This experiment trains both an attacker and a defender agent simultaneously against each other 
   using DQN.       
   