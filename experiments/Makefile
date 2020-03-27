# V0 Experiments

tabular_q_learning_vs_random_defense-v0: training/v0/random_defense/tabular_q_learning training/v0/random_defense/tabular_q_learning
	cd training/random_defense/tabular_q_learning/ && $(MAKE) run

tabular_q_learning_vs_minimal_defense-v0: training/v0/minimal_defense/tabular_q_learning training/v0/minimal_defense/tabular_q_learning
	cd training/v0/minimal_defense/tabular_q_learning/ && $(MAKE) run

random_attack_vs_tabular_q_learning-v0: training/v0/random_attack/tabular_q_learning training/v0/random_attack/tabular_q_learning
	cd training/v0/random_attack/tabular_q_learning/ && $(MAKE) run

maximal_attack_vs_tabular_q_learning-v0: training/v0/maximal_attack/tabular_q_learning training/v0/maximal_attack/tabular_q_learning
	cd training/v0/maximal_attack/tabular_q_learning/ && $(MAKE) run

random_vs_random-v0: simulations/v0/random_vs_random/run.py simulations/v0/random_vs_random/run.sh
	cd simulations/v0/random_vs_random/ && $(MAKE) run

random_vs_defend_minimal-v0: simulations/v0/random_vs_defend_minimal/run.py simulations/v0/random_vs_defend_minimal/run.sh
	cd simulations/v0/random_vs_defend_minimal/ && $(MAKE) run

attack_maximal_vs_defend_minimal-v0: simulations/v0/attack_maximal_vs_defend_minimal/run.py simulations/v0/attack_maximal_vs_defend_minimal/run.sh
	cd simulations/v0/attack_maximal_vs_defend_minimal/ && $(MAKE) run

attack_maximal_vs_random-v0: simulations/v0/attack_maximal_vs_random/run.py simulations/v0/attack_maximal_vs_random/run.sh
	cd simulations/v0/attack_maximal_vs_random/ && $(MAKE) run

tabular_q_agent_vs_random-v0: simulations/v0/tabular_q_agent_vs_random/run.py simulations/v0/tabular_q_agent_vs_random/run.sh
	cd simulations/v0/tabular_q_agent_vs_random/ && $(MAKE) run

random_vs_tabular_q_agent-v0: simulations/v0/random_Vs_tabular_q_agent/run.py simulations/v0/random_vs_tabular_q_agent/run.sh
	cd simulations/v0/random_vs_tabular_q_agent/ && $(MAKE) run

v0: tabular_q_learning_vs_random_defense-v0 random_vs_random-v0 random_vs_defend_minimal-v0
v0: attack_maximal_vs_defend_minimal-v0 attack_maximal_vs_random-v0 random_attack_vs_tabular_q_learning-v0
v0: maximal_attack_vs_tabular_q_learning-v0 tabular_q_agent_vs_random-v0

# V1 Experiments

tabular_q_learning_vs_random_defense-v1: training/v1/random_defense/tabular_q_learning training/v1/random_defense/tabular_q_learning
	cd training/v1/random_defense/tabular_q_learning/ && $(MAKE) run

tabular_q_learning_vs_minimal_defense-v1: training/v1/minimal_defense/tabular_q_learning training/v1/minimal_defense/tabular_q_learning
	cd training/v1/minimal_defense/tabular_q_learning/ && $(MAKE) run

random_attack_vs_tabular_fq_learning-v1: training/v1/random_attack/tabular_q_learning training/v1/random_attack/tabular_q_learning
	cd training/v1/random_attack/tabular_q_learning/ && $(MAKE) run

maximal_attack_vs_tabular_q_learning-v1: training/v1/maximal_attack/tabular_q_learning training/v1/maximal_attack/tabular_q_learning
	cd training/v1/maximal_attack/tabular_q_learning/ && $(MAKE) run

random_vs_random-v1: simulations/v1/random_vs_random/run.py simulations/v1/random_vs_random/run.sh
	cd simulations/v1/random_vs_random/ && $(MAKE) run

random_vs_defend_minimal-v1: simulations/v1/random_vs_defend_minimal/run.py simulations/v1/random_vs_defend_minimal/run.sh
	cd simulations/v1/random_vs_defend_minimal/ && $(MAKE) run

attack_maximal_vs_defend_minimal-v1: simulations/v1/attack_maximal_vs_defend_minimal/run.py simulations/v1/attack_maximal_vs_defend_minimal/run.sh
	cd simulations/v1/attack_maximal_vs_defend_minimal/ && $(MAKE) run

attack_maximal_vs_random-v1: simulations/v1/attack_maximal_vs_random/run.py simulations/v1/attack_maximal_vs_random/run.sh
	cd simulations/v1/attack_maximal_vs_random/ && $(MAKE) run

v1: tabular_q_learning_vs_random_defense-v1 random_vs_random-v1 random_vs_defend_minimal-v1
v1: attack_maximal_vs_defend_minimal-v1 attack_maximal_vs_random-v1 random_attack_vs_tabular_q_learning-v1
v1: maximal_attack_vs_tabular_q_learning-v1

# V2 Experiments

tabular_q_learning_vs_random_defense-v2: training/v2/random_defense/tabular_q_learning training/v2/random_defense/tabular_q_learning
	cd training/v2/random_defense/tabular_q_learning/ && $(MAKE) run

tabular_q_learning_vs_minimal_defense-v2: training/v2/minimal_defense/tabular_q_learning training/v2/minimal_defense/tabular_q_learning
	cd training/v2/minimal_defense/tabular_q_learning/ && $(MAKE) run

random_attack_vs_tabular_fq_learning-v2: training/v2/random_attack/tabular_q_learning training/v2/random_attack/tabular_q_learning
	cd training/v2/random_attack/tabular_q_learning/ && $(MAKE) run

maximal_attack_vs_tabular_q_learning-v2: training/v2/maximal_attack/tabular_q_learning training/v2/maximal_attack/tabular_q_learning
	cd training/v2/maximal_attack/tabular_q_learning/ && $(MAKE) run

v2: tabular_q_learning_vs_random_defense-v2 tabular_q_learning_vs_minimal_defense-v2
v2: random_attack_vs_tabular_fq_learning-v2 maximal_attack_vs_tabular_q_learning-v2

# All

all: v0 v1 v2

# Clean

clean:

	# V0
	cd ./training/v0/random_defense/tabular_q_learning/ && $(MAKE) clean
	cd ./training/v0/minimal_defense/tabular_q_learning/ && $(MAKE) clean
	cd ./training/v0/random_attack/tabular_q_learning/ && $(MAKE) clean
	cd ./training/v0/maximal_attack/tabular_q_learning/ && $(MAKE) clean
	cd simulations/v0/random_vs_random/ && $(MAKE) clean
	cd simulations/v0/random_vs_defend_minimal/ && $(MAKE) clean
	cd simulations/v0/attack_maximal_vs_defend_minimal/ && $(MAKE) clean
	cd simulations/v0/attack_maximal_vs_random/ && $(MAKE) clean
	cd simulations/v0/tabular_q_agent_vs_random/ && $(MAKE) clean
	cd simulations/v0/random_vs_tabular_q_agent/ && $(MAKE) clean

	# V1
	cd ./training/v1/random_defense/tabular_q_learning/ && $(MAKE) clean
	cd ./training/v1/minimal_defense/tabular_q_learning/ && $(MAKE) clean
	cd ./training/v1/random_attack/tabular_q_learning/ && $(MAKE) clean
	cd ./training/v1/maximal_attack/tabular_q_learning/ && $(MAKE) clean
	cd simulations/v1/random_vs_random/ && $(MAKE) clean
	cd simulations/v1/random_vs_defend_minimal/ && $(MAKE) clean
	cd simulations/v1/attack_maximal_vs_defend_minimal/ && $(MAKE) clean
	cd simulations/v1/attack_maximal_vs_random/ && $(MAKE) clean

	# V2
	cd ./training/v2/random_defense/tabular_q_learning/ && $(MAKE) clean
	cd ./training/v2/minimal_defense/tabular_q_learning/ && $(MAKE) clean
	cd ./training/v2/random_attack/tabular_q_learning/ && $(MAKE) clean
	cd ./training/v2/maximal_attack/tabular_q_learning/ && $(MAKE) clean