# V0 Experiments

tabular_q_learning_vs_random_defense-v0: training/random_defense-v0/tabular_q_learning training/random_defense-v0/tabular_q_learning
	cd training/random_defense-v0/tabular_q_learning/ && $(MAKE) run

tabular_q_learning_vs_minimal_defense-v0: training/minimal_defense-v0/tabular_q_learning training/minimal_defense-v0/tabular_q_learning
	cd training/minimal_defense-v0/tabular_q_learning/ && $(MAKE) run

random_attack_vs_tabular_q_learning-v0: training/random_attack-v0/tabular_q_learning training/random_attack-v0/tabular_q_learning
	cd training/random_attack-v0/tabular_q_learning/ && $(MAKE) run

maximal_attack_vs_tabular_q_learning-v0: training/maximal_attack-v0/tabular_q_learning training/maximal_attack-v0/tabular_q_learning
	cd training/maximal_attack-v0/tabular_q_learning/ && $(MAKE) run

random_vs_random-v0: simulations/v0/random_vs_random/run.py simulations/v0/random_vs_random/run.sh
	cd simulations/v0/random_vs_random/ && $(MAKE) run

random_vs_defend_minimal-v0: simulations/1l-1s-10ad-v0/random_vs_defend_minimal/run.py simulations/v0/random_vs_defend_minimal/run.sh
	cd simulations/v0/random_vs_defend_minimal/ && $(MAKE) run

attack_maximal_vs_defend_minimal-v0: simulations/v0/attack_maximal_vs_defend_minimal/run.py simulations/v0/attack_maximal_vs_defend_minimal/run.sh
	cd simulations/v0/attack_maximal_vs_defend_minimal/ && $(MAKE) run

attack_maximal_vs_random-v0: simulations/v0/attack_maximal_vs_random/run.py simulations/1l-1s-10ad-v0/attack_maximal_vs_random/run.sh
	cd simulations/v0/attack_maximal_vs_random/ && $(MAKE) run

v0: tabular_q_learning_vs_random_defense-v0 random_vs_random-v0 random_vs_defend_minimal-v0
v0: attack_maximal_vs_defend_minimal-v0 attack_maximal_vs_random-v0 random_attack_vs_tabular_q_learning-v0

# V1 Experiments

tabular_q_learning_vs_random_defense-v1: training/random_defense-v1/tabular_q_learning training/random_defense-v1/tabular_q_learning
	cd training/random_defense-v1/tabular_q_learning/ && $(MAKE) run

tabular_q_learning_vs_minimal_defense-v1: training/minimal_defense-v1/tabular_q_learning training/minimal_defense-v1/tabular_q_learning
	cd training/minimal_defense-v1/tabular_q_learning/ && $(MAKE) run

random_attack_vs_tabular_fq_learning-v1: training/random_attack-v1/tabular_q_learning training/random_attack-v1/tabular_q_learning
	cd training/random_attack-v1/tabular_q_learning/ && $(MAKE) run

maximal_attack_vs_tabular_q_learning-v1: training/maximal_attack-v1/tabular_q_learning training/maximal_attack-v1/tabular_q_learning
	cd training/maximal_attack-v1/tabular_q_learning/ && $(MAKE) run

random_vs_random-v1: simulations/v1/random_vs_random/run.py simulations/v1/random_vs_random/run.sh
	cd simulations/v1/random_vs_random/ && $(MAKE) run

random_vs_defend_minimal-v1: simulations/1l-1s-10ad-v1/random_vs_defend_minimal/run.py simulations/v1/random_vs_defend_minimal/run.sh
	cd simulations/v1/random_vs_defend_minimal/ && $(MAKE) run

attack_maximal_vs_defend_minimal-v1: simulations/v1/attack_maximal_vs_defend_minimal/run.py simulations/v1/attack_maximal_vs_defend_minimal/run.sh
	cd simulations/v1/attack_maximal_vs_defend_minimal/ && $(MAKE) run

attack_maximal_vs_random-v1: simulations/v1/attack_maximal_vs_random/run.py simulations/1l-1s-10ad-v1/attack_maximal_vs_random/run.sh
	cd simulations/v1/attack_maximal_vs_random/ && $(MAKE) run

v1: tabular_q_learning_vs_random_defense-v1 random_vs_random-v1 random_vs_defend_minimal-v1
v1: attack_maximal_vs_defend_minimal-v1 attack_maximal_vs_random-v1 random_attack_vs_tabular_q_learning-v1

# All

all: v0 v1

# Clean

clean:

	# V0
	cd ./training/random_defense-v0/tabular_q_learning/ && $(MAKE) clean
	cd ./training/minimal_defense-v0/tabular_q_learning/ && $(MAKE) clean
	cd ./training/random_attack-v0/tabular_q_learning/ && $(MAKE) clean
	cd ./training/maximal_attack-v0/tabular_q_learning/ && $(MAKE) clean
	cd simulations/v0/random_vs_random/ && $(MAKE) clean
	cd simulations/v0/random_vs_defend_minimal/ && $(MAKE) clean
	cd simulations/v0/attack_maximal_vs_defend_minimal/ && $(MAKE) clean
	cd simulations/v0/attack_maximal_vs_random/ && $(MAKE) clean

	# V1
	cd ./training/random_defense-v1/tabular_q_learning/ && $(MAKE) clean
	cd ./training/minimal_defense-v1/tabular_q_learning/ && $(MAKE) clean
	cd ./training/random_attack-v1/tabular_q_learning/ && $(MAKE) clean
	cd ./training/maximal_attack-v1/tabular_q_learning/ && $(MAKE) clean
	cd simulations/v1/random_vs_random/ && $(MAKE) clean
	cd simulations/v1/random_vs_defend_minimal/ && $(MAKE) clean
	cd simulations/v1/attack_maximal_vs_defend_minimal/ && $(MAKE) clean
	cd simulations/v1/attack_maximal_vs_random/ && $(MAKE) clean