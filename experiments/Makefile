q_learning_vs_random_defense-v0: training/random_defense-v0/q_learning/run.py training/random_defense-v0/q_learning/run.sh
	cd training/random_defense-v0/q_learning/ && $(MAKE) run

random_vs_random-v0: simulations/v0/random_vs_random/run.py simulations/v0/random_vs_random/run.sh
	cd simulations/v0/random_vs_random/ && $(MAKE) run

random_vs_defend_minimal-v0: simulations/1l-1s-10ad-v0/random_vs_defend_minimal/run.py simulations/v0/random_vs_defend_minimal/run.sh
	cd simulations/v0/random_vs_defend_minimal/ && $(MAKE) run

attack_maximal_vs_defend_minimal-v0: simulations/v0/attack_maximal_vs_defend_minimal/run.py simulations/v0/attack_maximal_vs_defend_minimal/run.sh
	cd simulations/v0/attack_maximal_vs_defend_minimal/ && $(MAKE) run

attack_maximal_vs_random-v0: simulations/v0/attack_maximal_vs_random/run.py simulations/1l-1s-10ad-v0/attack_maximal_vs_random/run.sh
	cd simulations/v0/attack_maximal_vs_random/ && $(MAKE) run

v0: q_learning_vs_random_defense-v0 random_vs_random-v0 random_vs_defend_minimal-v0
v0: attack_maximal_vs_defend_minimal-v0 attack_maximal_vs_random-v0

random_vs_random-v1: simulations/v1/random_vs_random/run.py simulations/v1/random_vs_random/run.sh
	cd simulations/v1/random_vs_random/ && $(MAKE) run

random_vs_defend_minimal-v1: simulations/1l-1s-10ad-v1/random_vs_defend_minimal/run.py simulations/v1/random_vs_defend_minimal/run.sh
	cd simulations/v1/random_vs_defend_minimal/ && $(MAKE) run

attack_maximal_vs_defend_minimal-v1: simulations/v1/attack_maximal_vs_defend_minimal/run.py simulations/v1/attack_maximal_vs_defend_minimal/run.sh
	cd simulations/v1/attack_maximal_vs_defend_minimal/ && $(MAKE) run

attack_maximal_vs_random-v1: simulations/v1/attack_maximal_vs_random/run.py simulations/1l-1s-10ad-v1/attack_maximal_vs_random/run.sh
	cd simulations/v1/attack_maximal_vs_random/ && $(MAKE) run

v1: random_vs_random-v1 random_vs_defend_minimal-v1
v1: attack_maximal_vs_defend_minimal-v1 attack_maximal_vs_random-v1

all: v0 v1

clean:
	cd ./training/random_defense-v0/q_learning/ && $(MAKE) clean
	cd simulations/v0/random_vs_random/ && $(MAKE) clean
	cd simulations/v0/random_vs_defend_minimal/ && $(MAKE) clean
	cd simulations/v0/attack_maximal_vs_defend_minimal/ && $(MAKE) clean
	cd simulations/v0/attack_maximal_vs_random/ && $(MAKE) clean
	cd simulations/v1/random_vs_random/ && $(MAKE) clean
	cd simulations/v1/random_vs_defend_minimal/ && $(MAKE) clean
	cd simulations/v1/attack_maximal_vs_defend_minimal/ && $(MAKE) clean
	cd simulations/v1/attack_maximal_vs_random/ && $(MAKE) clean