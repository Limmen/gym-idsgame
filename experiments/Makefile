random_defense-1l-1s-10ad-v0: training/random_defense-1l-1s-10ad-v0/q_learning/run.py training/random_defense-1l-1s-10ad-v0/q_learning/run.sh
	cd training/random_defense-1l-1s-10ad-v0/q_learning/ && $(MAKE) run

random_vs_random-1l-1s-10ad-v0: simulations/1l-1s-10ad-v0/random_vs_random/run.py simulations/1l-1s-10ad-v0/random_vs_random/run.sh
	cd simulations/1l-1s-10ad-v0/random_vs_random/ && $(MAKE) run

random_vs_defend_minimal-1l-1s-10ad-v0: simulations/1l-1s-10ad-v0/random_vs_defend_minimal/run.py simulations/1l-1s-10ad-v0/random_vs_defend_minimal/run.sh
	cd simulations/1l-1s-10ad-v0/random_vs_random/ && $(MAKE) run

all: random_defense-1l-1s-10ad-v0 random_vs_random-1l-1s-10ad-v0

clean:
	cd ./training/random_defense-1l-1s-10ad-v0/q_learning/ && $(MAKE) clean