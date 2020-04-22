.PHONY: clean plots

tabular_q:
	cd maximal_attack/tabular_q_learning && $(MAKE) run
	cd minimal_defense/tabular_q_learning && $(MAKE) run
	cd random_attack/tabular_q_learning && $(MAKE) run
	cd random_defense/tabular_q_learning && $(MAKE) run
	cd two_agents/tabular_q_learning && $(MAKE) run

tabular_q_server
	cd maximal_attack/tabular_q_learning && $(MAKE) run_server
	cd minimal_defense/tabular_q_learning && $(MAKE) run_server
	cd random_attack/tabular_q_learning && $(MAKE) run_server
	cd random_defense/tabular_q_learning && $(MAKE) run_server
	cd two_agents/tabular_q_learning && $(MAKE) run_server

plots:
	./make_plots.sh

all: run

clean:
	cd maximal_attack/tabular_q_learning && $(MAKE) clean
	cd minimal_defense/tabular_q_learning && $(MAKE) clean
	cd random_attack/tabular_q_learning && $(MAKE) clean
	cd random_defense/tabular_q_learning && $(MAKE) clean
	cd two_agents/tabular_q_learning && $(MAKE) clean
	rm -rf plots