.PHONY: clean plots

tabular_q:
	cd maximal_attack/tabular_q_learning && $(MAKE) run
	cd minimal_defense/tabular_q_learning && $(MAKE) run
	cd random_attack/tabular_q_learning && $(MAKE) run
	cd random_defense/tabular_q_learning && $(MAKE) run
	cd two_agents/tabular_q_learning && $(MAKE) run

tabular_q_server:
	cd maximal_attack/tabular_q_learning && $(MAKE) run_server
	cd minimal_defense/tabular_q_learning && $(MAKE) run_server
	cd random_attack/tabular_q_learning && $(MAKE) run_server
	cd random_defense/tabular_q_learning && $(MAKE) run_server
	cd two_agents/tabular_q_learning && $(MAKE) run_server

tabular_q_server_no_config:
	cd maximal_attack/tabular_q_learning && $(MAKE) run_server_no_config
	cd minimal_defense/tabular_q_learning && $(MAKE) run_server_no_config
	cd random_attack/tabular_q_learning && $(MAKE) run_server_no_config
	cd random_defense/tabular_q_learning && $(MAKE) run_server_no_config
	cd two_agents/tabular_q_learning && $(MAKE) run_server_no_config

dqn:
	cd maximal_attack/dqn && $(MAKE) run
	cd minimal_defense/dqn && $(MAKE) run
	cd random_attack/dqn && $(MAKE) run
	cd random_defense/dqn && $(MAKE) run
	cd two_agents/dqn && $(MAKE) run

dqn_server:
	cd maximal_attack/dqn && $(MAKE) run_server
	cd minimal_defense/dqn && $(MAKE) run_server
	cd random_attack/dqn && $(MAKE) run_server
	cd random_defense/dqn && $(MAKE) run_server
	cd two_agents/dqn && $(MAKE) run_server

dqn_server_no_config:
	cd maximal_attack/dqn && $(MAKE) run_server_no_config
	cd minimal_defense/dqn && $(MAKE) run_server_no_config
	cd random_attack/dqn && $(MAKE) run_server_no_config
	cd random_defense/dqn && $(MAKE) run_server_no_config
	cd two_agents/dqn && $(MAKE) run_server_no_config

plots:
	./make_plots.sh

all: run

clean:
	cd maximal_attack/tabular_q_learning && $(MAKE) clean
	cd minimal_defense/tabular_q_learning && $(MAKE) clean
	cd random_attack/tabular_q_learning && $(MAKE) clean
	cd random_defense/tabular_q_learning && $(MAKE) clean
	cd two_agents/tabular_q_learning && $(MAKE) clean

	cd maximal_attack/dqn && $(MAKE) clean
	cd minimal_defense/dqn && $(MAKE) clean
	cd random_attack/dqn && $(MAKE) clean
	cd random_defense/dqn && $(MAKE) clean
	cd two_agents/dqn && $(MAKE) clean

	rm -rf plots