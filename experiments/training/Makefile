.PHONY: clean plots

tabular_q_v7_v8_v9_server:
	cd v7/ && $(MAKE) tabular_q_server
	cd v8/ && $(MAKE) tabular_q_server
	cd v9/ && $(MAKE) tabular_q_server

tabular_q_v7_v8_v9_server_no_config:
	cd v7/ && $(MAKE) tabular_q_server_no_config
	cd v8/ && $(MAKE) tabular_q_server_no_config
	cd v9/ && $(MAKE) tabular_q_server_no_config

tabular_q_v0_v2_v3_server:
	cd v0/ && $(MAKE) tabular_q_server
	cd v2/ && $(MAKE) tabular_q_server
	cd v3/ && $(MAKE) tabular_q_server

tabular_q_v0_v2_v3_server_no_config:
	cd v0/ && $(MAKE) tabular_q_server_no_config
	cd v2/ && $(MAKE) tabular_q_server_no_config
	cd v3/ && $(MAKE) tabular_q_server_no_config

dqn_v7_v8_v9_server:
	cd v7/ && $(MAKE) dqn_server
	cd v8/ && $(MAKE) dqn_server
	cd v9/ && $(MAKE) dqn_server

dqn_v7_v8_v9_server_no_config:
	cd v7/ && $(MAKE) dqn_server_no_config
	cd v8/ && $(MAKE) dqn_server_no_config
	cd v9/ && $(MAKE) dqn_server_no_config

dqn_v0_v2_v3_server:
	cd v0/ && $(MAKE) dqn_server
	cd v2/ && $(MAKE) dqn_server
	cd v3/ && $(MAKE) dqn_server

dqn_v0_v2_v3_server_no_config:
	cd v0/ && $(MAKE) dqn_server_no_config
	cd v2/ && $(MAKE) dqn_server_no_config
	cd v3/ && $(MAKE) dqn_server_no_config

plots:
	./make_plots.sh

all: run

clean:
	cd v7/ && $(MAKE) clean
	cd v8/ && $(MAKE) clean
	cd v9/ && $(MAKE) clean
	cd v0/ && $(MAKE) clean
	cd v2/ && $(MAKE) clean
	cd v3/ && $(MAKE) clean
	rm -rf plots