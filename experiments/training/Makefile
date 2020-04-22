.PHONY: clean plots

tabular_q_v7_v8_v9_server:
	cd v7/ && $(MAKE) tabular_q_server
	cd v8/ && $(MAKE) tabular_q_server
	cd v9/ && $(MAKE) tabular_q_server

plots:
	./make_plots.sh

all: run

clean:
	cd v7/ && $(MAKE) clean
	cd v8/ && $(MAKE) clean
	cd v9/ && $(MAKE) clean
	rm -rf plots