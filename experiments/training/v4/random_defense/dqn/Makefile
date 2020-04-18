.PHONY: clean tensorboard

run: run.py run.sh
	./run.sh

run_server: run.py run_server.sh
	./run_server.sh

tensorboard:
	tensorboard --logdir=./tensorboard

plots:
	./make_plots.sh

all: run

clean:
	rm -rf results data logs plots hyperparameters gifs videos tensorboard nohup.out