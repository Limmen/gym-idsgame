.PHONY: clean tensorboard

run: run.py run.sh
	./run.sh

run_server: run.py run_server.sh
	./run_server.sh

tensorboard:
	tensorboard --logdir=./tensorboard

all: run

clean:
	rm -rf data logs plots hyperparameters gifs videos tensorboard nohup.out