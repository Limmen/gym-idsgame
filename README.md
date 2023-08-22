# `gym-idsgame` An Abstract Cyber Security Simulation and Markov Game for OpenAI Gym

`gym-idsgame` is a reinforcement learning environment for simulating attack and defense operations in an abstract network intrusion
game. The environment extends the abstract model described in (Elderman et al. 2017). The model constitutes a
two-player Markov game between an attacker agent and a defender agent that face each other in a simulated computer
network. The reinforcement learning environment exposes an interface to a partially observed Markov decision process
(POMDP) model of the Markov game. The interface can be used to train, simulate, and evaluate attack- and defend policies against each other.
Moreover, the repository contains code to reproduce baseline results for various reinforcement learning algorithms, including:

- Tabular Q-learning
- Neural-fitted Q-learning using the DQN algorithm.
- REINFORCE with baseline
- Actor-Critic REINFORCE
- PPO

Please use this bibtex if you make use of this code in your publications (paper: https://arxiv.org/abs/2009.08120):
```
@INPROCEEDINGS{Hamm2011:Finding,
AUTHOR="Kim Hammar and Rolf Stadler",
TITLE="Finding Effective Security Strategies through Reinforcement Learning and
{Self-Play}",
BOOKTITLE="International Conference on Network and Service Management (CNSM 2020)
(CNSM 2020)",
ADDRESS="Izmir, Turkey",
DAYS=1,
MONTH=nov,
YEAR=2020,
KEYWORDS="Network Security; Reinforcement Learning; Markov Security Games",
ABSTRACT="We present a method to automatically find security strategies for the use
case of intrusion prevention. Following this method, we model the
interaction between an attacker and a defender as a Markov game and let
attack and defense strategies evolve through reinforcement learning and
self-play without human intervention. Using a simple infrastructure
configuration, we demonstrate that effective security strategies can emerge
from self-play. This shows that self-play, which has been applied in other
domains with great success, can be effective in the context of network
security. Inspection of the converged policies show that the emerged
policies reflect common-sense knowledge and are similar to strategies of
humans. Moreover, we address known challenges of reinforcement learning in
this domain and present an approach that uses function approximation, an
opponent pool, and an autoregressive policy representation. Through
evaluations we show that our method is superior to two baseline methods but
that policy convergence in self-play remains a challenge."
}
```

## Publications

- [CNSM20](https://ieeexplore.ieee.org/document/9269092)
- [CNSM21](http://dl.ifip.org/db/conf/cnsm/cnsm2021/1570732932.pdf)
- [TNSM22](https://arxiv.org/abs/2111.00289)
- [NOMS22](https://ieeexplore.ieee.org/document/9789707)
- [ICML22](https://limmen.dev/assets/papers/icml_ml4cyber_Hammar_Stadler_final_24_june_2022.pdf)
- [CNSM22](https://limmen.dev/assets/papers/CNSM22_preprint_8_sep_Hammar_Stadler.pdf)

<p align="center">
<img src="docs/episode_15000.gif" width="600">
</p>

## See also

- [awesome-rl-for-cybersecurity](https://github.com/Limmen/awesome-rl-for-cybersecurity)
- [gym-optimal-intrusion-response](https://github.com/Limmen/gym-optimal-intrusion-response)

Table of Contents
=================

   * [Design](#Design)
   * [Included Environments](#Included-Environments)
   * [Requirements](#Requirements)
   * [Installation](#Installation)
   * [Usage](#Usage)
   * [Manual Play](#Manual-Play)
   * [Baseline Experiments](#Baseline-Experiments)
   * [Future Work](#Future-Work)
   * [Author & Maintainer](#Author-&-Maintainer)
   * [Copyright and license](#copyright-and-license)

## Design

<p align="center">
<img src="docs/model.png" width="600">
</p>

## Included Environments

A rich set of configurations of the Markov game are registered as openAI gym environments.
The environments are specified and implemented in `gym_idsgame/envs/idsgame_env.py` see also `gym_idsgame/__init__.py`.

## `minimal_defense`

This is an environment where the agent is supposed to play the attacker in the Markov game and the defender is following the `defend_minimal` baseline defense policy.
The `defend_minimal` policy entails that the defender will always defend the attribute with the minimal value out of all of its neighbors.

Registered configurations:

- `idsgame-minimal_defense-v0`
- `idsgame-minimal_defense-v1`
- `idsgame-minimal_defense-v2`
- `idsgame-minimal_defense-v3`
- `idsgame-minimal_defense-v4`
- `idsgame-minimal_defense-v5`
- `idsgame-minimal_defense-v6`
- `idsgame-minimal_defense-v7`
- `idsgame-minimal_defense-v8`
- `idsgame-minimal_defense-v9`
- `idsgame-minimal_defense-v10`
- `idsgame-minimal_defense-v11`
- `idsgame-minimal_defense-v12`
- `idsgame-minimal_defense-v13`
- `idsgame-minimal_defense-v14`
- `idsgame-minimal_defense-v15`
- `idsgame-minimal_defense-v16`
- `idsgame-minimal_defense-v17`
- `idsgame-minimal_defense-v18`
- `idsgame-minimal_defense-v19`
- `idsgame-minimal_defense-v20`

## `maximal_attack`

This is an environment where the agent is supposed to play the defender and the attacker is following the `attack_maximal` baseline attack policy.
The `attack_maximal` policy entails that the attacker will always attack the attribute with the maximum value out of all of its neighbors.

Registered configurations:

- `idsgame-maximal_attack-v0`
- `idsgame-maximal_attack-v1`
- `idsgame-maximal_attack-v2`
- `idsgame-maximal_attack-v3`
- `idsgame-maximal_attack-v4`
- `idsgame-maximal_attack-v5`
- `idsgame-maximal_attack-v6`
- `idsgame-maximal_attack-v7`
- `idsgame-maximal_attack-v8`
- `idsgame-maximal_attack-v9`
- `idsgame-maximal_attack-v10`
- `idsgame-maximal_attack-v11`
- `idsgame-maximal_attack-v12`
- `idsgame-maximal_attack-v13`
- `idsgame-maximal_attack-v14`
- `idsgame-maximal_attack-v15`
- `idsgame-maximal_attack-v16`
- `idsgame-maximal_attack-v17`
- `idsgame-maximal_attack-v18`
- `idsgame-maximal_attack-v19`
- `idsgame-maximal_attack-v20`

## `random_attack`

This is an environment where the agent is supposed to play as the defender and the attacker is following a random baseline attack policy.

Registered configurations:

- `idsgame-random_attack-v0`
- `idsgame-random_attack-v1`
- `idsgame-random_attack-v2`
- `idsgame-random_attack-v3`
- `idsgame-random_attack-v4`
- `idsgame-random_attack-v5`
- `idsgame-random_attack-v6`
- `idsgame-random_attack-v7`
- `idsgame-random_attack-v8`
- `idsgame-random_attack-v9`
- `idsgame-random_attack-v10`
- `idsgame-random_attack-v11`
- `idsgame-random_attack-v12`
- `idsgame-random_attack-v13`
- `idsgame-random_attack-v14`
- `idsgame-random_attack-v15`
- `idsgame-random_attack-v16`
- `idsgame-random_attack-v17`
- `idsgame-random_attack-v18`
- `idsgame-random_attack-v19`
- `idsgame-random_attack-v20`

## `random_defense`

An environment where the agent is supposed to play as the attacker and the defender is following a random baseline defense policy.

Registered configurations:

- `idsgame-random_defense-v0`
- `idsgame-random_defense-v1`
- `idsgame-random_defense-v2`
- `idsgame-random_defense-v3`
- `idsgame-random_defense-v4`
- `idsgame-random_defense-v5`
- `idsgame-random_defense-v6`
- `idsgame-random_defense-v7`
- `idsgame-random_defense-v8`
- `idsgame-random_defense-v9`
- `idsgame-random_defense-v10`
- `idsgame-random_defense-v11`
- `idsgame-random_defense-v12`
- `idsgame-random_defense-v13`
- `idsgame-random_defense-v14`
- `idsgame-random_defense-v15`
- `idsgame-random_defense-v16`
- `idsgame-random_defense-v17`
- `idsgame-random_defense-v18`
- `idsgame-random_defense-v19`
- `idsgame-random_defense-v20`

## `two_agents`

This is an environment where neither the attacker nor defender is part of the environment, i.e. it is intended for 2-agent simulations or RL training.
In the experiments folder you can see examples of using this environment for training PPO-attacker vs PPO-defender, DQN-attacker vs REINFORCE-defender, etc..

Registered configurations:

- `idsgame-v0`
- `idsgame-v1`
- `idsgame-v2`
- `idsgame-v3`
- `idsgame-v4`
- `idsgame-v5`
- `idsgame-v6`
- `idsgame-v7`
- `idsgame-v8`
- `idsgame-v9`
- `idsgame-v10`
- `idsgame-v11`
- `idsgame-v12`
- `idsgame-v13`
- `idsgame-v14`
- `idsgame-v15`
- `idsgame-v16`
- `idsgame-v17`
- `idsgame-v18`
- `idsgame-v19`
- `idsgame-v20`

## Requirements
- Python 3.5+
- OpenAI Gym
- NumPy
- Pyglet (OpenGL 3D graphics)
- GPU for 3D graphics acceleration (optional)
- jsonpickle (for configuration files)
- torch (for baseline algorithms)


## Installation & Tests

```bash
# install from pip
pip install gym-idsgame==1.0.12

# git clone and install from source
git clone https://github.com/Limmen/gym-idsgame
cd gym-idsgame
pip3 install -e .

# local install from source
$ pip install -e gym-idsgame
# force upgrade deps
$ pip install -e gym-idsgame --upgrade

# run unit tests
pytest

# run it tests
cd experiments
make tests
```

## Usage
The environment can be accessed like any other OpenAI environment with `gym.make`.
Once the environment has been created, the API functions
`step()`, `reset()`, `render()`, and `close()` can be used to train any RL algorithm of
your preference.
```python
import gymnasium as gym
from gym_idsgame.envs import IdsGameEnv
env_name = "idsgame-maximal_attack-v3"
env = gym.make(env_name)
```

The environment ships with implementation of several baseline algorithms, e.g. the tabular Q(0) algorithm, see the example code below.

```python
import gymnasium as gym
from gym_idsgame.agents.training_agents.q_learning.q_agent_config import QAgentConfig
from gym_idsgame.agents.training_agents.q_learning.tabular_q_learning.tabular_q_agent import TabularQAgent
random_seed = 0
util.create_artefact_dirs(default_output_dir(), random_seed)
q_agent_config = QAgentConfig(gamma=0.999, alpha=0.0005, epsilon=1, render=False, eval_sleep=0.9,
                              min_epsilon=0.01, eval_episodes=100, train_log_frequency=100,
                              epsilon_decay=0.9999, video=True, eval_log_frequency=1,
                              video_fps=5, video_dir=default_output_dir() + "/results/videos/" + str(random_seed), num_episodes=20001,
                              eval_render=False, gifs=True, gif_dir=default_output_dir() + "/results/gifs/" + str(random_seed),
                              eval_frequency=1000, attacker=True, defender=False, video_frequency=101,
                              save_dir=default_output_dir() + "/results/data/" + str(random_seed))
env_name = "idsgame-minimal_defense-v2"
env = gym.make(env_name, save_dir=default_output_dir() + "/results/data/" + str(random_seed))
attacker_agent = TabularQAgent(env, q_agent_config)
attacker_agent.train()
train_result = attacker_agent.train_result
eval_result = attacker_agent.eval_result
```

## Manual Play
You can run the environment in a mode of "manual control" as well:

```python
from gym_idsgame.agents.manual_agents.manual_defense_agent import ManualDefenseAgent
random_seed = 0
env_name = "idsgame-random_attack-v2"
env = gym.make(env_name)
ManualDefenseAgent(env.idsgame_config)
```

## Baseline Experiments

The `experiments` folder contains results, hyperparameters and code to reproduce reported results using this environment.
For more information about each individual experiment, see this [README](experiments/README.md).

### Clean All Experiment Results

```bash
cd experiments # cd into experiments folder
make clean
```

### Run All Experiment Results (Takes a long time)

```bash
cd experiments # cd into experiments folder
make all
```

### Run All Experiments For a specific environment (Takes a long time)

```bash
cd experiments # cd into experiments folder
make v0
```

### Run a specific experiment

```bash
cd experiments/training/v0/random_defense/tabular_q_learning/ # cd into the experiment folder
make run
```

### Clean a specific experiment

```bash
cd experiments/training/v0/random_defense/tabular_q_learning/ # cd into the experiment folder
make clean
```

### Start tensorboard for a specifc experiment

```bash
cd experiments/training/v0/random_defense/tabular_q_learning/ # cd into the experiment folder
make tensorboard
```

### Fetch Baseline Experiment Results

By default when cloning the repo the experiment results are not included, to fetch the experiment results,
install and setup `git-lfs` then run:
```bash
git lfs fetch --all
git lfs pull
```

## Author & Maintainer

Kim Hammar <kimham@kth.se>

## Copyright and license

[LICENSE](LICENSE.md)

MIT

(C) 2020, Kim Hammar
