# Experiment `maximal_attack-v8`_`tabular_q_learning`

This is an experiment in the `maximal_attack-v8` environment.
An environment where the attack is following the `attack_maximal` attack policy.
The `attack_maximal` policy entails that the attacker will always attack the attribute with
the maximum value out of all of its neighbors. The defender is implemented with a
random defense policy.
 
This experiment trains a defender agent using tabular q-learning to act optimally in the given
environment and detect the attacker.

The network configuration of the environment is as follows:

- `num_layers=1` (number of layers between the start and end nodes)
- `num_servers_per_layer=1`
- `num_attack_types=10`
- `max_value=9`  

<p align="center">
<img src="docs/env.png" width="600">
</p>

The starting state for each node in the environment is initialized as follows (with some randomness for where the vulnerabilities are placed).

- `defense_val=2`
- `attack_val=0`
- `num_vulnerabilities_per_node=1` (which type of defense at the node that is vulnerable is selected randomly when the environment is initialized)
- `det_val=2`
- `vulnerability_val=0`
- `num_vulnerabilities_per_layer=1` 

The environment has dense rewards (+1,-1 given whenever the attacker reaches a new level in the network)

## Environment 

- Env: `random_attack-v8`

## Algorithm

- Tabular Q-learning with linear exploration annealing 
 
## Instructions 

To configure the experiment use the `config.json` file. Alternatively, 
it is also possible to delete the config file and edit the configuration directly in
`run.py` (this will cause the configuration to be overridden on the next run). 

Example configuration in `config.json`:

```json
{
    "attacker_type": 0,
    "defender_type": 0,
    "env_name": "idsgame-maximal_attack-v8",
    "idsgame_config": null,
    "initial_state_path": null,
    "logger": null,
    "mode": 1,
    "output_dir": "/Users/kimham/workspace/rl/gym-idsgame/experiments/training/v8/maximal_attack/tabular_q_learning",
    "py/object": "gym_idsgame.config.client_config.ClientConfig",
    "q_agent_config": {
        "alpha": 0.0005,
        "attacker": false,
        "attacker_load_path": null,
        "checkpoint_freq": 100000,
        "defender": true,
        "defender_load_path": null,
        "dqn_config": null,
        "epsilon": 1,
        "epsilon_decay": 0.9999,
        "eval_episodes": 100,
        "eval_frequency": 1000,
        "eval_log_frequency": 1,
        "eval_render": false,
        "eval_sleep": 0.9,
        "gamma": 0.999,
        "gif_dir": "/Users/kimham/workspace/rl/gym-idsgame/experiments/training/v8/maximal_attack/tabular_q_learning/results/gifs",
        "gifs": true,
        "logger": null,
        "min_epsilon": 0.01,
        "num_episodes": 15001,
        "py/object": "gym_idsgame.agents.q_learning.q_agent_config.QAgentConfig",
        "random_seed": 0,
        "render": false,
        "save_dir": "/Users/kimham/workspace/rl/gym-idsgame/experiments/training/v8/maximal_attack/tabular_q_learning/results/data",
        "train_log_frequency": 1,
        "video": true,
        "video_dir": "/Users/kimham/workspace/rl/gym-idsgame/experiments/training/v8/maximal_attack/tabular_q_learning/results/videos",
        "video_fps": 5,
        "video_frequency": 101
    },
    "random_seeds": [
        0,
        999,
        299,
        399,
        499
    ],
    "run_many": true,
    "simulation_config": null,
    "title": "AttackMaximalAttacker vs TrainingQAgent"
}
```

Example configuration in `run.py`:

```python
q_agent_config = QAgentConfig(gamma=0.999, alpha=0.0005, epsilon=1, render=False, eval_sleep=0.9,
                                  min_epsilon=0.01, eval_episodes=100, train_log_frequency=1,
                                  epsilon_decay=0.9999, video=True, eval_log_frequency=1,
                                  video_fps=5, video_dir=default_output_dir() + "/results/videos", num_episodes=15001,
                                  eval_render=False, gifs=True, gif_dir=default_output_dir() + "/results/gifs",
                                  eval_frequency=1000, attacker=False, defender=True,
                                  video_frequency=101,
                                  save_dir=default_output_dir() + "/results/data")
env_name = "idsgame-maximal_attack-v8"
client_config = ClientConfig(env_name=env_name, defender_type=AgentType.TABULAR_Q_AGENT.value,
                             mode=RunnerMode.TRAIN_DEFENDER.value,
                             q_agent_config=q_agent_config, output_dir=default_output_dir(),
                             title="AttackMaximalAttacker vs TrainingQAgent",
                             run_many=True, random_seeds=[0, 999, 299, 399, 499])
```

After the experiment has finished, the results are written to the following sub-directories:

- **/data**: CSV file with metrics per episode for train and eval, e.g. `avg_episode_rewards`, `avg_episode_steps`, etc.
- **/gifs**: If the gif configuration-flag is set to true, the experiment will render the game during evaluation and save gif files to this directory. You can control the frequency of evaluation with the configuration parameter `eval_frequency` and the frequency of video/gif recording during evaluation with the parameter `video_frequency`
- **/hyperparameters**: CSV file with hyperparameters for the experiment.
- **/logs**: Log files from the experiment
- **/plots**: Basic plots of the results
- **/videos**: If the video configuration-flag is set to true, the experiment will render the game during evaluation and save video files to this directory. You can control the frequency of evaluation with the configuration parameter `eval_frequency` and the frequency of video/gif recording during evaluation with the parameter `video_frequency`
  

## Example Results

### Hack probability

#### Train

<p align="center">
<img src="docs/hack_probability_train.png" width="800">
</p>

#### Eval

<p align="center">
<img src="docs/hack_probability_eval.png" width="800">
</p>

### Episode lengths

#### Train
<p align="center">
<img src="docs/avg_episode_lengths_train.png" width="800">
</p>

#### Eval

<p align="center">
<img src="docs/avg_episode_lengths_eval.png" width="800">
</p>

### Exploration Rate

<p align="center">
<img src="docs/epsilon_train.png" width="800">
</p>

### Cumulative Rewards

#### Attacker (Train)
<p align="center">
<img src="docs/attacker_cumulative_reward_train.png" width="800">
</p>

#### Defender (Train)
<p align="center">
<img src="docs/defender_cumulative_reward_train.png" width="800">
</p>

### Policy Inspection

#### Evaluation after 0 Training Episodes

<p align="center">
<img src="docs/episode_0.gif" width="600">
</p> 

#### Evaluation after 1000 Training Episodes

<p align="center">
<img src="docs/episode_1000.gif" width="600">
</p>

#### Evaluation after 5000 Training Episodes

<p align="center">
<img src="docs/episode_5000.gif" width="600">
</p>  

## Commands

Below is a list of commands for running the experiment

### Run

**Option 1**:
```bash
./run.sh
```

**Option 2**:
```bash
make all
```

**Option 3**:
```bash
python run.py
```

### Run Server (Without Display)

**Option 1**:
```bash
./run_server.sh
```

**Option 2**:
```bash
make run_server
```

### Clean

```bash
make clean
```

### Plots only

```
./make_plots.sh
```

or: 

```
make plots
```
