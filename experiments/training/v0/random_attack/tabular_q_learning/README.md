# Experiment `random_attack-v0`_`tabular_q_learning`

This is an experiment in the `random_attack-v0` environment. 
An environment where the attack is following a random attack policy. 
This experiment trains a defender agent using tabular q-learning to act optimally in the given
environment and defeat the random attacker.

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

## Environment 

- Env: `random_attack-v0`

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
    "env_name": "idsgame-random_attack-v0",
    "logger": null,
    "mode": 1,
    "output_dir": tabular_q_learning,
    "py/object": "gym_idsgame.config.client_config.ClientConfig",
    "q_agent_config": {
        "alpha": 0.3,
        "attacker": false,
        "defender": true,
        "epsilon": 1,
        "epsilon_decay": 0.99,
        "eval_episodes": 1,
        "eval_frequency": 100,
        "eval_log_frequency": 1,
        "eval_render": false,
        "eval_sleep": 0.9,
        "gamma": 0.99,
        "gif_dir": tabular_q_learning,
        "gifs": true,
        "logger": null,
        "min_epsilon": 0.1,
        "num_episodes": 1000,
        "py/object": "gym_idsgame.agents.dao.q_agent_config.QAgentConfig",
        "render": false,
        "train_log_frequency": 100,
        "video": true,
        "video_dir": tabular_q_learning,
        "video_fps": 5,
        "video_frequency": 1
    },
    "simulation_config": null,
    "title": "RandomAttacker vs TrainingQAgent"
}
```

Example configuration in `run.py`:

```python
q_agent_config = QAgentConfig(gamma=0.99, alpha=0.3, epsilon=1, render=False, eval_sleep=0.9,
                              min_epsilon=0.1, eval_episodes=1, train_log_frequency=100,
                              epsilon_decay=0.99, video=True, eval_log_frequency=1,
                              video_fps=5, video_dir=default_output_dir() + "/videos", num_episodes=1000,
                              eval_render=False, gifs=True, gif_dir=default_output_dir() + "/gifs",
                              eval_frequency=100, attacker=False, defender=True)
env_name = "idsgame-random_attack-v0"
client_config = ClientConfig(env_name=env_name, defender_type=AgentType.Q_AGENT.value,
                             mode=RunnerMode.TRAIN_DEFENDER.value,
                             q_agent_config=q_agent_config, output_dir=default_output_dir(),
                             title="RandomAttacker vs TrainingQAgent")
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

### Policy Inspection

#### Evaluation after 0 Training Episodes

<p align="center">
<img src="docs/episode_0.gif" width="600">
</p> 

#### Evaluation after 100 Training Episodes

<p align="center">
<img src="docs/episode_100.gif" width="600">
</p>

#### Evaluation after 200 Training Episodes

<p align="center">
<img src="docs/episode_200.gif" width="600">
</p>  

#### Evaluation after 1000 Training Episodes

<p align="center">
<img src="docs/episode_1000.gif" width="600">
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

### Clean

```bash
make clean
```

