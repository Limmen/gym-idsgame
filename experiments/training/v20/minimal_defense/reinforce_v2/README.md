# Experiment `minimal_defense-v19`_`reinforce`

This is an experiment in the `minimal_defense-v19` environment. 
An environment where the defender is following the `defend_minimal` defense policy. 
The `defend_minimal` policy entails that the defender will always 
defend the attribute with the minimal value out of all of its neighbors.
 
This experiment trains an attacker agent using REINFORCE to act optimally in the given
environment and defeat the defender.

The network configuration of the environment is as follows:

- `num_layers=1` (number of layers between the start and end nodes)
- `num_servers_per_layer=1`
- `num_attack_types=4`
- `max_value=4`  

<p align="center">
<img src="docs/env.png" width="600">
</p>

The starting state for each node in the environment is initialized as follows (with some randomness for where the vulnerabilities are placed).

- `defense_val=3-4 (random)`
- `attack_val=0`
- `num_vulnerabilities_per_node=1` (which type of defense at the node that is vulnerable is selected randomly when the environment is initialized)
- `det_val=1`
- `vulnerability_val=0` 
- `num_vulnerabilities_per_layer=1`

The environment has dense rewards (+1,-1 given whenever the attacker reaches a new level in the network)

The environment is partially observed (attacker can only see attack attributes of neighboring nodes, defender can only see defense attributes).

The attacker is permitted to perform reconnaissance activities to explore the defense values of opponents.

The state of the environment is randomized on upon every `reset()`

## Environment 

- Env: `minimal_defense-v19`

## Algorithm

- REINFORCE  
 
## Instructions 

To configure the experiment use the `config.json` file. Alternatively, 
it is also possible to delete the config file and edit the configuration directly in
`run.py` (this will cause the configuration to be overridden on the next run). 

Example configuration in `config.json`:

```json
{
    "attacker_type": 7,
    "bot_attacker": false,
    "defender_type": 1,
    "env_name": "idsgame-minimal_defense-v14",
    "hp_tuning": false,
    "hp_tuning_config": null,
    "idsgame_config": null,
    "initial_state_path": null,
    "logger": null,
    "mode": 0,
    "output_dir": "/media/kim/HDD/workspace/gym-idsgame/experiments/training/v14/minimal_defense/reinforce",
    "pg_agent_config": {
        "alpha_attacker": 0.001,
        "alpha_defender": 0.1,
        "alternating_optimization": false,
        "alternating_period": 15000,
        "attacker": true,
        "attacker_load_path": null,
        "batch_size": 8,
        "checkpoint_freq": 1000,
        "clip_gradient": false,
        "critic_loss_fn": "MSE",
        "defender": false,
        "defender_load_path": null,
        "eps_clip": 0.2,
        "epsilon": 1,
        "epsilon_decay": 0.9999,
        "eval_episodes": 100,
        "eval_epsilon": 0.0,
        "eval_frequency": 1000,
        "eval_log_frequency": 1,
        "eval_render": false,
        "eval_sleep": 0.9,
        "gamma": 0.999,
        "gif_dir": "/media/kim/HDD/workspace/gym-idsgame/experiments/training/v14/minimal_defense/reinforce/results/gifs",
        "gifs": true,
        "gpu": true,
        "gpu_id": 0,
        "hidden_activation": "ReLU",
        "hidden_dim": 32,
        "input_dim_attacker": 14,
        "input_dim_defender": 30,
        "logger": null,
        "lr_decay_rate": 0.999,
        "lr_exp_decay": false,
        "lstm_network": false,
        "lstm_seq_length": 4,
        "max_gradient_norm": 40,
        "merged_ad_features": true,
        "min_epsilon": 0.01,
        "normalize_features": false,
        "num_episodes": 10000,
        "num_hidden_layers": 1,
        "num_lstm_layers": 2,
        "opponent_pool": false,
        "opponent_pool_config": null,
        "optimization_iterations": 28,
        "optimizer": "Adam",
        "output_dim_attacker": 8,
        "output_dim_defender": 33,
        "py/object": "gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config.PolicyGradientAgentConfig",
        "random_seed": 0,
        "render": false,
        "save_dir": "/media/kim/HDD/workspace/gym-idsgame/experiments/training/v14/minimal_defense/reinforce/results/data",
        "state_length": 1,
        "tensorboard": true,
        "tensorboard_dir": "/media/kim/HDD/workspace/gym-idsgame/experiments/training/v14/minimal_defense/reinforce/results/tensorboard",
        "train_log_frequency": 100,
        "video": true,
        "video_dir": "/media/kim/HDD/workspace/gym-idsgame/experiments/training/v14/minimal_defense/reinforce/results/videos",
        "video_fps": 5,
        "video_frequency": 101,
        "zero_mean_features": false
    },
    "py/object": "gym_idsgame.config.client_config.ClientConfig",
    "q_agent_config": null,
    "random_seed": 0,
    "random_seeds": [
        0,
        999,
        299,
        399,
        499
    ],
    "run_many": false,
    "simulation_config": null,
    "title": "REINFORCE vs DefendMinimalDefender"
}
```

Example configuration in `run.py`:

```python
pg_agent_config = PolicyGradientAgentConfig(gamma=0.999, alpha_attacker=0.001, epsilon=1, render=False, eval_sleep=0.9,
                                                min_epsilon=0.01, eval_episodes=100, train_log_frequency=100,
                                                epsilon_decay=0.9999, video=True, eval_log_frequency=1,
                                                video_fps=5, video_dir=default_output_dir() + "/results/videos",
                                                num_episodes=10000,
                                                eval_render=False, gifs=True,
                                                gif_dir=default_output_dir() + "/results/gifs",
                                                eval_frequency=1000, attacker=True, defender=False, video_frequency=101,
                                                save_dir=default_output_dir() + "/results/data",
                                                checkpoint_freq=1000, input_dim_attacker=(4 + 3) * 2,
                                                output_dim_attacker=4 * 2,
                                                hidden_dim=32,
                                                num_hidden_layers=1, batch_size=8,
                                                gpu=True, tensorboard=True,
                                                tensorboard_dir=default_output_dir() + "/results/tensorboard",
                                                optimizer="Adam", lr_exp_decay=False, lr_decay_rate=0.999,
                                                state_length=1, normalize_features=False, merged_ad_features=True,
                                                zero_mean_features=False, gpu_id=0, lstm_network=False,
                                                lstm_seq_length=4, num_lstm_layers=2)
env_name = "idsgame-minimal_defense-v19"
client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.REINFORCE_AGENT.value,
                             mode=RunnerMode.TRAIN_ATTACKER.value,
                             pg_agent_config=pg_agent_config, output_dir=default_output_dir(),
                             title="REINFORCE vs DefendMinimalDefender",
                             run_many=False, random_seeds=[0, 999, 299, 399, 499])
#client_config = hp_tuning_config(client_config)
return client_config
```

After the experiment has finished, the results are written to the following sub-directories:

- **/data**: CSV file with metrics per episode for train and eval, e.g. `avg_episode_rewards`, `avg_episode_steps`, etc.
- **/gifs**: If the gif configuration-flag is set to true, the experiment will render the game during evaluation and save gif files to this directory. You can control the frequency of evaluation with the configuration parameter `eval_frequency` and the frequency of video/gif recording during evaluation with the parameter `video_frequency`
- **/hyperparameters**: CSV file with hyperparameters for the experiment.
- **/logs**: Log files from the experiment
- **/plots**: Basic plots of the results
- **/videos**: If the video configuration-flag is set to true, the experiment will render the game during evaluation and save video files to this directory. You can control the frequency of evaluation with the configuration parameter `eval_frequency` and the frequency of video/gif recording during evaluation with the parameter `video_frequency`
  
## Example Results

<p align="center">
<img src="docs/avg_summary.png" width="800">
</p>

### Policy Inspection

#### Evaluation after 0 Training Episodes

<p align="center">
<img src="docs/episode_0.gif" width="600">
</p> 

#### Evaluation after 15000 Training Episodes

<p align="center">
<img src="docs/episode_15000.gif" width="600">
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

