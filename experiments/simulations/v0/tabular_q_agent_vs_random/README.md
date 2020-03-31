# Experiment `tabular_q_agent`_vs_`random`

This is an experiment in the `idsgame-v0` environment. 
An environment where neither the attacker nor defender is part of the environment, i.e.
it is intended for 2-agent simulations or RL training.

In this experiment, the attacker is implemented with a greedy policy 
based on a save Q-table. The defender is implemented with a random defense policy. 

The network configuration of the environment is as follows:

- `num_layers=1` (number of layers between the start and end nodes)
- `num_servers_per_layer=1`
- `num_attack_types=10`
- `max_value=9`  

<p align="center">
<img src="./docs/env.png" width="600">
</p>

The starting state for each node in the environment is initialized as follows (with some randomness for where the vulnerabilities are placed).

- `defense_val=2`
- `attack_val=0`
- `num_vulnerabilities_per_node=1` (which type of defense at the node that is vulnerable is selected randomly when the environment is initialized)
- `det_val=2`
- `vulnerability_val=0`  

## Environment 

- Env: `idsgame-v0`

## Algorithm

- Bots
 
## Instructions 

To configure the experiment use the `config.json` file. Alternatively, 
it is also possible to delete the config file and edit the configuration directly in
`run.py` (this will cause the configuration to be overridden on the next run). 

Example configuration in `config.json`:

```json
{
    "attacker_type": 0,
    "defender_type": 1,
    "env_name": "idsgame-v0",
    "idsgame_config": null,
    "initial_state_path": "/home/kim/storage/workspace/gym-idsgame/experiments/simulations/v0/tabular_q_agent_vs_random/initial_state/initial_state.pkl",
    "logger": null,
    "mode": 2,
    "output_dir": "/home/kim/storage/workspace/gym-idsgame/experiments/simulations/v0/tabular_q_agent_vs_random",
    "py/object": "gym_idsgame.config.client_config.ClientConfig",
    "q_agent_config": {
        "alpha": 0.1,
        "attacker": true,
        "defender": false,
        "epsilon": 0.9,
        "epsilon_decay": 0.999,
        "eval_episodes": 1,
        "eval_frequency": 1000,
        "eval_log_frequency": 1,
        "eval_render": false,
        "eval_sleep": 0.35,
        "gamma": 0.8,
        "gif_dir": null,
        "gifs": false,
        "load_path": "/home/kim/storage/workspace/gym-idsgame/experiments/simulations/v0/tabular_q_agent_vs_random/q_table/q_table.npy",
        "logger": null,
        "min_epsilon": 0.1,
        "num_episodes": 5000,
        "py/object": "gym_idsgame.agents.tabular_q_learning.q_agent_config.QAgentConfig",
        "render": false,
        "save_dir": null,
        "train_log_frequency": 100,
        "video": false,
        "video_dir": null,
        "video_fps": 5,
        "video_frequency": 1
    },
    "simulation_config": {
        "gif_dir": "/home/kim/storage/workspace/gym-idsgame/experiments/simulations/v0/tabular_q_agent_vs_random/gifs",
        "gifs": true,
        "log_frequency": 1,
        "logger": null,
        "num_episodes": 1000,
        "py/object": "gym_idsgame.simulation.dao.simulation_config.SimulationConfig",
        "render": true,
        "sleep": 0.8,
        "video": true,
        "video_dir": "/home/kim/storage/workspace/gym-idsgame/experiments/simulations/v0/tabular_q_agent_vs_random/videos",
        "video_fps": 5,
        "video_frequency": 1
    },
    "title": "TabularQAgentAttacker vs RandomDefender"
}
```

Example configuration in `run.py`:

```python
simulation_config = SimulationConfig(render=True, sleep=0.8, video=True, log_frequency=1,
                                     video_fps=5, video_dir=default_output_dir() + "/videos", num_episodes=1000,
                                     gifs=True, gif_dir=default_output_dir() + "/gifs", video_frequency = 1)
q_agent_config = QAgentConfig(load_path=default_output_dir() + "/q_table/q_table.npy")
env_name = "idsgame-v0"
client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.TABULAR_Q_AGENT.value,
                             defender_type=AgentType.RANDOM.value, mode=RunnerMode.SIMULATE.value,
                             simulation_config=simulation_config, output_dir=default_output_dir(),
                             title="TabularQAgentAttacker vs RandomDefender",
                             q_agent_config=q_agent_config,
                             initial_state_path = default_output_dir() + "/initial_state/initial_state.pkl")
```

## Example Simulation

<p align="center">
<img src="./docs/simulation.gif" width="600">
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