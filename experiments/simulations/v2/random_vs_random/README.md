# Experiment `random`_vs_`random`

This is an experiment in the `idsgame-v2` environment. 
An environment where neither the attacker nor defender is part of the environment, i.e.
it is intended for 2-agent simulations or RL training.

In this experiment, the attacker is implemented with a random attack policy.
Similarly, the defender is implemented with a random defense policy.

The network configuration of the environment is as follows:

- `num_layers=1` (number of layers between the start and end nodes)
- `num_servers_per_layer=2`
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
- `num_vulnerabilities_per_layer=2`

## Environment 

- Env: `idsgame-v2`

## Algorithm

- Bots
 
## Instructions 

To configure the experiment use the `config.json` file. Alternatively, 
it is also possible to delete the config file and edit the configuration directly in
`run.py` (this will cause the configuration to be overridden on the next run). 

Example configuration in `config.json`:

```json
{
    "attacker_type": 1,
    "defender_type": 1,
    "env_name": "idsgame-v2",
    "idsgame_config": null,
    "initial_state_path": null,
    "logger": null,
    "mode": 2,
    "output_dir": "/home/kim/storage/workspace/gym-idsgame/experiments/simulations/v2/random_vs_random",
    "py/object": "gym_idsgame.config.client_config.ClientConfig",
    "q_agent_config": null,
    "simulation_config": {
        "gif_dir": "/home/kim/storage/workspace/gym-idsgame/experiments/simulations/v2/random_vs_random/gifs",
        "gifs": true,
        "log_frequency": 1,
        "logger": null,
        "num_episodes": 1000,
        "py/object": "gym_idsgame.simulation.dao.simulation_config.SimulationConfig",
        "render": false,
        "sleep": 0.8,
        "video": true,
        "video_dir": "/home/kim/storage/workspace/gym-idsgame/experiments/simulations/v2/random_vs_random/videos",
        "video_fps": 5,
        "video_frequency": 1
    },
    "title": "RandomAttacker vs RandomDefender"
}
```

Example configuration in `run.py`:

```python
simulation_config = SimulationConfig(render=False, sleep=0.8, video=True, log_frequency=1,
                                     video_fps=5, video_dir=default_output_dir() + "/videos", num_episodes=1000,
                                     gifs=True, gif_dir=default_output_dir() + "/gifs", video_frequency = 1)
env_name = "idsgame-v2"
client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.RANDOM.value,
                             defender_type=AgentType.RANDOM.value, mode=RunnerMode.SIMULATE.value,
                             simulation_config=simulation_config, output_dir=default_output_dir(),
                             title="RandomAttacker vs RandomDefender")
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