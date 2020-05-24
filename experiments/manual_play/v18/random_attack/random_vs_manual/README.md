# Experiment `random_attack-v18`_`manual_attacker`

This is an experiment in the `random_attack-v18` environment. 
An environment where the attacker is following a random attack policy.
The experiment gives the control of the defender to the user that can control the attacker
using the keyboard and mouse.   

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

- Env: `random_attack-v18`

## Algorithm

- Manual play
 
## Instructions 

To configure the experiment use the `config.json` file. Alternatively, 
it is also possible to delete the config file and edit the configuration directly in
`run.py` (this will cause the configuration to be overridden on the next run). 

Example configuration in `config.json`:

```json
{
    "attacker_type": 0,
    "bot_attacker": false,
    "defender_type": 4,
    "env_name": "idsgame-random_attack-v18",
    "hp_tuning": false,
    "hp_tuning_config": null,
    "idsgame_config": null,
    "initial_state_path": null,
    "logger": null,
    "mode": 4,
    "output_dir": "/home/kim/storage/workspace/gym-idsgame/experiments/manual_play/v18/random_attack/random_vs_manual",
    "pg_agent_config": null,
    "py/object": "gym_idsgame.config.client_config.ClientConfig",
    "q_agent_config": null,
    "random_seed": 0,
    "random_seeds": null,
    "run_many": false,
    "simulation_config": null,
    "title": "RandomAttacker vs ManualDefender"
}
```

Example configuration in `run.py`:

```python
env_name = "idsgame-random_attack-v18"
client_config = ClientConfig(env_name=env_name, defender_type=AgentType.MANUAL_DEFENSE.value,
                             mode=RunnerMode.MANUAL_DEFENDER.value, output_dir=default_output_dir(),
                             title="RandomAttacker vs ManualDefender")
```

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