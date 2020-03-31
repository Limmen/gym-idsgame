# Experiment `random_attack-v4`_`manual_defender`

This is an experiment in the `random_attack-v4` environment. 
An environment where the attacker is following a random attack policy.
The experiment gives the control of the defender to the user that can control the attacker
using the keyboard and mouse. 

The network configuration of the environment is as follows:

- `num_layers=4` (number of layers between the start and end nodes)
- `num_servers_per_layer=5`
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

- Env: `random_attack-v4`

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
    "defender_type": 4,
    "env_name": "idsgame-random_attack-v4",
    "idsgame_config": null,
    "initial_state_path": null,
    "logger": null,
    "mode": 4,
    "output_dir": "/home/kim/storage/workspace/gym-idsgame/experiments/manual_play/v4/random_attack/random_vs_manual",
    "py/object": "gym_idsgame.config.client_config.ClientConfig",
    "q_agent_config": null,
    "simulation_config": null,
    "title": "RandomAttacker vs ManualDefender"
}
```

Example configuration in `run.py`:

```python
env_name = "idsgame-random_attack-v4"
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

### Clean

```bash
make clean
```