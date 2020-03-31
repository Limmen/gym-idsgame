# Experiment `minimal_defense-v1`_`manual_attacker`

This is an experiment in the `minimal_defense-v1` environment. 
An environment where the defender is following the `defend_minimal` defense policy. 
The `defend_minimal` policy entails that the defender will always 
defend the attribute with the minimal value out of all of its neighbors.
The experiment gives the control of the attacker to the user that can control the attacker
using the keyboard and mouse. 

The network configuration of the environment is as follows:

- `num_layers=1` (number of layers between the start and end nodes)
- `num_servers_per_layer=1`
- `num_attack_types=10`
- `max_value=9`  

<p align="center">
<img src="docs/env.png" width="600">
</p>

The starting state for each node in the environment is initialized as follows (with some randomness for where the vulnerabilities are placed).

- `defense_val=4`
- `attack_val=0`
- `num_vulnerabilities_per_node=4` (which type of defenses at the node that are vulnerable is selected randomly when the environment is initialized)
- `det_val=3`
- `vulnerability_val=0`

## Environment 

- Env: `random_defense-v1`

## Algorithm

- Manual play
 
## Instructions 

To configure the experiment use the `config.json` file. Alternatively, 
it is also possible to delete the config file and edit the configuration directly in
`run.py` (this will cause the configuration to be overridden on the next run). 

Example configuration in `config.json`:

```json
{
    "attacker_type": 3,
    "defender_type": 1,
    "env_name": "idsgame-minimal_defense-v1",
    "idsgame_config": null,
    "initial_state_path": null,
    "logger": null,
    "mode": 3,
    "output_dir": "/home/kim/storage/workspace/gym-idsgame/experiments/manual_play/v1/minimal_defense/manual_vs_random",
    "py/object": "gym_idsgame.config.client_config.ClientConfig",
    "q_agent_config": null,
    "simulation_config": null,
    "title": "ManualAttacker vs DefendMinimalDefender"
}
```

Example configuration in `run.py`:

```python
env_name = "idsgame-minimal_defense-v1"
client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.MANUAL_ATTACK.value,
                             mode=RunnerMode.MANUAL_ATTACKER.value, output_dir=default_output_dir(),
                             title="ManualAttacker vs DefendMinimalDefender")
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