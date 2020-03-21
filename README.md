# `gym-idsgame` TODO

TODO

Table of Contents
=================

   * [Useful Commands](#Useful-Commands)
   * [Requirements](#Requirements)
   * [Usage](#Usage)
   * [MDP Model](#MDP-Model)
   * [Author & Maintainer](#Author-&-Maintainer)
   * [Copyright and license](#copyright-and-license)

## Useful Commands

```bash
# install from pip
pip install gym-idsgame==1.0.0
# local install from source
$ pip install -e gym-idsgame
# force upgrade deps
$ pip install -e gym-idsgame --upgrade

# git clone and install from source
git clone https://github.com/Limmen/gym-idsgame
cd gym-idsgame
pip3 install -e .

# run tests
pytest
```

## Requirements
- Python 3.5+
- OpenAI Gym
- NumPy
- Pyglet (OpenGL 3D graphics)
- GPU for 3D graphics acceleration (optional)
- jsonpickle (for configuration files)

## Usage
The environment can be accessed like any other OpenAI environment with `gym.make`.
Once the environment has been created, the API functions
`step()`, `reset()`, `render()`, and `close()` can be used to train any RL algorithm of
your preference.
```python
import gym
> env = gym.make("gym_idsgame:idsgame-v1")
TODO
```

The environment ships with an implementation of the tabular Q(0) algorithm, see the example code below.

```python
import gym
from gym_idsgame.algorithms.q_agent import QAgent
TODO
```

## MDP Model
TODO

## Manual game
You can run the environment in a mode of "manual control" as well:

```python
from gym_ids.envs.rendering.viewer import Viewer
TODO
```

## Future Work

TODO

## Author & Maintainer

Kim Hammar <kimham@kth.se>

## Copyright and license

[LICENSE](LICENSE.md)

MIT

(C) 2020, Kim Hammar
