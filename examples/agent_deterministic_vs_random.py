"""
Runner for the IDS Environment using a deterministic attack policy against a Random Defense Policy
"""
from gym_idsgame.envs.idsgame_env import IdsGameEnv
from gym_idsgame.agents.random_defense_agent import RandomDefenseBotAgent
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig
from gym_idsgame.envs.dao.render_config import RenderConfig
import time

if __name__ == '__main__':
    game_config = GameConfig(num_layers=2, num_servers_per_layer=3, num_attack_types=10, max_value=9)
    defender_policy = RandomDefenseBotAgent(game_config)
    render_config = RenderConfig(num_blinks=6, blink_interval=0.01)
    idsgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_policy,
                                   render_config=render_config)
    env = IdsGameEnv(idsgame_config=idsgame_config)
    actions = [42, 42, 42, 12, 12, 12, 3, 3, 3]
    i = 0
    done = False
    while True:
        #env.render()
        if done or i == len(actions):
            i = 0
            env.reset()
        time.sleep(0.001)
        action = actions[i]
        obs, reward, done, _ = env.step(action)
        node_id = env.state.get_attacker_node_from_observation(obs)
        i+=1
