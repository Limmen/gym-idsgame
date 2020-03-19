"""
Runner for the IDS Game GUI Manual Game
"""
from gym_idsgame.envs.idsgame_env import IdsGameEnv

if __name__ == '__main__':
    env = IdsGameEnv(num_layers = 2, num_servers_per_layer = 3, num_attack_types = 10, max_value = 10,
                     blink_interval=0.000001,num_blinks=6)
    actions = [12,12,12,42,42,42,73,73,73]
    i = 0
    done = False
    while True:
        env.render()
        if done:
            i = 0
            env.reset()
        #time.sleep(0.01)
        action = actions[i]
        s, reward, done, _ = env.step(action)
        i+=1
