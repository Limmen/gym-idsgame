import gym
from gym_idsgame.envs import IdsGameEnv

def attack_against_baseline_defense_env():
    versions = range(0,20)
    version = versions[0]
    env_name = "idsgame-minimal_defense-v" + str(version)
    env = gym.make(env_name)
    done = False
    while not done:
        attack_action = env.attacker_action_space.sample()
        defense_action = None
        a = (attack_action, defense_action)
        obs, reward, done, info = env.step(a)


def attack_against_random_defense_env():
    versions = range(0,20)
    version = versions[0]
    env_name = "idsgame-random_defense-v" + str(version)
    env = gym.make(env_name)
    done = False
    while not done:
        attack_action = env.attacker_action_space.sample()
        defense_action = None
        a = (attack_action, defense_action)
        obs, reward, done, info = env.step(a)

def defense_against_baseline_attack_env():
    versions = range(0,20)
    version = versions[0]
    env_name = "idsgame-maximal_attack-v" + str(version)
    env = gym.make(env_name)
    done = False
    while not done:
        attack_action = None
        defense_action = env.defender_action_space.sample()
        a = (attack_action, defense_action)
        obs, reward, done, info = env.step(a)


def defense_against_random_attack_env():
    versions = range(0,20)
    version = versions[0]
    env_name = "idsgame-random_attack-v" + str(version)
    env = gym.make(env_name)
    done = False
    while not done:
        attack_action = None
        defense_action = env.defender_action_space.sample()
        a = (attack_action, defense_action)
        obs, reward, done, info = env.step(a)

def two_agents_env():
    versions = range(0,20)
    version = versions[0]
    env_name = "idsgame-v" + str(version)
    env = gym.make(env_name)
    done = False
    while not done:
        attack_action = env.attacker_action_space.sample()
        defense_action = env.defender_action_space.sample()
        a = (attack_action, defense_action)
        obs, reward, done, info = env.step(a)

def main():
    #attack_against_baseline_defense_env()
    attack_against_random_defense_env()
    #defense_against_baseline_attack_env()
    #defense_against_random_attack_env()
    #two_agents_env()

if __name__ == '__main__':
    main()