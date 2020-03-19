from gym_idsgame.envs.policy_baselines.policy import Policy

class Defender():

    def __init__(self, policy: Policy):
        self.policy = policy
