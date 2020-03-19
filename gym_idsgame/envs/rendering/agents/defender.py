from gym_idsgame.envs.policy_baselines.policy import Policy

class Defender():
    """
    Represents a defender in the IDS game
    """
    def __init__(self, policy: Policy):
        """
        Initializes the defender with a defense-policy

        :param policy: the defense policy
        """
        self.policy = policy
