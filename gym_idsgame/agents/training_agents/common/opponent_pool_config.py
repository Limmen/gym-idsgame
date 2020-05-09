import csv

class OpponentPoolConfig:

    def __init__(self, pool_maxsize : int = 10000, pool_increment_period : int = 1000,
                 head_to_head_period : int = 1000):
        """
        Initialize the DTO

        :param pool_maxsize: maximum size of the opponent pool, when max size is reached FIFO replacement will happen
        :param pool_increment_period: number of episodes of training before the policy is added to the pool
        :param head_to_head_period: the period of head-to-head between two policies before a new opponent is sampled
        """
        self.pool_maxsize = pool_maxsize
        self.pool_increment_period = pool_increment_period
        self.head_to_head_period = head_to_head_period

    def to_str(self) -> str:
        """
        :return: a string with information about all of the parameters
        """
        return "Opponent Pool Hyperparameters: pool_maxsize:{0},pool_increment_period:{1},h2h_period:{2}".format(
            self.pool_maxsize, self.pool_increment_period, self.head_to_head_period)

    def to_csv(self, file_path: str) -> None:
        """
        Write parameters to csv file

        :param file_path: path to the file
        :return: None
        """
        with open(file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["parameter", "value"])
            writer.writerow(["pool_maxsize", str(self.pool_maxsize)])
            writer.writerow(["pool_increment_period", str(self.pool_increment_period)])
            writer.writerow(["head_to_head_period", str(self.head_to_head_period)])