import csv

class OpponentPoolConfig:

    def __init__(self, pool_maxsize : int = 10000, pool_increment_period : int = 1000,
                 head_to_head_period : int = 1000, quality_scores : bool = False, quality_score_eta : float = 0.01,
                 pool_prob : float = 0.20, initial_quality : int = 1):
        """
        Initialize the DTO

        :param pool_maxsize: maximum size of the opponent pool, when max size is reached FIFO replacement will happen
        :param pool_increment_period: number of episodes of training before the policy is added to the pool
        :param head_to_head_period: the period of head-to-head between two policies before a new opponent is sampled
        :param quality_scores: boolean flag whether to track quality scores of opponents in the pool
        :param quality_score_eta: the learning rate for updating quality scores in the opponent pool
        :param pool_prob: probability of selecting opponent from pool (probability of playing against current
                          best parameters is then 1-pool_prob)
        :param initial_quality: the initial quality when using quality scores for sampling from the pool
        """
        self.pool_maxsize = pool_maxsize
        self.pool_increment_period = pool_increment_period
        self.head_to_head_period = head_to_head_period
        self.quality_scores = quality_scores
        self.quality_score_eta = quality_score_eta
        self.pool_prob = pool_prob
        self.initial_quality = initial_quality

    def to_str(self) -> str:
        """
        :return: a string with information about all of the parameters
        """
        return "Opponent Pool Hyperparameters: pool_maxsize:{0},pool_increment_period:{1},h2h_period:{2}," \
               "quality_scores:{3},quality_score_eta:{4},pool_prob:{5},initial_quality:{6}".format(
            self.pool_maxsize, self.pool_increment_period, self.head_to_head_period, self.quality_scores,
            self.quality_score_eta, self.pool_prob, self.initial_quality)

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
            writer.writerow(["quality_scores", str(self.quality_scores)])
            writer.writerow(["quality_score_eta", str(self.quality_score_eta)])
            writer.writerow(["pool_prob", str(self.pool_prob)])
            writer.writerow(["initial_quality", str(self.initial_quality)])