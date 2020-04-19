import os
import glob
import sys
import pandas as pd
from experiments.util import plotting_util


def get_script_path():
    """
    :return: the script path
    """
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def default_output_dir() -> str:
    """
    :return: the default output dir
    """
    script_dir = get_script_path()
    return script_dir


def default_config_path() -> str:
    """
    :return: the default path to configuration file
    """
    config_path = os.path.join(default_output_dir(), './config.json')
    return config_path


def plot_summary(algorithm : str, eval_freq : int, train_log_freq : int):
    seeds = [0, 999, 299, 399, 499]
    versions = [0,2,3]
    base_dir = default_output_dir()

    # V0
    maximal_attack_train_csv_paths_v0 = []
    maximal_attack_eval_csv_paths_v0 = []
    for seed in seeds:
        maximal_attack_train_csv_path = glob.glob(base_dir + "/v0/maximal_attack/" + algorithm +
                                                  "/results/data/" + str(seed) + "/*_train.csv")[0]
        maximal_attack_eval_csv_path = glob.glob(base_dir + "/v0/maximal_attack/" + algorithm +
                                                 "/results/data/" + str(seed) + "/*_eval.csv")[0]
        maximal_attack_train_csv_paths_v0.append(maximal_attack_train_csv_path)
        maximal_attack_eval_csv_paths_v0.append(maximal_attack_eval_csv_path)

    minimal_defense_train_csv_paths_v0 = []
    minimal_defense_eval_csv_paths_v0 = []
    for seed in seeds:
        base_dir = default_output_dir()
        minimal_defense_train_csv_path = glob.glob(base_dir + "/v0/minimal_defense/" + algorithm +
                                                   "/results/data/" + str(seed) + "/*_train.csv")[0]
        minimal_defense_eval_csv_path = glob.glob(base_dir + "/v0/minimal_defense/" + algorithm +
                                                  "/results/data/" + str(seed) + "/*_eval.csv")[0]
        minimal_defense_train_csv_paths_v0.append(minimal_defense_train_csv_path)
        minimal_defense_eval_csv_paths_v0.append(minimal_defense_eval_csv_path)

    random_attack_train_csv_paths_v0 = []
    random_attack_eval_csv_paths_v0 = []
    for seed in seeds:
        base_dir = default_output_dir()
        random_attack_train_csv_path = glob.glob(base_dir + "/v0/random_attack/" + algorithm +
                                                 "/results/data/" + str(seed) + "/*_train.csv")[0]
        random_attack_eval_csv_path = glob.glob(base_dir + "/v0/random_attack/" + algorithm +
                                                "/results/data/" + str(seed) + "/*_eval.csv")[0]
        random_attack_train_csv_paths_v0.append(random_attack_train_csv_path)
        random_attack_eval_csv_paths_v0.append(random_attack_eval_csv_path)

    random_defense_train_csv_paths_v0 = []
    random_defense_eval_csv_paths_v0 = []
    for seed in seeds:
        base_dir = default_output_dir()
        random_defense_train_csv_path = glob.glob(base_dir + "/v0/random_defense/" + algorithm +
                                                  "/results/data/" + str(seed) + "/*_train.csv")[0]
        random_defense_eval_csv_path = glob.glob(base_dir + "/v0/random_defense/" + algorithm +
                                                 "/results/data/" + str(seed) + "/*_eval.csv")[0]
        random_defense_train_csv_paths_v0.append(random_defense_train_csv_path)
        random_defense_eval_csv_paths_v0.append(random_defense_eval_csv_path)

    two_agents_train_csv_paths_v0 = []
    two_agents_eval_csv_paths_v0 = []
    for seed in seeds:
        base_dir = default_output_dir()
        two_agents_train_csv_path = glob.glob(base_dir + "/v0/two_agents/" + algorithm +
                                              "/results/data/" + str(seed) + "/*_train.csv")[0]
        two_agents_eval_csv_path = glob.glob(base_dir + "/v0/two_agents/" + algorithm +
                                             "/results/data/" + str(seed) + "/*_eval.csv")[0]
        two_agents_train_csv_paths_v0.append(two_agents_train_csv_path)
        two_agents_eval_csv_paths_v0.append(two_agents_eval_csv_path)



    # V2

    maximal_attack_train_csv_paths_v2 = []
    maximal_attack_eval_csv_paths_v2 = []
    for seed in seeds:
        maximal_attack_train_csv_path = glob.glob(base_dir + "/v2/maximal_attack/" + algorithm +
                                                  "/results/data/" + str(seed) + "/*_train.csv")[0]
        maximal_attack_eval_csv_path = glob.glob(base_dir + "/v2/maximal_attack/" + algorithm +
                                                 "/results/data/" + str(seed) + "/*_eval.csv")[0]
        maximal_attack_train_csv_paths_v2.append(maximal_attack_train_csv_path)
        maximal_attack_eval_csv_paths_v2.append(maximal_attack_eval_csv_path)

    minimal_defense_train_csv_paths_v2 = []
    minimal_defense_eval_csv_paths_v2 = []
    for seed in seeds:
        base_dir = default_output_dir()
        minimal_defense_train_csv_path = glob.glob(base_dir + "/v2/minimal_defense/" + algorithm +
                                                   "/results/data/" + str(seed) + "/*_train.csv")[0]
        minimal_defense_eval_csv_path = glob.glob(base_dir + "/v2/minimal_defense/" + algorithm +
                                                  "/results/data/" + str(seed) + "/*_eval.csv")[0]
        minimal_defense_train_csv_paths_v2.append(minimal_defense_train_csv_path)
        minimal_defense_eval_csv_paths_v2.append(minimal_defense_eval_csv_path)

    random_attack_train_csv_paths_v2 = []
    random_attack_eval_csv_paths_v2 = []
    for seed in seeds:
        base_dir = default_output_dir()
        random_attack_train_csv_path = glob.glob(base_dir + "/v2/random_attack/" + algorithm +
                                                 "/results/data/" + str(seed) + "/*_train.csv")[0]
        random_attack_eval_csv_path = glob.glob(base_dir + "/v2/random_attack/" + algorithm +
                                                "/results/data/" + str(seed) + "/*_eval.csv")[0]
        random_attack_train_csv_paths_v2.append(random_attack_train_csv_path)
        random_attack_eval_csv_paths_v2.append(random_attack_eval_csv_path)

    random_defense_train_csv_paths_v2 = []
    random_defense_eval_csv_paths_v2 = []
    for seed in seeds:
        base_dir = default_output_dir()
        random_defense_train_csv_path = glob.glob(base_dir + "/v2/random_defense/" + algorithm +
                                                  "/results/data/" + str(seed) + "/*_train.csv")[0]
        random_defense_eval_csv_path = glob.glob(base_dir + "/v2/random_defense/" + algorithm +
                                                 "/results/data/" + str(seed) + "/*_eval.csv")[0]
        random_defense_train_csv_paths_v2.append(random_defense_train_csv_path)
        random_defense_eval_csv_paths_v2.append(random_defense_eval_csv_path)

    two_agents_train_csv_paths_v2 = []
    two_agents_eval_csv_paths_v2 = []
    for seed in seeds:
        base_dir = default_output_dir()
        two_agents_train_csv_path = glob.glob(base_dir + "/v2/two_agents/" + algorithm +
                                              "/results/data/" + str(seed) + "/*_train.csv")[0]
        two_agents_eval_csv_path = glob.glob(base_dir + "/v2/two_agents/" + algorithm +
                                             "/results/data/" + str(seed) + "/*_eval.csv")[0]
        two_agents_train_csv_paths_v2.append(two_agents_train_csv_path)
        two_agents_eval_csv_paths_v2.append(two_agents_eval_csv_path)


    # V3
    maximal_attack_train_csv_paths_v3 = []
    maximal_attack_eval_csv_paths_v3 = []
    for seed in seeds:
        maximal_attack_train_csv_path = glob.glob(base_dir + "/v3/maximal_attack/" + algorithm +
                                                  "/results/data/" + str(seed) + "/*_train.csv")[0]
        maximal_attack_eval_csv_path = glob.glob(base_dir + "/v3/maximal_attack/" + algorithm +
                                                 "/results/data/" + str(seed) + "/*_eval.csv")[0]
        maximal_attack_train_csv_paths_v3.append(maximal_attack_train_csv_path)
        maximal_attack_eval_csv_paths_v3.append(maximal_attack_eval_csv_path)

    minimal_defense_train_csv_paths_v3 = []
    minimal_defense_eval_csv_paths_v3 = []
    for seed in seeds:
        base_dir = default_output_dir()
        minimal_defense_train_csv_path = glob.glob(base_dir + "/v3/minimal_defense/" + algorithm +
                                                   "/results/data/" + str(seed) + "/*_train.csv")[0]
        minimal_defense_eval_csv_path = glob.glob(base_dir + "/v3/minimal_defense/" + algorithm +
                                                  "/results/data/" + str(seed) + "/*_eval.csv")[0]
        minimal_defense_train_csv_paths_v3.append(minimal_defense_train_csv_path)
        minimal_defense_eval_csv_paths_v3.append(minimal_defense_eval_csv_path)

    random_attack_train_csv_paths_v3 = []
    random_attack_eval_csv_paths_v3 = []
    for seed in seeds:
        base_dir = default_output_dir()
        random_attack_train_csv_path = glob.glob(base_dir + "/v3/random_attack/" + algorithm +
                                                 "/results/data/" + str(seed) + "/*_train.csv")[0]
        random_attack_eval_csv_path = glob.glob(base_dir + "/v3/random_attack/" + algorithm +
                                                "/results/data/" + str(seed) + "/*_eval.csv")[0]
        random_attack_train_csv_paths_v3.append(random_attack_train_csv_path)
        random_attack_eval_csv_paths_v3.append(random_attack_eval_csv_path)

    random_defense_train_csv_paths_v3 = []
    random_defense_eval_csv_paths_v3 = []
    for seed in seeds:
        base_dir = default_output_dir()
        random_defense_train_csv_path = glob.glob(base_dir + "/v3/random_defense/" + algorithm +
                                                  "/results/data/" + str(seed) + "/*_train.csv")[0]
        random_defense_eval_csv_path = glob.glob(base_dir + "/v3/random_defense/" + algorithm +
                                                 "/results/data/" + str(seed) + "/*_eval.csv")[0]
        random_defense_train_csv_paths_v3.append(random_defense_train_csv_path)
        random_defense_eval_csv_paths_v3.append(random_defense_eval_csv_path)

    two_agents_train_csv_paths_v3 = []
    two_agents_eval_csv_paths_v3 = []
    for seed in seeds:
        base_dir = default_output_dir()
        two_agents_train_csv_path = glob.glob(base_dir + "/v3/two_agents/" + algorithm +
                                              "/results/data/" + str(seed) + "/*_train.csv")[0]
        two_agents_eval_csv_path = glob.glob(base_dir + "/v3/two_agents/" + algorithm +
                                             "/results/data/" + str(seed) + "/*_eval.csv")[0]
        two_agents_train_csv_paths_v3.append(two_agents_train_csv_path)
        two_agents_eval_csv_paths_v3.append(two_agents_eval_csv_path)

    plotting_util.plot_all_averages_multiple_versions(
        maximal_attack_train_csv_paths_v0,  maximal_attack_eval_csv_paths_v0, minimal_defense_train_csv_paths_v0,
        minimal_defense_eval_csv_paths_v0,random_attack_train_csv_paths_v0, random_attack_eval_csv_paths_v0,
        random_defense_train_csv_paths_v0, random_defense_eval_csv_paths_v0,
        two_agents_train_csv_paths_v0, two_agents_eval_csv_paths_v0,
        maximal_attack_train_csv_paths_v2, maximal_attack_eval_csv_paths_v2,
        minimal_defense_train_csv_paths_v2, minimal_defense_eval_csv_paths_v2,
        random_attack_train_csv_paths_v2, random_attack_eval_csv_paths_v2,
        random_defense_train_csv_paths_v2, random_defense_eval_csv_paths_v2,
        two_agents_train_csv_paths_v2, two_agents_eval_csv_paths_v2,
        maximal_attack_train_csv_paths_v3, maximal_attack_eval_csv_paths_v3,
        minimal_defense_train_csv_paths_v3, minimal_defense_eval_csv_paths_v3,
        random_attack_train_csv_paths_v3, random_attack_eval_csv_paths_v3,
        random_defense_train_csv_paths_v3, random_defense_eval_csv_paths_v3,
        two_agents_train_csv_paths_v3, two_agents_eval_csv_paths_v3,
        algorithm, default_output_dir() + "/plots", eval_freq, train_log_freq)


def plot():
    if not os.path.exists(default_output_dir() + "/plots"):
        os.makedirs(default_output_dir() + "/plots")

    hyperparam_csv_path = glob.glob(default_output_dir() +
                                    "/v0/random_defense/tabular_q_learning/results/hyperparameters/0/*.csv")[0]
    hyperparameters = pd.read_csv(hyperparam_csv_path)
    eval_freq = hyperparameters.loc[hyperparameters['parameter'] == "eval_frequency"]["value"].values[0]
    train_log_freq = hyperparameters.loc[hyperparameters['parameter'] == "train_log_frequency"]["value"].values[0]
    plot_summary("tabular_q_learning", int(eval_freq), int(train_log_freq))


if __name__ == '__main__':
    plot()
