"""
Utility script for hyperparameter tuning
"""
import os
import time
import csv
from gym_idsgame.config.client_config import ClientConfig
from gym_idsgame.runnner import Runner

def create_dirs(output_dir: str, hparam_str : str) -> None:
    """
    Creates directories for hparam tuning results if they do not already exist

    :param output_dir: the base directory
    :param hparam_str: a string describing the hparam trial
    :return: None
    """
    if not os.path.exists(output_dir + "/" + hparam_str):
        os.makedirs(output_dir + "/" + hparam_str)


def hype_grid(client_config: ClientConfig):
    """
    Grid search for hyperparameter tuning

    :param client_config: client config for the experiment
    :param param_1: the name of the first hyperparameter
    :param param_2: the name of the second hyperparameter
    :return: None
    """
    assert client_config.hp_tuning
    assert client_config.hp_tuning_config is not None
    assert getattr(client_config.hp_tuning_config, client_config.hp_tuning_config.param_1) is not None
    assert getattr(client_config.hp_tuning_config, client_config.hp_tuning_config.param_2) is not None
    summary_results = []
    for p_1 in getattr(client_config.hp_tuning_config, client_config.hp_tuning_config.param_1):
        for p_2 in getattr(client_config.hp_tuning_config, client_config.hp_tuning_config.param_2):
            time_str = str(time.time())
            if client_config.logger is not None:
                client_config.logger.info("Starting Hyperparameter tuning with p_1: {}, eps_decay: {}".format(p_1, p_2))
            hparam_str = "{}={},{}={}".format(client_config.hp_tuning_config.param_1, p_1,
                                              client_config.hp_tuning_config.param_2, p_2)
            create_dirs(client_config.output_dir + "/results/hpo", hparam_str)

            try:
                setattr(client_config.q_agent_config, client_config.hp_tuning_config.param_1, p_1)
                setattr(client_config.q_agent_config, client_config.hp_tuning_config.param_2, p_2)
            except:
                try:
                    setattr(client_config.q_agent_config.dqn_config, client_config.hp_tuning_config.param_1, p_1)
                    setattr(client_config.q_agent_config.dqn_config, client_config.hp_tuning_config.param_2, p_2)
                except:
                    try:
                        setattr(client_config.pg_agent_config, client_config.hp_tuning_config.param_1, p_1)
                        setattr(client_config.pg_agent_config, client_config.hp_tuning_config.param_2, p_2)
                    except Exception as e:
                        raise ValueError("Could not find hparams")
            train_result, eval_result = Runner.run(client_config)
            if client_config.logger is not None:
                client_config.logger.info("Hyperparameter tuning with {}: {}, {}: {}, yielded hack prob:{}".format(
                    client_config.hp_tuning_config.param_1, p_1, client_config.hp_tuning_config.param_2,
                    p_2, eval_result.hack_probability[-1]))
            if len(train_result.avg_episode_steps) > 0 and len(eval_result.avg_episode_steps) > 0:
                train_csv_path = client_config.output_dir + "/results/hpo/" + hparam_str + "/" + \
                                 time_str + "_train" + ".csv"
                train_result.to_csv(train_csv_path)
                eval_csv_path = client_config.output_dir + "/results/hpo/" + hparam_str + "/" + time_str + "_eval" + ".csv"
                eval_result.to_csv(eval_csv_path)
                if client_config.q_agent_config is not None:
                    client_config.q_agent_config.to_csv(
                        client_config.output_dir + "/results/hpo/" + hparam_str + "/hparams_" + time_str + ".csv")
                if client_config.pg_agent_config is not None:
                    client_config.pg_agent_config.to_csv(
                        client_config.output_dir + "/results/hpo/" + hparam_str + "/hparams_" + time_str + ".csv")
            summary_results.append([p_1, p_2, eval_result.hack_probability[-1],
                                    train_result.cumulative_hack_probability[-1]])

    file_name = client_config.output_dir + "/results/hpo/" + "grid_" + client_config.hp_tuning_config.param_1 + "_" \
                + client_config.hp_tuning_config.param_2 + "_summary.csv"
    with open(file_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow([client_config.hp_tuning_config.param_1, client_config.hp_tuning_config.param_2, "eval_hp",
                         "cumulative_hp"])
        for row_d in summary_results:
            writer.writerow(row_d)




