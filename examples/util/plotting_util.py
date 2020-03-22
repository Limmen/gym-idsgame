"""
Basic plotting functions
"""

from typing import Union
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

def read_data(eval_file, train_file) -> Union[pd.DataFrame, pd.DataFrame]:
    """
    Utility function for reading csv files into pandas dataframes

    :param eval_file: path to the evaluation csv file
    :param train_file: path to the train csv file
    :return: eval_df, train_df
    """
    eval_df = pd.read_csv(eval_file)
    train_df = pd.read_csv(train_file)
    return eval_df, train_df

def simple_line_plot(x: np.ndarray, y: np.ndarray, title: str ="Test", xlabel: str ="test", ylabel: str ="test",
                     file_name: str ="test.eps", xlims: Union[float, float] = None, ylims: Union[float, float] = None,
                     log: bool = False, smooth: bool = True) -> None:
    """
    Plots a line plot with a raw line and a smooth line (optionally)

    :param x: data for x-axis
    :param y: data for y-axis
    :param title: title of the plot
    :param xlabel: label of x-axis
    :param ylabel: label of y-axis
    :param file_name: name of the file to save the plot
    :param xlims: limits for the x-axis
    :param ylims: limits for the y-axis
    :param log: whether to log-scale the y-axis
    :return: None
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
    if xlims is None:
        xlims = (min(x), max(x))
    if ylims is None:
        ylims = (min(y), max(y))
    ax.errorbar(x, y, yerr=None, color="red", ls='-', ecolor='black')
    if smooth:
        smooth = interp1d(x, y)
        x_smooth = np.linspace(min(x), max(x), len(x) // 10)
        ax.errorbar(x_smooth, smooth(x_smooth), yerr=None, color="black", ls='-', ecolor='black')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    fig.tight_layout()
    #fig.show()
    fig.savefig(file_name, format="png")
    #fig.savefig(file_name, format='eps', dpi=500, bbox_inches='tight', transparent=True)

def plot_results(avg_episode_rewards: np.ndarray, avg_episode_steps: np.ndarray, epsilon_values: np.ndarray,
                 hack_probability: np.ndarray, attacker_cumulative_reward: np.ndarray,
                 defender_cumulative_reward: np.ndarray, log_frequency: int, output_dir: str,
                 eval: bool = False) -> None:
    """
    Utility function for plotting results of an experiment in the idsgame environment

    :param avg_episode_rewards: list of average episode rewards recorded every <log_frequency>
    :param avg_episode_steps:  list of average episode steps recorded every <log_frequency>
    :param epsilon_values: list of epsilon values recorded every <log_frequency>
    :param hack_probability: list of hack probabilities recorded every <log_frequency>
    :param attacker_cumulative_reward: list of attacker cumulative rewards recorded every <log_frequency>
    :param defender_cumulative_reward: list of defender cumulative rewards recorded every <log_frequency>
    :param log_frequency: frequency that the metrics were recorded
    :param output_dir: base directory to save the plots
    :param eval: if True save plots with "eval.png" suffix, otherwise "train.png" suffix.
    :return: None
    """
    suffix = "train.png" if not eval else "eval.png"
    simple_line_plot(np.array(list(range(len(avg_episode_rewards))))*log_frequency, avg_episode_rewards,
                      title="Avg Episodic Returns",
                      xlabel="Episode", ylabel="Avg Return",
                     file_name=output_dir + "/plots/avg_episode_returns_" + suffix)
    simple_line_plot(np.array(list(range(len(avg_episode_steps))))*log_frequency, avg_episode_steps,
                     title="Avg Episode Lengths",
                     xlabel="Episode", ylabel="Avg Length (num steps)",
                     file_name=output_dir + "/plots/avg_episode_lengths_" + suffix)
    simple_line_plot(np.array(list(range(len(epsilon_values))))*log_frequency, epsilon_values,
                     title="Exploration rate (Epsilon)",
                     xlabel="Episode", ylabel="Epsilon", file_name=output_dir + "/plots/epsilon_" + suffix)
    simple_line_plot(np.array(list(range(len(hack_probability)))) * log_frequency, hack_probability,
                     title="Hack probability", ylims=(0,1),
                     xlabel="Episode", ylabel="P(Hacked)", file_name=output_dir + "/plots/hack_probability_" + suffix)
    simple_line_plot(np.array(list(range(len(attacker_cumulative_reward)))) * log_frequency, attacker_cumulative_reward,
                     title="Attacker Cumulative Reward",
                     xlabel="Episode", ylabel="Cumulative Reward",
                     file_name=output_dir + "/plots/attacker_cumulative_reward_" + suffix)
    simple_line_plot(np.array(list(range(len(defender_cumulative_reward)))) * log_frequency, defender_cumulative_reward,
                     title="Defender Cumulative Reward",
                     xlabel="Episode", ylabel="Cumulative Reward",
                     file_name=output_dir + "/plots/defender_cumulative_reward_" + suffix)

