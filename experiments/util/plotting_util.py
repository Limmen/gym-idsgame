"""
Basic plotting functions
"""

import csv
from typing import Union
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import os
import matplotlib.ticker as tkr

def read_data(file) -> pd.DataFrame:
    """
    Utility function for reading csv files into pandas dataframes

    :param file: path to the csv file
    :return: df
    """
    return pd.read_csv(file)


def plot_all(train_df, eval_df, eval_step, a_state_values, file_name):
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(16, 6))

    # Plot avg_episode_steps
    xlims = (min(min(np.array(list(range(len(train_df["avg_episode_steps"]))))),
                 min(np.array(list(range(len(eval_df["avg_episode_steps"])))) * eval_step)),
             max(max(np.array(list(range(len(train_df["avg_episode_steps"]))))),
                 max(np.array(list(range(len(eval_df["avg_episode_steps"])))) * eval_step)))
    ylims = (min(min(train_df["avg_episode_steps"]), min(eval_df["avg_episode_steps"])),
             max(max(train_df["avg_episode_steps"]), max(eval_df["avg_episode_steps"])))

    ax[0][0].errorbar(np.array(list(range(len(train_df["avg_episode_steps"])))),
                      train_df["avg_episode_steps"], yerr=None, ls='-', color="#599ad3", label="Train")
    ax[0][0].errorbar(np.array(list(range(len(eval_df["avg_episode_steps"])))) * eval_step,
                      eval_df["avg_episode_steps"], yerr=None, ls='--', color='#f9a65a', label="Eval")

    ax[0][0].set_title("Avg Episode Lengths")
    ax[0][0].set_xlabel("Episode \#")
    ax[0][0].set_ylabel("Avg Length (num steps)")

    # set the grid on
    ax[0][0].grid('on')

    # tweak the axis labels
    xlab = ax[0][0].xaxis.get_label()
    ylab = ax[0][0].yaxis.get_label()

    xlab.set_size(10)
    ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax[0][0].spines['right'].set_color((.8, .8, .8))
    ax[0][0].spines['top'].set_color((.8, .8, .8))

    ax[0][0].legend(loc="upper right")

    # Plot Cumulative Reward
    xlims = (min(min(np.array(list(range(len(train_df["attacker_cumulative_reward"]))))),
                 min(np.array(list(range(len(train_df["defender_cumulative_reward"])))))),
             max(max(np.array(list(range(len(train_df["attacker_cumulative_reward"]))))),
                 max(np.array(list(range(len(train_df["defender_cumulative_reward"])))))))
    ylims = (min(min(train_df["attacker_cumulative_reward"]), min(train_df["defender_cumulative_reward"])),
             max(max(train_df["attacker_cumulative_reward"]), max(train_df["defender_cumulative_reward"])))

    ax[0][1].errorbar(np.array(list(range(len(train_df["attacker_cumulative_reward"])))),
                      train_df["attacker_cumulative_reward"], yerr=None, ls='-', color="#599ad3", label="Attacker")
    ax[0][1].errorbar(np.array(list(range(len(train_df["defender_cumulative_reward"])))),
                      train_df["defender_cumulative_reward"], yerr=None, ls='--', color='#f9a65a', label="Defender")

    ax[0][1].set_title("Cumulative Reward")
    ax[0][1].set_xlabel("Episode \#")
    ax[0][1].set_ylabel("Cumulative Reward")

    # set the grid on
    ax[0][1].grid('on')

    # tweak the axis labels
    xlab = ax[0][1].xaxis.get_label()
    ylab = ax[0][1].yaxis.get_label()

    xlab.set_size(10)
    ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax[0][1].spines['right'].set_color((.8, .8, .8))
    ax[0][1].spines['top'].set_color((.8, .8, .8))

    ax[0][1].legend(loc="upper left")

    # Plot histogram of average episode lengths

    weights_1 = np.ones_like(train_df["avg_episode_steps"].values) / float(len(train_df["avg_episode_steps"].values))
    weights_2 = np.ones_like(eval_df["avg_episode_steps"].values) / float(len(eval_df["avg_episode_steps"].values))
    ax[0, 2].hist(train_df["avg_episode_steps"].values, alpha=0.5, bins=5, weights=weights_1,
                  color="#599ad3", label="Train",
                  stacked=True)
    ax[0, 2].hist(eval_df["avg_episode_steps"].values, alpha=0.5, bins=5, weights=weights_2,
                  color='#f9a65a', label="Eval",
                  stacked=True)
    ax[0, 2].set_title("Avg Episode Lengths")
    ax[0, 2].set_xlabel("Avg Length (num steps)")
    ax[0, 2].set_ylabel("Normalized Frequency")
    ax[0, 2].legend()

    # set the grid on
    ax[0, 2].grid('on')

    # remove tick marks
    ax[0, 2].xaxis.set_tick_params(size=0)
    ax[0, 2].yaxis.set_tick_params(size=0)

    # change the color of the top and right spines to opaque gray
    ax[0, 2].spines['right'].set_color((.8, .8, .8))
    ax[0, 2].spines['top'].set_color((.8, .8, .8))

    # tweak the axis labels
    xlab = ax[0, 2].xaxis.get_label()
    ylab = ax[0, 2].yaxis.get_label()
    xlab.set_size(10)
    ylab.set_size(10)

    # Plot Hack Probability (Train)

    l = ax[1][0].fill_between(np.array(list(range(len(train_df["hack_probability"].values)))),
                              train_df["hack_probability"].values)

    # set the basic properties
    ax[1][0].set_xlabel("Episode \#")
    ax[1][0].set_ylabel("$\mathbb{P}[Hacked]$")
    ax[1][0].set_title("Likelihood of Successful Hack (Train)")

    ax[1][0].set_ylim(0, 1)

    # set the grid on
    ax[1][0].grid('on')

    # change the fill into a blueish color with opacity .3
    l.set_facecolors([[.5, .5, .8, .3]])

    # change the edge color (bluish and transparentish) and thickness
    l.set_edgecolors([[0, 0, .5, .3]])
    l.set_linewidths([3])

    # remove tick marks
    ax[1][0].xaxis.set_tick_params(size=0)
    ax[1][0].yaxis.set_tick_params(size=0)

    # change the color of the top and right spines to opaque gray
    ax[1][0].spines['right'].set_color((.8, .8, .8))
    ax[1][0].spines['top'].set_color((.8, .8, .8))

    # tweak the axis labels
    xlab = ax[1][0].xaxis.get_label()
    ylab = ax[1][0].yaxis.get_label()
    xlab.set_size(10)
    ylab.set_size(10)

    # Plot Hack Probability (Eval)
    l = ax[1][1].fill_between(np.array(list(range(len(eval_df["hack_probability"].values)))) * eval_step,
                              eval_df["hack_probability"].values)

    # set the basic properties
    ax[1][1].set_xlabel("Episode \#")
    ax[1][1].set_ylabel("$\mathbb{P}[Hacked]$")
    ax[1][1].set_title("Likelihood of Successful Hack (Eval)")

    ax[1][1].set_ylim(0, 1)

    # set the grid on
    ax[1][1].grid('on')

    # change the fill into a blueish color with opacity .3
    l.set_facecolors([[.5, .5, .8, .3]])

    # change the edge color (bluish and transparentish) and thickness
    l.set_edgecolors([[0, 0, .5, .3]])
    l.set_linewidths([3])

    # remove tick marks
    ax[1][1].xaxis.set_tick_params(size=0)
    ax[1][1].yaxis.set_tick_params(size=0)

    # change the color of the top and right spines to opaque gray
    ax[1][1].spines['right'].set_color((.8, .8, .8))
    ax[1][1].spines['top'].set_color((.8, .8, .8))

    # tweak the axis labels
    xlab = ax[1][1].xaxis.get_label()
    ylab = ax[1][1].yaxis.get_label()
    xlab.set_size(10)
    ylab.set_size(10)

    # Plot loss
    ax[1][2].errorbar(np.array(list(range(len(train_df["avg_episode_loss_attacker"].values)))),
                      train_df["avg_episode_loss_attacker"].values, yerr=None, mfc=[.5, .5, .8, .3],
                      mec=[0, 0, .5, .3], ls='-', ecolor='black')

    ax[1][2].set_title("Avg Episode Loss (Attacker)")
    ax[1][2].set_xlabel("Episode \#")
    ax[1][2].set_ylabel("Loss")
    ax[1][2].grid('on')

    # tweak the axis labels
    xlab = ax[1][2].xaxis.get_label()
    ylab = ax[1][2].yaxis.get_label()
    xlab.set_size(10)
    ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax[1][2].spines['right'].set_color((.8, .8, .8))
    ax[1][2].spines['top'].set_color((.8, .8, .8))

    # Plot learning rate
    ax[2][0].errorbar(np.array(list(range(len(train_df["lr_list"].values)))),
                      train_df["lr_list"].values, yerr=None, mfc=[.5, .5, .8, .3],
                      mec=[0, 0, .5, .3], ls='-', ecolor='black')

    ax[2][0].set_title("Learning rate (Eta) (Train)")
    ax[2][0].set_xlabel("Episode \#")
    ax[2][0].set_ylabel("Learning Rate")
    ax[2][0].grid('on')

    # tweak the axis labels
    xlab = ax[2][0].xaxis.get_label()
    ylab = ax[2][0].yaxis.get_label()
    xlab.set_size(10)
    ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax[2][0].spines['right'].set_color((.8, .8, .8))
    ax[2][0].spines['top'].set_color((.8, .8, .8))

    # Plot exploration rate
    ax[2][1].errorbar(np.array(list(range(len(train_df["epsilon_values"].values)))),
                      train_df["epsilon_values"].values, yerr=None, mfc=[.5, .5, .8, .3],
                      mec=[0, 0, .5, .3], ls='-', ecolor='black')

    ax[2][1].set_title("Exploration rate (Epsilon)")
    ax[2][1].set_xlabel("Episode \#")
    ax[2][1].set_ylabel("Epsilon")
    ax[2][1].grid('on')

    # tweak the axis labels
    xlab = ax[2][1].xaxis.get_label()
    ylab = ax[2][1].yaxis.get_label()
    xlab.set_size(10)
    ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax[2][1].spines['right'].set_color((.8, .8, .8))
    ax[2][1].spines['top'].set_color((.8, .8, .8))

    # Plot Attack State Values for Final Trajectory
    l = ax[2][2].fill_between(np.array(list(range(len(a_state_values)))),
                              a_state_values)

    # set the basic properties
    ax[2][2].set_xlabel("Time step (t)")
    ax[2][2].set_ylabel("$V(s_t)$")
    ax[2][2].set_title("Attacker State Values")

    # ax[2][2].set_ylim(0,1)

    # set the grid on
    ax[2][2].grid('on')

    # change the fill into a blueish color with opacity .3
    l.set_facecolors([[.5, .5, .8, .3]])

    # change the edge color (bluish and transparentish) and thickness
    l.set_edgecolors([[0, 0, .5, .3]])
    l.set_linewidths([3])

    # remove tick marks
    ax[2][2].xaxis.set_tick_params(size=0)
    ax[2][2].yaxis.set_tick_params(size=0)

    # change the color of the top and right spines to opaque gray
    ax[2][2].spines['right'].set_color((.8, .8, .8))
    ax[2][2].spines['top'].set_color((.8, .8, .8))

    # tweak the axis labels
    xlab = ax[2][2].xaxis.get_label()
    ylab = ax[2][2].yaxis.get_label()
    xlab.set_size(10)
    ylab.set_size(10)

    fig.tight_layout()
    #plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(file_name + ".png", format="png")
    fig.savefig(file_name + ".pdf", format='pdf', dpi=500, bbox_inches='tight', transparent=True)
    plt.close(fig)


def plot_two_histograms(d_1: np.ndarray, d_2: np.ndarray, title: str = "Test", xlabel: str = "test",
                  ylabel: str = "test", file_name: str = "test.eps",
                   hist1_label = "Train", hist2_label = "Eval",
                  num_bins = 5) -> None:
    """
    Plots two histograms

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
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')

    # let us make a simple graph
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))

    weights_1 = np.ones_like(d_1)/float(len(d_1))
    weights_2 = np.ones_like(d_2)/float(len(d_2))
    plt.hist(d_1, alpha=0.5, bins=num_bins, weights = weights_1, color="#599ad3", label=hist1_label,
             stacked=True)
    plt.hist(d_2, alpha=0.5, bins=num_bins, weights = weights_2, color='#f9a65a', label=hist2_label,
            stacked=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    # set the grid on
    ax.grid('on')

    # remove tick marks
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(size=0)

    # change the color of the top and right spines to opaque gray
    ax.spines['right'].set_color((.8, .8, .8))
    ax.spines['top'].set_color((.8, .8, .8))

    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()

    # xlab.set_style('italic')
    # ylab.set_style('italic')
    xlab.set_size(10)
    ylab.set_size(10)

    fig.tight_layout()
    fig.savefig(file_name + ".png", format="png")
    fig.savefig(file_name + ".pdf", format='pdf', dpi=500, bbox_inches='tight', transparent=True)
    plt.close(fig)

def two_line_plot(x_1: np.ndarray, y_1: np.ndarray, x_2: np.ndarray, y_2: np.ndarray,
                  title: str = "Test", xlabel: str = "test", ylabel: str = "test",
                  file_name: str = "test.eps", xlims: Union[float, float] = None, ylims: Union[float, float] = None,
                  line1_label="Train", line2_label="Eval", legend_loc='upper right') -> None:
    """
    Plots two lines

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
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
    if xlims is None:
        xlims = (min(min(x_1), min(x_2)),
                 max(max(x_1), max(x_2)))
    if ylims is None:
        ylims = (min(min(y_1), min(y_2)),
                 max(max(y_1), max(y_2)))

    ax.errorbar(x_1, y_1, yerr=None, ls='-', color="#599ad3", label=line1_label)
    ax.errorbar(x_2, y_2, yerr=None, ls='--', color='#f9a65a', label=line2_label)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # set the grid on
    ax.grid('on')

    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()

    # xlab.set_style('italic')
    xlab.set_size(10)
    # ylab.set_style('italic')
    ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax.spines['right'].set_color((.8, .8, .8))
    ax.spines['top'].set_color((.8, .8, .8))

    plt.legend(loc=legend_loc)

    fig.tight_layout()
    fig.savefig(file_name + ".png", format="png")
    fig.savefig(file_name + ".pdf", format='pdf', dpi=500, bbox_inches='tight', transparent=True)
    plt.close(fig)


def simple_line_plot(x: np.ndarray, y: np.ndarray, title: str = "Test", xlabel: str = "test", ylabel: str = "test",
                     file_name: str = "test.eps", xlims: Union[float, float] = None, ylims: Union[float, float] = None,
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
    ax.errorbar(x, y, yerr=None, mfc=[.5, .5, .8, .3], mec=[0, 0, .5, .3], ls='-', ecolor='black')

    if smooth:
        smooth = interp1d(x, y)
        x_smooth = np.linspace(min(x), max(x), len(x) // 10)
        ax.errorbar(x_smooth, smooth(x_smooth), yerr=None, color="black", ls='-', ecolor='black')

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # set the grid on
    ax.grid('on')

    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()

    # xlab.set_style('italic')
    xlab.set_size(10)
    # ylab.set_style('italic')
    ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax.spines['right'].set_color((.8, .8, .8))
    ax.spines['top'].set_color((.8, .8, .8))

    if log:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(file_name + ".png", format="png")
    fig.savefig(file_name + ".pdf", format='pdf', dpi=500, bbox_inches='tight', transparent=True)
    plt.close(fig)


def probability_plot(x: np.ndarray, y: np.ndarray, title: str = "Test", xlabel: str = "test", ylabel: str = "test",
                     file_name: str = "test.eps", xlims: Union[float, float] = None,
                     ylims: Union[float, float] = None) -> None:
    """
    Plots a CDF-like probability plot

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
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')

    # let us make a simple graph
    fig = plt.figure(figsize=[8, 3])
    ax = plt.subplot(111)
    l = ax.fill_between(x, y)

    # set the basic properties
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    # set the grid on
    ax.grid('on')

    # change the fill into a blueish color with opacity .3
    l.set_facecolors([[.5, .5, .8, .3]])

    # change the edge color (bluish and transparentish) and thickness
    l.set_edgecolors([[0, 0, .5, .3]])
    l.set_linewidths([3])

    # remove tick marks
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(size=0)

    # change the color of the top and right spines to opaque gray
    ax.spines['right'].set_color((.8, .8, .8))
    ax.spines['top'].set_color((.8, .8, .8))

    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()

    # xlab.set_style('italic')
    xlab.set_size(10)
    # ylab.set_style('italic')
    ylab.set_size(10)

    fig.tight_layout()
    fig.savefig(file_name + ".png", format="png")
    fig.savefig(file_name + ".pdf", format='pdf', dpi=500, bbox_inches='tight', transparent=True)
    plt.close(fig)


def two_line_plot_w_shades(x_1: np.ndarray, y_1: np.ndarray,
                           x_2: np.ndarray, y_2: np.ndarray,
                           stds_1, stds_2,
                           title: str = "Test", xlabel: str = "test", ylabel: str = "test",
                           file_name: str = "test.eps", xlims: Union[float, float] = None,
                           ylims: Union[float, float] = None,
                           line1_label="Train", line2_label="Eval", legend_loc='upper right',
                           markevery_1=5, markevery_2=5) -> None:
    """
    Plot two line plots with shaded error bars

    :param x_1: x data of line 1
    :param y_1: y data of line 1
    :param x_2: x data of line 2
    :param y_2: y data of line 2
    :param stds_1: standard deviations of line 1
    :param stds_2: standard deviations of line 2
    :param title: plot title
    :param xlabel: label on x axis
    :param ylabel: label on y axis
    :param file_name: name of file to save
    :param xlims: limits on x axis
    :param ylims: limits on y axis
    :param line1_label: legend for line 1
    :param line2_label: legend for line 2
    :param legend_loc: location of the legend
    :param markevery_1: marker frequency for line 1
    :param markevery_2: marker frequency for line 2
    :return:
    """
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
    if xlims is None:
        xlims = (min(min(x_1), min(x_2)),
                 max(max(x_1), max(x_2)))
    if ylims is None:
        ylims = (min(min(y_1), min(y_2)),
                 max(max(y_1), max(y_2)))

    ax.plot(x_1, y_1, label=line1_label, marker="s", ls='-', color="#599ad3", markevery=markevery_1)
    ax.fill_between(x_1, y_1 - stds_1, y_1 + stds_1, alpha=0.35, color="#599ad3")

    ax.plot(x_2, y_2, label=line2_label, marker="o", ls='-', color='#f9a65a', markevery=markevery_2)
    ax.fill_between(x_2, y_2 - stds_2, y_2 + stds_2, alpha=0.35, color='#f9a65a')

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # set the grid on
    ax.grid('on')

    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()

    # xlab.set_style('italic')
    xlab.set_size(10)
    # ylab.set_style('italic')
    ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax.spines['right'].set_color((.8, .8, .8))
    ax.spines['top'].set_color((.8, .8, .8))

    plt.legend(loc=legend_loc)

    fig.tight_layout()
    fig.savefig(file_name + ".png", format="png")
    fig.savefig(file_name + ".pdf", format='pdf', dpi=500, bbox_inches='tight', transparent=True)
    plt.close(fig)


def five_line_plot_w_shades(x_1: np.ndarray, y_1: np.ndarray,
                            x_2: np.ndarray, y_2: np.ndarray,
                            x_3: np.ndarray, y_3: np.ndarray,
                            x_4: np.ndarray, y_4: np.ndarray,
                            x_5: np.ndarray, y_5: np.ndarray,
                            stds_1, stds_2, stds_3, stds_4, stds_5,
                            title: str = "Test", xlabel: str = "test", ylabel: str = "test",
                            file_name: str = "test.eps", xlims: Union[float, float] = None,
                            ylims: Union[float, float] = None,
                            line1_label="Train", line2_label="Eval", line3_label="Eval",
                            line4_label="Eval", line5_label="Eval",
                            legend_loc='upper right',
                            markevery_1=5, markevery_2=5, markevery_3=5, markevery_4=5,
                            markevery_5=5) -> None:
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    if xlims is None:
        xlims = (min(min(x_1), min(x_2), min(x_3), min(x_4), min(x_5)),
                 max(max(x_1), max(x_2), max(x_3), max(x_4), max(x_5)))
    if ylims is None:
        ylims = (min(min(y_1), min(y_2), min(y_3), min(y_4), min(y_5)),
                 max(max(y_1), max(y_2), max(y_3), max(y_4), max(y_5)))

    ax.plot(x_1, y_1, label=line1_label, marker="s", ls='-', color="#599ad3", markevery=markevery_1)
    ax.fill_between(x_1, y_1 - stds_1, y_1 + stds_1, alpha=0.35, color="#599ad3")

    ax.plot(x_2, y_2, label=line2_label, marker="o", ls='-', color='#f9a65a', markevery=markevery_2)
    ax.fill_between(x_2, y_2 - stds_2, y_2 + stds_2, alpha=0.35, color='#f9a65a')

    ax.plot(x_3, y_3, label=line3_label, marker="p", ls='-', color="#9e66ab", markevery=markevery_3)
    ax.fill_between(x_3, y_3 - stds_3, y_3 + stds_3, alpha=0.35, color="#9e66ab")

    ax.plot(x_4, y_4, label=line4_label, marker="d", ls='-', color='g', markevery=markevery_4)
    ax.fill_between(x_4, y_4 - stds_4, y_4 + stds_4, alpha=0.35, color='g')

    ax.plot(x_5, y_5, label=line5_label, marker="^", ls='-', color='r', markevery=markevery_5)
    ax.fill_between(x_5, y_5 - stds_5, y_5 + stds_5, alpha=0.35, color='r')

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # set the grid on
    ax.grid('on')

    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()

    # xlab.set_style('italic')
    xlab.set_size(10)
    # ylab.set_style('italic')
    ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax.spines['right'].set_color((.8, .8, .8))
    ax.spines['top'].set_color((.8, .8, .8))

    # plt.legend(loc=legend_loc)

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.05,
                     box.width, box.height * 0.95])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
              fancybox=True, shadow=True, ncol=2)

    fig.tight_layout()
    fig.savefig(file_name + ".png", format="png")
    fig.savefig(file_name + ".pdf", format='pdf', dpi=500, bbox_inches='tight', transparent=True)
    plt.close(fig)


def plot_all_avg_summary_1(x_1_1, y_1_1, x_1_2, y_1_2, x_1_3, y_1_3, x_1_4, y_1_4, x_1_5, y_1_5,
                           std_1_1, std_1_2, std_1_3, std_1_4, std_1_5,
                           line_1_1_label, line_1_2_label, line_1_3_label, line_1_4_label,
                           line_1_5_label, title_1, xlabel_1, ylabel_1,
                           markevery_1_1, markevery_1_2, markevery_1_3, markevery_1_4, markevery_1_5,
                           x_2_1, y_2_1, x_2_2, y_2_2, x_2_3, y_2_3, x_2_4, y_2_4, x_2_5, y_2_5,
                           std_2_1, std_2_2, std_2_3, std_2_4, std_2_5,
                           line_2_1_label, line_2_2_label, line_2_3_label, line_2_4_label,
                           line_2_5_label, title_2, xlabel_2, ylabel_2,
                           markevery_2_1, markevery_2_2, markevery_2_3, markevery_2_4, markevery_2_5,
                           x_3_1, y_3_1, x_3_2, y_3_2, x_3_3, y_3_3, x_3_4, y_3_4, x_3_5, y_3_5,
                           std_3_1, std_3_2, std_3_3, std_3_4, std_3_5,
                           line_3_1_label, line_3_2_label, line_3_3_label, line_3_4_label,
                           line_3_5_label, title_3, xlabel_3, ylabel_3,
                           markevery_3_1, markevery_3_2, markevery_3_3, markevery_3_4, markevery_3_5,
                           x_4_1, y_4_1, x_4_2, y_4_2, x_4_3, y_4_3, x_4_4, y_4_4, x_4_5, y_4_5,
                           std_4_1, std_4_2, std_4_3, std_4_4, std_4_5,
                           line_4_1_label, line_4_2_label, line_4_3_label, line_4_4_label,
                           line_4_5_label, title_4, xlabel_4, ylabel_4,
                           markevery_4_1, markevery_4_2, markevery_4_3, markevery_4_4, markevery_4_5,
                           file_name
                           ):
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 2.75))
    #gs1 = gridspec.GridSpec(1, 4)
    #gs1.update(wspace=0.005, hspace=0.05)

    # Plot avg hack_probability train
    xlims = (min(min(x_1_1), min(x_1_2), min(x_1_3), min(x_1_4), min(x_1_5)),
             max(max(x_1_1), max(x_1_2), max(x_1_3), max(x_1_4), max(x_1_5)))
    ylims = (0, 1)

    ax[0].plot(x_1_1, y_1_1, label=line_1_1_label, marker="s", ls='-', color="#599ad3", markevery=markevery_1_1)
    ax[0].fill_between(x_1_1, y_1_1 - std_1_1, y_1_1 + std_1_1, alpha=0.35, color="#599ad3")

    ax[0].plot(x_1_2, y_1_2, label=line_1_2_label, marker="o", ls='-', color='#f9a65a', markevery=markevery_1_2)
    ax[0].fill_between(x_1_2, y_1_2 - std_1_2, y_1_2 + std_1_2, alpha=0.35, color='#f9a65a')

    ax[0].plot(x_1_3, y_1_3, label=line_1_3_label, marker="p", ls='-', color="#9e66ab", markevery=markevery_1_3)
    ax[0].fill_between(x_1_3, y_1_3 - std_1_3, y_1_3 + std_1_3, alpha=0.35, color="#9e66ab")

    ax[0].plot(x_1_4, y_1_4, label=line_1_4_label, marker="d", ls='-', color='g', markevery=markevery_1_4)
    ax[0].fill_between(x_1_4, y_1_4 - std_1_4, y_1_4 + std_1_4, alpha=0.35, color='g')

    ax[0].plot(x_1_5, y_1_5, label=line_1_5_label, marker="^", ls='-', color='r', markevery=markevery_1_5)
    ax[0].fill_between(x_1_5, y_1_5 - std_1_5, y_1_5 + std_1_5, alpha=0.35, color='r')

    ax[0].set_xlim(xlims)
    ax[0].set_ylim(ylims)

    ax[0].set_title(title_1)
    ax[0].set_xlabel(xlabel_1)
    ax[0].set_ylabel(ylabel_1)
    # set the grid on
    ax[0].grid('on')

    # tweak the axis labels
    xlab = ax[0].xaxis.get_label()
    ylab = ax[0].yaxis.get_label()

    # xlab.set_style('italic')
    #xlab.set_size(8)
    # ylab.set_style('italic')
    #ylab.set_size(8)

    # change the color of the top and right spines to opaque gray
    ax[0].spines['right'].set_color((.8, .8, .8))
    ax[0].spines['top'].set_color((.8, .8, .8))

    # plt.legend(loc=legend_loc)

    # Shrink current axis's height by 10% on the bottom
    # box = ax[0].get_position()
    # ax[0].set_position([box.x0, 0.8*box.y0,
    #                     box.width, box.height * 0.99])
    fig.subplots_adjust(bottom=0.4)

    # Put a legend below current axis
    # ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
    #              fancybox=True, shadow=True, ncol=2)

    # Plot avg hack_probability eval
    xlims = (min(min(x_2_1), min(x_2_2), min(x_2_3), min(x_2_4), min(x_2_5)),
             max(max(x_2_1), max(x_2_2), max(x_2_3), max(x_2_4), max(x_2_5)))
    ylims = (0, 1)

    ax[1].plot(x_2_1, y_2_1, label=line_2_1_label, marker="s", ls='-', color="#599ad3", markevery=markevery_2_1)
    ax[1].fill_between(x_2_1, y_2_1 - std_2_1, y_2_1 + std_2_1, alpha=0.35, color="#599ad3")

    ax[1].plot(x_2_2, y_2_2, label=line_2_2_label, marker="o", ls='-', color='#f9a65a', markevery=markevery_2_2)
    ax[1].fill_between(x_2_2, y_2_2 - std_2_2, y_2_2 + std_2_2, alpha=0.35, color='#f9a65a')

    ax[1].plot(x_2_3, y_2_3, label=line_2_3_label, marker="p", ls='-', color="#9e66ab", markevery=markevery_2_3)
    ax[1].fill_between(x_2_3, y_2_3 - std_2_3, y_2_3 + std_2_3, alpha=0.35, color="#9e66ab")

    ax[1].plot(x_2_4, y_2_4, label=line_2_4_label, marker="d", ls='-', color='g', markevery=markevery_2_4)
    ax[1].fill_between(x_2_4, y_2_4 - std_2_4, y_2_4 + std_2_4, alpha=0.35, color='g')

    ax[1].plot(x_2_5, y_2_5, label=line_2_5_label, marker="^", ls='-', color='r', markevery=markevery_2_5)
    ax[1].fill_between(x_2_5, y_2_5 - std_2_5, y_2_5 + std_2_5, alpha=0.35, color='r')

    ax[1].set_xlim(xlims)
    ax[1].set_ylim(ylims)

    ax[1].set_title(title_2)
    ax[1].set_xlabel(xlabel_2)
    ax[1].set_ylabel(ylabel_2)
    # set the grid on
    ax[1].grid('on')

    # tweak the axis labels
    xlab = ax[1].xaxis.get_label()
    ylab = ax[1].yaxis.get_label()

    # xlab.set_style('italic')
    #xlab.set_size(10)
    # ylab.set_style('italic')
    #ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax[1].spines['right'].set_color((.8, .8, .8))
    ax[1].spines['top'].set_color((.8, .8, .8))

    # plt.legend(loc=legend_loc)

    # Shrink current axis's height by 10% on the bottom
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0 + box.height * 0.03,
                        box.width, box.height * 0.9])

    # Put a legend below current axis
    # ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
    #              fancybox=True, shadow=True, ncol=2)

    # Plot attacker cumulative reward
    xlims = (min(min(x_3_1), min(x_3_2), min(x_3_3), min(x_3_4), min(x_3_5)),
             max(max(x_3_1), max(x_3_2), max(x_3_3), max(x_3_4), max(x_3_5)))
    ylims = (min(min(y_3_1), min(y_3_2), min(y_3_3), min(y_3_4), min(y_3_5)),
             max(max(y_3_1), max(y_3_2), max(y_3_3), max(y_3_4), max(y_3_5)))

    ax[2].plot(x_3_1, y_3_1, label=line_3_1_label, marker="s", ls='-', color="#599ad3", markevery=markevery_3_1)
    ax[2].fill_between(x_3_1, y_3_1 - std_3_1, y_3_1 + std_3_1, alpha=0.35, color="#599ad3")

    ax[2].plot(x_3_2, y_3_2, label=line_3_2_label, marker="o", ls='-', color='#f9a65a', markevery=markevery_3_2)
    ax[2].fill_between(x_3_2, y_3_2 - std_3_2, y_3_2 + std_3_2, alpha=0.35, color='#f9a65a')

    ax[2].plot(x_3_3, y_3_3, label=line_3_3_label, marker="p", ls='-', color="#9e66ab", markevery=markevery_3_3)
    ax[2].fill_between(x_3_3, y_3_3 - std_3_3, y_3_3 + std_3_3, alpha=0.35, color="#9e66ab")

    ax[2].plot(x_3_4, y_3_4, label=line_3_4_label, marker="d", ls='-', color='g', markevery=markevery_3_4)
    ax[2].fill_between(x_3_4, y_3_4 - std_3_4, y_3_4 + std_3_4, alpha=0.35, color='g')

    ax[2].plot(x_3_5, y_3_5, label=line_3_5_label, marker="^", ls='-', color='r', markevery=markevery_3_5)
    ax[2].fill_between(x_3_5, y_3_5 - std_3_5, y_3_5 + std_3_5, alpha=0.35, color='r')

    ax[2].set_xlim(xlims)
    ax[2].set_ylim(ylims)

    ax[2].set_title(title_3)
    ax[2].set_xlabel(xlabel_3)
    ax[2].set_ylabel(ylabel_3)
    # set the grid on
    ax[2].grid('on')

    # tweak the axis labels
    xlab = ax[2].xaxis.get_label()
    ylab = ax[2].yaxis.get_label()

    ax[2].yaxis.labelpad = -5

    # xlab.set_style('italic')
    #xlab.set_size(10)
    # ylab.set_style('italic')
    #ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax[2].spines['right'].set_color((.8, .8, .8))
    ax[2].spines['top'].set_color((.8, .8, .8))

    # plt.legend(loc=legend_loc)

    # Shrink current axis's height by 10% on the bottom
    box = ax[2].get_position()
    ax[2].set_position([box.x0, box.y0 + box.height * 0.03,
                        box.width, box.height * 0.9])

    ax[2].get_yaxis().set_major_formatter(tkr.FuncFormatter(lambda x, p: "${:1.0f}K$".format(x * 1e-3)))

    # Put a legend below current axis
    # ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
    #              fancybox=True, shadow=True, ncol=2)

    #handles, labels = ax.get_legend_handles_labels()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    #fig.legend(handles, labels, loc='upper center')
    ax[2].legend(loc='upper center', bbox_to_anchor=(-0.5, -0.35), fancybox=True, shadow=True, ncol=3)

    # # Plot defender cumulative reward
    # xlims = (min(min(x_4_1), min(x_4_2), min(x_4_3), min(x_4_4), min(x_4_5)),
    #          max(max(x_4_1), max(x_4_2), max(x_4_3), max(x_4_4), max(x_4_5)))
    # ylims = (min(min(y_4_1), min(y_4_2), min(y_4_3), min(y_4_4), min(y_4_5)),
    #          max(max(y_4_1), max(y_4_2), max(y_4_3), max(y_4_4), max(y_4_5)))
    #
    # ax[3].plot(x_4_1, y_4_1, label=line_4_1_label, marker="s", ls='-', color="#599ad3", markevery=markevery_4_1)
    # ax[3].fill_between(x_4_1, y_4_1 - std_4_1, y_4_1 + std_4_1, alpha=0.35, color="#599ad3")
    #
    # ax[3].plot(x_4_2, y_4_2, label=line_4_2_label, marker="o", ls='-', color='#f9a65a', markevery=markevery_4_2)
    # ax[3].fill_between(x_4_2, y_4_2 - std_4_2, y_4_2 + std_4_2, alpha=0.35, color='#f9a65a')
    #
    # ax[3].plot(x_4_3, y_4_3, label=line_4_3_label, marker="p", ls='-', color="#9e66ab", markevery=markevery_4_3)
    # ax[3].fill_between(x_4_3, y_4_3 - std_4_3, y_4_3 + std_4_3, alpha=0.35, color="#9e66ab")
    #
    # ax[3].plot(x_4_4, y_4_4, label=line_4_4_label, marker="d", ls='-', color='g', markevery=markevery_4_4)
    # ax[3].fill_between(x_4_4, y_4_4 - std_4_4, y_4_4 + std_4_4, alpha=0.35, color='g')
    #
    # ax[3].plot(x_4_5, y_4_5, label=line_4_5_label, marker="^", ls='-', color='r', markevery=markevery_4_5)
    # ax[3].fill_between(x_4_5, y_4_5 - std_4_5, y_4_5 + std_4_5, alpha=0.35, color='r')
    #
    # ax[3].set_xlim(xlims)
    # ax[3].set_ylim(ylims)
    #
    # ax[3].set_title(title_4)
    # ax[3].set_xlabel(xlabel_4)
    # ax[3].set_ylabel(ylabel_4)
    # # set the grid on
    # ax[3].grid('on')
    #
    # # tweak the axis labels
    # xlab = ax[3].xaxis.get_label()
    # ylab = ax[3].yaxis.get_label()
    #
    # # xlab.set_style('italic')
    # #xlab.set_size(10)
    # # ylab.set_style('italic')
    # #ylab.set_size(10)
    #
    # # change the color of the top and right spines to opaque gray
    # ax[3].spines['right'].set_color((.8, .8, .8))
    # ax[3].spines['top'].set_color((.8, .8, .8))
    #
    # # plt.legend(loc=legend_loc)
    #
    # # Shrink current axis's height by 10% on the bottom
    # box = ax[3].get_position()
    # ax[3].set_position([box.x0, box.y0 + box.height * 0.03,
    #                     box.width, box.height * 0.9])
    # ax[3].get_yaxis().set_major_formatter(tkr.FuncFormatter(lambda x, p: "${:1.0f}K$".format(x*1e-3)))

    # Put a legend below current axis
    # ax[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
    #              fancybox=True, shadow=True, ncol=2)

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0)
    fig.savefig(file_name + ".png", format="png")
    fig.savefig(file_name + ".pdf", format='pdf', dpi=500, bbox_inches='tight', transparent=True)



def plot_all_avg_summary_2(x_1_1, y_1_1, x_1_2, y_1_2, x_1_3, y_1_3, x_1_4, y_1_4, x_1_5, y_1_5,
                           std_1_1, std_1_2, std_1_3, std_1_4, std_1_5,
                           line_1_1_label, line_1_2_label, line_1_3_label, line_1_4_label,
                           line_1_5_label, title_1, xlabel_1, ylabel_1,
                           markevery_1_1, markevery_1_2, markevery_1_3, markevery_1_4, markevery_1_5,
                           x_2_1, y_2_1, x_2_2, y_2_2, x_2_3, y_2_3, x_2_4, y_2_4, x_2_5, y_2_5,
                           std_2_1, std_2_2, std_2_3, std_2_4, std_2_5,
                           line_2_1_label, line_2_2_label, line_2_3_label, line_2_4_label,
                           line_2_5_label, title_2, xlabel_2, ylabel_2,
                           markevery_2_1, markevery_2_2, markevery_2_3, markevery_2_4, markevery_2_5,
                           x_3_1, y_3_1, x_3_2, y_3_2, x_3_3, y_3_3, x_3_4, y_3_4, x_3_5, y_3_5,
                           std_3_1, std_3_2, std_3_3, std_3_4, std_3_5,
                           line_3_1_label, line_3_2_label, line_3_3_label, line_3_4_label,
                           line_3_5_label, title_3, xlabel_3, ylabel_3,
                           markevery_3_1, markevery_3_2, markevery_3_3, markevery_3_4, markevery_3_5,
                           x_4_1, y_4_1, x_4_2, y_4_2, x_4_3, y_4_3, x_4_4, y_4_4, x_4_5, y_4_5,
                           std_4_1, std_4_2, std_4_3, std_4_4, std_4_5,
                           line_4_1_label, line_4_2_label, line_4_3_label, line_4_4_label,
                           line_4_5_label, title_4, xlabel_4, ylabel_4,
                           markevery_4_1, markevery_4_2, markevery_4_3, markevery_4_4, markevery_4_5,
                           file_name
                           ):
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 2.75))
    #gs1 = gridspec.GridSpec(1, 4)
    #gs1.update(wspace=0.005, hspace=0.05)

    # Plot avg hack_probability train
    xlims = (min(min(x_1_1), min(x_1_2), min(x_1_3), min(x_1_4), min(x_1_5)),
             max(max(x_1_1), max(x_1_2), max(x_1_3), max(x_1_4), max(x_1_5)))
    ylims = (min(min(y_1_1), min(y_1_2), min(y_1_3), min(y_1_4), min(y_1_5)),
             max(max(y_1_1), max(y_1_2), max(y_1_3), max(y_1_4), max(y_1_5)))

    ax[0].plot(x_1_1, y_1_1, label=line_1_1_label, marker="s", ls='-', color="#599ad3", markevery=markevery_1_1)
    ax[0].fill_between(x_1_1, y_1_1 - std_1_1, y_1_1 + std_1_1, alpha=0.35, color="#599ad3")

    ax[0].plot(x_1_2, y_1_2, label=line_1_2_label, marker="o", ls='-', color='#f9a65a', markevery=markevery_1_2)
    ax[0].fill_between(x_1_2, y_1_2 - std_1_2, y_1_2 + std_1_2, alpha=0.35, color='#f9a65a')

    ax[0].plot(x_1_3, y_1_3, label=line_1_3_label, marker="p", ls='-', color="#9e66ab", markevery=markevery_1_3)
    ax[0].fill_between(x_1_3, y_1_3 - std_1_3, y_1_3 + std_1_3, alpha=0.35, color="#9e66ab")

    ax[0].plot(x_1_4, y_1_4, label=line_1_4_label, marker="d", ls='-', color='g', markevery=markevery_1_4)
    ax[0].fill_between(x_1_4, y_1_4 - std_1_4, y_1_4 + std_1_4, alpha=0.35, color='g')

    ax[0].plot(x_1_5, y_1_5, label=line_1_5_label, marker="^", ls='-', color='r', markevery=markevery_1_5)
    ax[0].fill_between(x_1_5, y_1_5 - std_1_5, y_1_5 + std_1_5, alpha=0.35, color='r')

    ax[0].set_xlim(xlims)
    ax[0].set_ylim(ylims)

    ax[0].set_title(title_1)
    ax[0].set_xlabel(xlabel_1)
    ax[0].set_ylabel(ylabel_1)
    # set the grid on
    ax[0].grid('on')

    # tweak the axis labels
    xlab = ax[0].xaxis.get_label()
    ylab = ax[0].yaxis.get_label()

    # xlab.set_style('italic')
    #xlab.set_size(8)
    # ylab.set_style('italic')
    #ylab.set_size(8)

    # change the color of the top and right spines to opaque gray
    ax[0].spines['right'].set_color((.8, .8, .8))
    ax[0].spines['top'].set_color((.8, .8, .8))

    # plt.legend(loc=legend_loc)

    # Shrink current axis's height by 10% on the bottom
    # box = ax[0].get_position()
    # ax[0].set_position([box.x0, 0.8*box.y0,
    #                     box.width, box.height * 0.99])
    fig.subplots_adjust(bottom=0.4)

    # Put a legend below current axis
    # ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
    #              fancybox=True, shadow=True, ncol=2)

    # Plot avg hack_probability eval
    xlims = (min(min(x_2_1), min(x_2_2), min(x_2_3), min(x_2_4), min(x_2_5)),
             max(max(x_2_1), max(x_2_2), max(x_2_3), max(x_2_4), max(x_2_5)))
    ylims = (min(min(y_2_1), min(y_2_2), min(y_2_3), min(y_2_4), min(y_2_5)),
             max(max(y_2_1), max(y_2_2), max(y_2_3), max(y_2_4), max(y_2_5)))

    ax[1].plot(x_2_1, y_2_1, label=line_2_1_label, marker="s", ls='-', color="#599ad3", markevery=markevery_2_1)
    ax[1].fill_between(x_2_1, y_2_1 - std_2_1, y_2_1 + std_2_1, alpha=0.35, color="#599ad3")

    ax[1].plot(x_2_2, y_2_2, label=line_2_2_label, marker="o", ls='-', color='#f9a65a', markevery=markevery_2_2)
    ax[1].fill_between(x_2_2, y_2_2 - std_2_2, y_2_2 + std_2_2, alpha=0.35, color='#f9a65a')

    ax[1].plot(x_2_3, y_2_3, label=line_2_3_label, marker="p", ls='-', color="#9e66ab", markevery=markevery_2_3)
    ax[1].fill_between(x_2_3, y_2_3 - std_2_3, y_2_3 + std_2_3, alpha=0.35, color="#9e66ab")

    ax[1].plot(x_2_4, y_2_4, label=line_2_4_label, marker="d", ls='-', color='g', markevery=markevery_2_4)
    ax[1].fill_between(x_2_4, y_2_4 - std_2_4, y_2_4 + std_2_4, alpha=0.35, color='g')

    ax[1].plot(x_2_5, y_2_5, label=line_2_5_label, marker="^", ls='-', color='r', markevery=markevery_2_5)
    ax[1].fill_between(x_2_5, y_2_5 - std_2_5, y_2_5 + std_2_5, alpha=0.35, color='r')

    ax[1].set_xlim(xlims)
    ax[1].set_ylim(ylims)

    ax[1].set_title(title_2)
    ax[1].set_xlabel(xlabel_2)
    ax[1].set_ylabel(ylabel_2)
    # set the grid on
    ax[1].grid('on')

    # tweak the axis labels
    xlab = ax[1].xaxis.get_label()
    ylab = ax[1].yaxis.get_label()

    # xlab.set_style('italic')
    #xlab.set_size(10)
    # ylab.set_style('italic')
    #ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax[1].spines['right'].set_color((.8, .8, .8))
    ax[1].spines['top'].set_color((.8, .8, .8))

    # plt.legend(loc=legend_loc)

    # Shrink current axis's height by 10% on the bottom
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0 + box.height * 0.03,
                        box.width, box.height * 0.9])

    # Put a legend below current axis
    # ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
    #              fancybox=True, shadow=True, ncol=2)

    # Plot attacker cumulative reward
    xlims = (min(min(x_3_1), min(x_3_2), min(x_3_3), min(x_3_4), min(x_3_5)),
             max(max(x_3_1), max(x_3_2), max(x_3_3), max(x_3_4), max(x_3_5)))
    ylims = (min(min(y_3_1), min(y_3_2), min(y_3_3), min(y_3_4), min(y_3_5)),
             max(max(y_3_1), max(y_3_2), max(y_3_3), max(y_3_4), max(y_3_5)))

    ax[2].plot(x_3_1, y_3_1, label=line_3_1_label, marker="s", ls='-', color="#599ad3", markevery=markevery_3_1)
    ax[2].fill_between(x_3_1, y_3_1 - std_3_1, y_3_1 + std_3_1, alpha=0.35, color="#599ad3")

    ax[2].plot(x_3_2, y_3_2, label=line_3_2_label, marker="o", ls='-', color='#f9a65a', markevery=markevery_3_2)
    ax[2].fill_between(x_3_2, y_3_2 - std_3_2, y_3_2 + std_3_2, alpha=0.35, color='#f9a65a')

    ax[2].plot(x_3_3, y_3_3, label=line_3_3_label, marker="p", ls='-', color="#9e66ab", markevery=markevery_3_3)
    ax[2].fill_between(x_3_3, y_3_3 - std_3_3, y_3_3 + std_3_3, alpha=0.35, color="#9e66ab")

    ax[2].plot(x_3_4, y_3_4, label=line_3_4_label, marker="d", ls='-', color='g', markevery=markevery_3_4)
    ax[2].fill_between(x_3_4, y_3_4 - std_3_4, y_3_4 + std_3_4, alpha=0.35, color='g')

    ax[2].plot(x_3_5, y_3_5, label=line_3_5_label, marker="^", ls='-', color='r', markevery=markevery_3_5)
    ax[2].fill_between(x_3_5, y_3_5 - std_3_5, y_3_5 + std_3_5, alpha=0.35, color='r')

    ax[2].set_xlim(xlims)
    ax[2].set_ylim(ylims)

    ax[2].set_title(title_3)
    ax[2].set_xlabel(xlabel_3)
    ax[2].set_ylabel(ylabel_3)
    # set the grid on
    ax[2].grid('on')

    # tweak the axis labels
    xlab = ax[2].xaxis.get_label()
    ylab = ax[2].yaxis.get_label()

    ax[2].yaxis.labelpad = -5

    # xlab.set_style('italic')
    #xlab.set_size(10)
    # ylab.set_style('italic')
    #ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax[2].spines['right'].set_color((.8, .8, .8))
    ax[2].spines['top'].set_color((.8, .8, .8))

    # plt.legend(loc=legend_loc)

    # Shrink current axis's height by 10% on the bottom
    box = ax[2].get_position()
    ax[2].set_position([box.x0, box.y0 + box.height * 0.03,
                        box.width, box.height * 0.9])

    ax[2].get_yaxis().set_major_formatter(tkr.FuncFormatter(lambda x, p: "${:1.0f}K$".format(x * 1e-3)))

    # Put a legend below current axis
    # ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
    #              fancybox=True, shadow=True, ncol=2)

    #handles, labels = ax.get_legend_handles_labels()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    #fig.legend(handles, labels, loc='upper center')
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=3)

    # # Plot defender cumulative reward
    # xlims = (min(min(x_4_1), min(x_4_2), min(x_4_3), min(x_4_4), min(x_4_5)),
    #          max(max(x_4_1), max(x_4_2), max(x_4_3), max(x_4_4), max(x_4_5)))
    # ylims = (min(min(y_4_1), min(y_4_2), min(y_4_3), min(y_4_4), min(y_4_5)),
    #          max(max(y_4_1), max(y_4_2), max(y_4_3), max(y_4_4), max(y_4_5)))
    #
    # ax[3].plot(x_4_1, y_4_1, label=line_4_1_label, marker="s", ls='-', color="#599ad3", markevery=markevery_4_1)
    # ax[3].fill_between(x_4_1, y_4_1 - std_4_1, y_4_1 + std_4_1, alpha=0.35, color="#599ad3")
    #
    # ax[3].plot(x_4_2, y_4_2, label=line_4_2_label, marker="o", ls='-', color='#f9a65a', markevery=markevery_4_2)
    # ax[3].fill_between(x_4_2, y_4_2 - std_4_2, y_4_2 + std_4_2, alpha=0.35, color='#f9a65a')
    #
    # ax[3].plot(x_4_3, y_4_3, label=line_4_3_label, marker="p", ls='-', color="#9e66ab", markevery=markevery_4_3)
    # ax[3].fill_between(x_4_3, y_4_3 - std_4_3, y_4_3 + std_4_3, alpha=0.35, color="#9e66ab")
    #
    # ax[3].plot(x_4_4, y_4_4, label=line_4_4_label, marker="d", ls='-', color='g', markevery=markevery_4_4)
    # ax[3].fill_between(x_4_4, y_4_4 - std_4_4, y_4_4 + std_4_4, alpha=0.35, color='g')
    #
    # ax[3].plot(x_4_5, y_4_5, label=line_4_5_label, marker="^", ls='-', color='r', markevery=markevery_4_5)
    # ax[3].fill_between(x_4_5, y_4_5 - std_4_5, y_4_5 + std_4_5, alpha=0.35, color='r')
    #
    # ax[3].set_xlim(xlims)
    # ax[3].set_ylim(ylims)
    #
    # ax[3].set_title(title_4)
    # ax[3].set_xlabel(xlabel_4)
    # ax[3].set_ylabel(ylabel_4)
    # # set the grid on
    # ax[3].grid('on')
    #
    # # tweak the axis labels
    # xlab = ax[3].xaxis.get_label()
    # ylab = ax[3].yaxis.get_label()
    #
    # # xlab.set_style('italic')
    # #xlab.set_size(10)
    # # ylab.set_style('italic')
    # #ylab.set_size(10)
    #
    # # change the color of the top and right spines to opaque gray
    # ax[3].spines['right'].set_color((.8, .8, .8))
    # ax[3].spines['top'].set_color((.8, .8, .8))
    #
    # # plt.legend(loc=legend_loc)
    #
    # # Shrink current axis's height by 10% on the bottom
    # box = ax[3].get_position()
    # ax[3].set_position([box.x0, box.y0 + box.height * 0.03,
    #                     box.width, box.height * 0.9])
    # ax[3].get_yaxis().set_major_formatter(tkr.FuncFormatter(lambda x, p: "${:1.0f}K$".format(x*1e-3)))

    # Put a legend below current axis
    # ax[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
    #              fancybox=True, shadow=True, ncol=2)

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0)
    fig.savefig(file_name + ".png", format="png")
    fig.savefig(file_name + ".pdf", format='pdf', dpi=500, bbox_inches='tight', transparent=True)



def plot_all_avg_summary_3(x_1_1_v0, y_1_1_v0, x_1_2_v0, y_1_2_v0, x_1_3_v0, y_1_3_v0, x_1_4_v0, y_1_4_v0,
                           x_1_5_v0, y_1_5_v0,
                           std_1_1_v0, std_1_2_v0, std_1_3_v0, std_1_4_v0, std_1_5_v0,
                           line_1_1_label_v0, line_1_2_label_v0, line_1_3_label_v0, line_1_4_label_v0,
                           line_1_5_label_v0, title_1_v0, xlabel_1_v0, ylabel_1_v0,
                           markevery_1_1_v0, markevery_1_2_v0, markevery_1_3_v0, markevery_1_4_v0, markevery_1_5_v0,
                           x_2_1_v0, y_2_1_v0, x_2_2_v0, y_2_2_v0, x_2_3_v0, y_2_3_v0, x_2_4_v0, y_2_4_v0, x_2_5_v0,
                           y_2_5_v0, std_2_1_v0, std_2_2_v0, std_2_3_v0, std_2_4_v0, std_2_5_v0,
                           line_2_1_label_v0, line_2_2_label_v0, line_2_3_label_v0, line_2_4_label_v0,
                           line_2_5_label_v0, title_2_v0, xlabel_2_v0, ylabel_2_v0,
                           markevery_2_1_v0, markevery_2_2_v0, markevery_2_3_v0, markevery_2_4_v0, markevery_2_5_v0,
                           x_3_1_v0, y_3_1_v0, x_3_2_v0, y_3_2_v0, x_3_3_v0, y_3_3_v0, x_3_4_v0, y_3_4_v0, x_3_5_v0,
                           y_3_5_v0,
                           std_3_1_v0, std_3_2_v0, std_3_3_v0, std_3_4_v0, std_3_5_v0,
                           line_3_1_label_v0, line_3_2_label_v0, line_3_3_label_v0, line_3_4_label_v0,
                           line_3_5_label_v0, title_3_v0, xlabel_3_v0, ylabel_3_v0,
                           markevery_3_1_v0, markevery_3_2_v0, markevery_3_3_v0, markevery_3_4_v0, markevery_3_5_v0,

                           x_1_1_v2, y_1_1_v2, x_1_2_v2, y_1_2_v2, x_1_3_v2, y_1_3_v2, x_1_4_v2, y_1_4_v2,
                           x_1_5_v2, y_1_5_v2,
                           std_1_1_v2, std_1_2_v2, std_1_3_v2, std_1_4_v2, std_1_5_v2,
                           line_1_1_label_v2, line_1_2_label_v2, line_1_3_label_v2, line_1_4_label_v2,
                           line_1_5_label_v2, title_1_v2, xlabel_1_v2, ylabel_1_v2,
                           markevery_1_1_v2, markevery_1_2_v2, markevery_1_3_v2, markevery_1_4_v2, markevery_1_5_v2,
                           x_2_1_v2, y_2_1_v2, x_2_2_v2, y_2_2_v2, x_2_3_v2, y_2_3_v2, x_2_4_v2, y_2_4_v2, x_2_5_v2,
                           y_2_5_v2, std_2_1_v2, std_2_2_v2, std_2_3_v2, std_2_4_v2, std_2_5_v2,
                           line_2_1_label_v2, line_2_2_label_v2, line_2_3_label_v2, line_2_4_label_v2,
                           line_2_5_label_v2, title_2_v2, xlabel_2_v2, ylabel_2_v2,
                           markevery_2_1_v2, markevery_2_2_v2, markevery_2_3_v2, markevery_2_4_v2, markevery_2_5_v2,
                           x_3_1_v2, y_3_1_v2, x_3_2_v2, y_3_2_v2, x_3_3_v2, y_3_3_v2, x_3_4_v2, y_3_4_v2, x_3_5_v2,
                           y_3_5_v2,
                           std_3_1_v2, std_3_2_v2, std_3_3_v2, std_3_4_v2, std_3_5_v2,
                           line_3_1_label_v2, line_3_2_label_v2, line_3_3_label_v2, line_3_4_label_v2,
                           line_3_5_label_v2, title_3_v2, xlabel_3_v2, ylabel_3_v2,
                           markevery_3_1_v2, markevery_3_2_v2, markevery_3_3_v2, markevery_3_4_v2, markevery_3_5_v2,

                           x_1_1_v3, y_1_1_v3, x_1_2_v3, y_1_2_v3, x_1_3_v3, y_1_3_v3, x_1_4_v3, y_1_4_v3,
                           x_1_5_v3, y_1_5_v3,
                           std_1_1_v3, std_1_2_v3, std_1_3_v3, std_1_4_v3, std_1_5_v3,
                           line_1_1_label_v3, line_1_2_label_v3, line_1_3_label_v3, line_1_4_label_v3,
                           line_1_5_label_v3, title_1_v3, xlabel_1_v3, ylabel_1_v3,
                           markevery_1_1_v3, markevery_1_2_v3, markevery_1_3_v3, markevery_1_4_v3, markevery_1_5_v3,
                           x_2_1_v3, y_2_1_v3, x_2_2_v3, y_2_2_v3, x_2_3_v3, y_2_3_v3, x_2_4_v3, y_2_4_v3, x_2_5_v3,
                           y_2_5_v3, std_2_1_v3, std_2_2_v3, std_2_3_v3, std_2_4_v3, std_2_5_v3,
                           line_2_1_label_v3, line_2_2_label_v3, line_2_3_label_v3, line_2_4_label_v3,
                           line_2_5_label_v3, title_2_v3, xlabel_2_v3, ylabel_2_v3,
                           markevery_2_1_v3, markevery_2_2_v3, markevery_2_3_v3, markevery_2_4_v3, markevery_2_5_v3,
                           x_3_1_v3, y_3_1_v3, x_3_2_v3, y_3_2_v3, x_3_3_v3, y_3_3_v3, x_3_4_v3, y_3_4_v3, x_3_5_v3,
                           y_3_5_v3,
                           std_3_1_v3, std_3_2_v3, std_3_3_v3, std_3_4_v3, std_3_5_v3,
                           line_3_1_label_v3, line_3_2_label_v3, line_3_3_label_v3, line_3_4_label_v3,
                           line_3_5_label_v3, title_3_v3, xlabel_3_v3, ylabel_3_v3,
                           markevery_3_1_v3, markevery_3_2_v3, markevery_3_3_v3, markevery_3_4_v3, markevery_3_5_v3,

                           file_name,
                           wspace=0.28
                           ):
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(9, 6.8))
    #gs1 = gridspec.GridSpec(1, 4)
    #gs1.update(wspace=0.005, hspace=0.05)

    # Plot avg hack_probability train
    xlims = (min(min(x_1_1_v0), min(x_1_2_v0), min(x_1_3_v0), min(x_1_4_v0), min(x_1_5_v0)),
             max(max(x_1_1_v0), max(x_1_2_v0), max(x_1_3_v0), max(x_1_4_v0), max(x_1_5_v0)))
    ylims = (0, 1)

    ax[0][0].plot(x_1_1_v0, y_1_1_v0, label=line_1_1_label_v0, marker="s", ls='-', color="#599ad3", markevery=markevery_1_1_v0)
    ax[0][0].fill_between(x_1_1_v0, y_1_1_v0 - std_1_1_v0, y_1_1_v0 + std_1_1_v0, alpha=0.35, color="#599ad3")

    ax[0][0].plot(x_1_2_v0, y_1_2_v0, label=line_1_2_label_v0, marker="o", ls='-', color='#f9a65a', markevery=markevery_1_2_v0)
    ax[0][0].fill_between(x_1_2_v0, y_1_2_v0 - std_1_2_v0, y_1_2_v0 + std_1_2_v0, alpha=0.35, color='#f9a65a')

    ax[0][0].plot(x_1_3_v0, y_1_3_v0, label=line_1_3_label_v0, marker="p", ls='-', color="#9e66ab", markevery=markevery_1_3_v0)
    ax[0][0].fill_between(x_1_3_v0, y_1_3_v0 - std_1_3_v0, y_1_3_v0 + std_1_3_v0, alpha=0.35, color="#9e66ab")

    ax[0][0].plot(x_1_4_v0, y_1_4_v0, label=line_1_4_label_v0, marker="d", ls='-', color='g', markevery=markevery_1_4_v0)
    ax[0][0].fill_between(x_1_4_v0, y_1_4_v0 - std_1_4_v0, y_1_4_v0 + std_1_4_v0, alpha=0.35, color='g')

    ax[0][0].plot(x_1_5_v0, y_1_5_v0, label=line_1_5_label_v0, marker="^", ls='-', color='r', markevery=markevery_1_5_v0)
    ax[0][0].fill_between(x_1_5_v0, y_1_5_v0 - std_1_5_v0, y_1_5_v0 + std_1_5_v0, alpha=0.35, color='r')

    ax[0][0].set_xlim(xlims)
    ax[0][0].set_ylim(ylims)

    ax[0][0].set_title(title_1_v0)
    ax[0][0].set_xlabel(xlabel_1_v0)
    ax[0][0].set_ylabel(ylabel_1_v0)
    # set the grid on
    ax[0][0].grid('on')

    # tweak the axis labels
    xlab = ax[0][0].xaxis.get_label()
    ylab = ax[0][0].yaxis.get_label()

    # xlab.set_style('italic')
    #xlab.set_size(8)
    # ylab.set_style('italic')
    #ylab.set_size(8)

    # change the color of the top and right spines to opaque gray
    ax[0][0].spines['right'].set_color((.8, .8, .8))
    ax[0][0].spines['top'].set_color((.8, .8, .8))

    # plt.legend(loc=legend_loc)

    # Shrink current axis's height by 10% on the bottom
    # box = ax[0].get_position()
    # ax[0].set_position([box.x0, 0.8*box.y0,
    #                     box.width, box.height * 0.99])
    fig.subplots_adjust(bottom=0.4)

    # Put a legend below current axis
    # ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
    #              fancybox=True, shadow=True, ncol=2)

    # Plot avg hack_probability eval
    xlims = (min(min(x_2_1_v0), min(x_2_2_v0), min(x_2_3_v0), min(x_2_4_v0), min(x_2_5_v0)),
             max(max(x_2_1_v0), max(x_2_2_v0), max(x_2_3_v0), max(x_2_4_v0), max(x_2_5_v0)))
    ylims = (0, 1)

    ax[0][1].plot(x_2_1_v0, y_2_1_v0, label=line_2_1_label_v0, marker="s", ls='-', color="#599ad3", markevery=markevery_2_1_v0)
    ax[0][1].fill_between(x_2_1_v0, y_2_1_v0 - std_2_1_v0, y_2_1_v0 + std_2_1_v0, alpha=0.35, color="#599ad3")

    ax[0][1].plot(x_2_2_v0, y_2_2_v0, label=line_2_2_label_v0, marker="o", ls='-', color='#f9a65a', markevery=markevery_2_2_v0)
    ax[0][1].fill_between(x_2_2_v0, y_2_2_v0 - std_2_2_v0, y_2_2_v0 + std_2_2_v0, alpha=0.35, color='#f9a65a')

    ax[0][1].plot(x_2_3_v0, y_2_3_v0, label=line_2_3_label_v0, marker="p", ls='-', color="#9e66ab", markevery=markevery_2_3_v0)
    ax[0][1].fill_between(x_2_3_v0, y_2_3_v0 - std_2_3_v0, y_2_3_v0 + std_2_3_v0, alpha=0.35, color="#9e66ab")

    ax[0][1].plot(x_2_4_v0, y_2_4_v0, label=line_2_4_label_v0, marker="d", ls='-', color='g', markevery=markevery_2_4_v0)
    ax[0][1].fill_between(x_2_4_v0, y_2_4_v0 - std_2_4_v0, y_2_4_v0 + std_2_4_v0, alpha=0.35, color='g')

    ax[0][1].plot(x_2_5_v0, y_2_5_v0, label=line_2_5_label_v0, marker="^", ls='-', color='r', markevery=markevery_2_5_v0)
    ax[0][1].fill_between(x_2_5_v0, y_2_5_v0 - std_2_5_v0, y_2_5_v0 + std_2_5_v0, alpha=0.35, color='r')

    ax[0][1].set_xlim(xlims)
    ax[0][1].set_ylim(ylims)

    ax[0][1].set_title(title_2_v0)
    ax[0][1].set_xlabel(xlabel_2_v0)
    ax[0][1].set_ylabel(ylabel_2_v0)
    # set the grid on
    ax[0][1].grid('on')

    # tweak the axis labels
    xlab = ax[0][1].xaxis.get_label()
    ylab = ax[0][1].yaxis.get_label()

    # xlab.set_style('italic')
    #xlab.set_size(10)
    # ylab.set_style('italic')
    #ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax[0][1].spines['right'].set_color((.8, .8, .8))
    ax[0][1].spines['top'].set_color((.8, .8, .8))

    # plt.legend(loc=legend_loc)

    # Shrink current axis's height by 10% on the bottom
    box = ax[0][1].get_position()
    ax[0][1].set_position([box.x0, box.y0 + box.height * 0.03,
                        box.width, box.height * 0.9])

    # Put a legend below current axis
    # ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
    #              fancybox=True, shadow=True, ncol=2)

    # Plot attacker cumulative reward
    xlims = (min(min(x_3_1_v0), min(x_3_2_v0), min(x_3_3_v0), min(x_3_4_v0), min(x_3_5_v0)),
             max(max(x_3_1_v0), max(x_3_2_v0), max(x_3_3_v0), max(x_3_4_v0), max(x_3_5_v0)))
    ylims = (min(min(y_3_1_v0), min(y_3_2_v0), min(y_3_3_v0), min(y_3_4_v0), min(y_3_5_v0)),
             max(max(y_3_1_v0), max(y_3_2_v0), max(y_3_3_v0), max(y_3_4_v0), max(y_3_5_v0)))

    ax[0][2].plot(x_3_1_v0, y_3_1_v0, label=line_3_1_label_v0, marker="s", ls='-', color="#599ad3", markevery=markevery_3_1_v0)
    ax[0][2].fill_between(x_3_1_v0, y_3_1_v0 - std_3_1_v0, y_3_1_v0 + std_3_1_v0, alpha=0.35, color="#599ad3")

    ax[0][2].plot(x_3_2_v0, y_3_2_v0, label=line_3_2_label_v0, marker="o", ls='-', color='#f9a65a', markevery=markevery_3_2_v0)
    ax[0][2].fill_between(x_3_2_v0, y_3_2_v0 - std_3_2_v0, y_3_2_v0 + std_3_2_v0, alpha=0.35, color='#f9a65a')

    ax[0][2].plot(x_3_3_v0, y_3_3_v0, label=line_3_3_label_v0, marker="p", ls='-', color="#9e66ab", markevery=markevery_3_3_v0)
    ax[0][2].fill_between(x_3_3_v0, y_3_3_v0 - std_3_3_v0, y_3_3_v0 + std_3_3_v0, alpha=0.35, color="#9e66ab")

    ax[0][2].plot(x_3_4_v0, y_3_4_v0, label=line_3_4_label_v0, marker="d", ls='-', color='g', markevery=markevery_3_4_v0)
    ax[0][2].fill_between(x_3_4_v0, y_3_4_v0 - std_3_4_v0, y_3_4_v0 + std_3_4_v0, alpha=0.35, color='g')

    ax[0][2].plot(x_3_5_v0, y_3_5_v0, label=line_3_5_label_v0, marker="^", ls='-', color='r', markevery=markevery_3_5_v0)
    ax[0][2].fill_between(x_3_5_v0, y_3_5_v0 - std_3_5_v0, y_3_5_v0 + std_3_5_v0, alpha=0.35, color='r')

    ax[0][2].set_xlim(xlims)
    ax[0][2].set_ylim(ylims)

    ax[0][2].set_title(title_3_v0)
    ax[0][2].set_xlabel(xlabel_3_v0)
    ax[0][2].set_ylabel(ylabel_3_v0)
    # set the grid on
    ax[0][2].grid('on')

    # tweak the axis labels
    xlab = ax[0][2].xaxis.get_label()
    ylab = ax[0][2].yaxis.get_label()

    ax[0][2].yaxis.labelpad = -5

    # xlab.set_style('italic')
    #xlab.set_size(10)
    # ylab.set_style('italic')
    #ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax[0][2].spines['right'].set_color((.8, .8, .8))
    ax[0][2].spines['top'].set_color((.8, .8, .8))

    # plt.legend(loc=legend_loc)

    # Shrink current axis's height by 10% on the bottom
    box = ax[0][2].get_position()
    ax[0][2].set_position([box.x0, box.y0 + box.height * 0.03,
                        box.width, box.height * 0.9])

    ax[0][2].get_yaxis().set_major_formatter(tkr.FuncFormatter(lambda x, p: "${:1.0f}K$".format(x * 1e-3)))

    # Put a legend below current axis
    # ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
    #              fancybox=True, shadow=True, ncol=2)

    #handles, labels = ax.get_legend_handles_labels()
    #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    #fig.legend(handles, labels, loc='upper center')
    #ax[2].legend(loc='upper center', bbox_to_anchor=(-0.5, -0.35), fancybox=True, shadow=True, ncol=3)


    # V2

    # Plot avg hack_probability train
    xlims = (min(min(x_1_1_v2), min(x_1_2_v2), min(x_1_3_v2), min(x_1_4_v2), min(x_1_5_v2)),
             max(max(x_1_1_v2), max(x_1_2_v2), max(x_1_3_v2), max(x_1_4_v2), max(x_1_5_v2)))
    ylims = (0, 1)

    ax[1][0].plot(x_1_1_v2, y_1_1_v2, label=line_1_1_label_v2, marker="s", ls='-', color="#599ad3",
                  markevery=markevery_1_1_v2)
    ax[1][0].fill_between(x_1_1_v2, y_1_1_v2 - std_1_1_v2, y_1_1_v2 + std_1_1_v2, alpha=0.35, color="#599ad3")

    ax[1][0].plot(x_1_2_v2, y_1_2_v2, label=line_1_2_label_v2, marker="o", ls='-', color='#f9a65a',
                  markevery=markevery_1_2_v2)
    ax[1][0].fill_between(x_1_2_v2, y_1_2_v2 - std_1_2_v2, y_1_2_v2 + std_1_2_v2, alpha=0.35, color='#f9a65a')

    ax[1][0].plot(x_1_3_v2, y_1_3_v2, label=line_1_3_label_v2, marker="p", ls='-', color="#9e66ab",
                  markevery=markevery_1_3_v2)
    ax[1][0].fill_between(x_1_3_v2, y_1_3_v2 - std_1_3_v2, y_1_3_v2 + std_1_3_v2, alpha=0.35, color="#9e66ab")

    ax[1][0].plot(x_1_4_v2, y_1_4_v2, label=line_1_4_label_v2, marker="d", ls='-', color='g',
                  markevery=markevery_1_4_v2)
    ax[1][0].fill_between(x_1_4_v2, y_1_4_v2 - std_1_4_v2, y_1_4_v2 + std_1_4_v2, alpha=0.35, color='g')

    ax[1][0].plot(x_1_5_v2, y_1_5_v2, label=line_1_5_label_v2, marker="^", ls='-', color='r',
                  markevery=markevery_1_5_v2)
    ax[1][0].fill_between(x_1_5_v2, y_1_5_v2 - std_1_5_v2, y_1_5_v2 + std_1_5_v2, alpha=0.35, color='r')

    ax[1][0].set_xlim(xlims)
    ax[1][0].set_ylim(ylims)

    ax[1][0].set_title(title_1_v2)
    ax[1][0].set_xlabel(xlabel_1_v2)
    ax[1][0].set_ylabel(ylabel_1_v2)
    # set the grid on
    ax[1][0].grid('on')

    # tweak the axis labels
    xlab = ax[1][0].xaxis.get_label()
    ylab = ax[1][0].yaxis.get_label()

    # xlab.set_style('italic')
    # xlab.set_size(8)
    # ylab.set_style('italic')
    # ylab.set_size(8)

    # change the color of the top and right spines to opaque gray
    ax[1][0].spines['right'].set_color((.8, .8, .8))
    ax[1][0].spines['top'].set_color((.8, .8, .8))

    # plt.legend(loc=legend_loc)

    # Shrink current axis's height by 10% on the bottom
    # box = ax[0].get_position()
    # ax[0].set_position([box.x0, 0.8*box.y0,
    #                     box.width, box.height * 0.99])
    fig.subplots_adjust(bottom=0.4)

    # Put a legend below current axis
    # ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
    #              fancybox=True, shadow=True, ncol=2)

    # Plot avg hack_probability eval
    xlims = (min(min(x_2_1_v2), min(x_2_2_v2), min(x_2_3_v2), min(x_2_4_v2), min(x_2_5_v2)),
             max(max(x_2_1_v2), max(x_2_2_v2), max(x_2_3_v2), max(x_2_4_v2), max(x_2_5_v2)))
    ylims = (0, 1)

    ax[1][1].plot(x_2_1_v2, y_2_1_v2, label=line_2_1_label_v2, marker="s", ls='-', color="#599ad3",
                  markevery=markevery_2_1_v2)
    ax[1][1].fill_between(x_2_1_v2, y_2_1_v2 - std_2_1_v2, y_2_1_v2 + std_2_1_v2, alpha=0.35, color="#599ad3")

    ax[1][1].plot(x_2_2_v2, y_2_2_v2, label=line_2_2_label_v2, marker="o", ls='-', color='#f9a65a',
                  markevery=markevery_2_2_v2)
    ax[1][1].fill_between(x_2_2_v2, y_2_2_v2 - std_2_2_v2, y_2_2_v2 + std_2_2_v2, alpha=0.35, color='#f9a65a')

    ax[1][1].plot(x_2_3_v2, y_2_3_v2, label=line_2_3_label_v2, marker="p", ls='-', color="#9e66ab",
                  markevery=markevery_2_3_v2)
    ax[1][1].fill_between(x_2_3_v2, y_2_3_v2 - std_2_3_v2, y_2_3_v2 + std_2_3_v2, alpha=0.35, color="#9e66ab")

    ax[1][1].plot(x_2_4_v2, y_2_4_v2, label=line_2_4_label_v2, marker="d", ls='-', color='g',
                  markevery=markevery_2_4_v2)
    ax[1][1].fill_between(x_2_4_v2, y_2_4_v2 - std_2_4_v2, y_2_4_v2 + std_2_4_v2, alpha=0.35, color='g')

    ax[1][1].plot(x_2_5_v2, y_2_5_v2, label=line_2_5_label_v2, marker="^", ls='-', color='r',
                  markevery=markevery_2_5_v2)
    ax[1][1].fill_between(x_2_5_v2, y_2_5_v2 - std_2_5_v2, y_2_5_v2 + std_2_5_v2, alpha=0.35, color='r')

    ax[1][1].set_xlim(xlims)
    ax[1][1].set_ylim(ylims)

    ax[1][1].set_title(title_2_v2)
    ax[1][1].set_xlabel(xlabel_2_v2)
    ax[1][1].set_ylabel(ylabel_2_v2)
    # set the grid on
    ax[1][1].grid('on')

    # tweak the axis labels
    xlab = ax[1][1].xaxis.get_label()
    ylab = ax[1][1].yaxis.get_label()

    # xlab.set_style('italic')
    # xlab.set_size(10)
    # ylab.set_style('italic')
    # ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax[1][1].spines['right'].set_color((.8, .8, .8))
    ax[1][1].spines['top'].set_color((.8, .8, .8))

    # plt.legend(loc=legend_loc)

    # Shrink current axis's height by 10% on the bottom
    box = ax[1][1].get_position()
    ax[1][1].set_position([box.x0, box.y0 + box.height * 0.03,
                           box.width, box.height * 0.9])

    # Put a legend below current axis
    # ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
    #              fancybox=True, shadow=True, ncol=2)

    # Plot attacker cumulative reward
    xlims = (min(min(x_3_1_v2), min(x_3_2_v2), min(x_3_3_v2), min(x_3_4_v2), min(x_3_5_v2)),
             max(max(x_3_1_v2), max(x_3_2_v2), max(x_3_3_v2), max(x_3_4_v2), max(x_3_5_v2)))
    ylims = (min(min(y_3_1_v2), min(y_3_2_v2), min(y_3_3_v2), min(y_3_4_v2), min(y_3_5_v2)),
             max(max(y_3_1_v2), max(y_3_2_v2), max(y_3_3_v2), max(y_3_4_v2), max(y_3_5_v2)))

    ax[1][2].plot(x_3_1_v2, y_3_1_v2, label=line_3_1_label_v2, marker="s", ls='-', color="#599ad3",
                  markevery=markevery_3_1_v2)
    ax[1][2].fill_between(x_3_1_v2, y_3_1_v2 - std_3_1_v2, y_3_1_v2 + std_3_1_v2, alpha=0.35, color="#599ad3")

    ax[1][2].plot(x_3_2_v2, y_3_2_v2, label=line_3_2_label_v2, marker="o", ls='-', color='#f9a65a',
                  markevery=markevery_3_2_v2)
    ax[1][2].fill_between(x_3_2_v2, y_3_2_v2 - std_3_2_v2, y_3_2_v2 + std_3_2_v2, alpha=0.35, color='#f9a65a')

    ax[1][2].plot(x_3_3_v2, y_3_3_v2, label=line_3_3_label_v2, marker="p", ls='-', color="#9e66ab",
                  markevery=markevery_3_3_v2)
    ax[1][2].fill_between(x_3_3_v2, y_3_3_v2 - std_3_3_v2, y_3_3_v2 + std_3_3_v2, alpha=0.35, color="#9e66ab")

    ax[1][2].plot(x_3_4_v2, y_3_4_v2, label=line_3_4_label_v2, marker="d", ls='-', color='g',
                  markevery=markevery_3_4_v2)
    ax[1][2].fill_between(x_3_4_v2, y_3_4_v2 - std_3_4_v2, y_3_4_v2 + std_3_4_v2, alpha=0.35, color='g')

    ax[1][2].plot(x_3_5_v2, y_3_5_v2, label=line_3_5_label_v2, marker="^", ls='-', color='r',
                  markevery=markevery_3_5_v2)
    ax[1][2].fill_between(x_3_5_v2, y_3_5_v2 - std_3_5_v2, y_3_5_v2 + std_3_5_v2, alpha=0.35, color='r')

    ax[1][2].set_xlim(xlims)
    ax[1][2].set_ylim(ylims)

    ax[1][2].set_title(title_3_v2)
    ax[1][2].set_xlabel(xlabel_3_v2)
    ax[1][2].set_ylabel(ylabel_3_v2)
    # set the grid on
    ax[1][2].grid('on')

    # tweak the axis labels
    xlab = ax[1][2].xaxis.get_label()
    ylab = ax[1][2].yaxis.get_label()

    ax[1][2].yaxis.labelpad = -5

    # xlab.set_style('italic')
    # xlab.set_size(10)
    # ylab.set_style('italic')
    # ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax[1][2].spines['right'].set_color((.8, .8, .8))
    ax[1][2].spines['top'].set_color((.8, .8, .8))

    # plt.legend(loc=legend_loc)

    # Shrink current axis's height by 10% on the bottom
    box = ax[1][2].get_position()
    ax[1][2].set_position([box.x0, box.y0 + box.height * 0.03,
                           box.width, box.height * 0.9])

    ax[1][2].get_yaxis().set_major_formatter(tkr.FuncFormatter(lambda x, p: "${:1.0f}K$".format(x * 1e-3)))


    # V3

    # Plot avg hack_probability train
    xlims = (min(min(x_1_1_v3), min(x_1_2_v3), min(x_1_3_v3), min(x_1_4_v3), min(x_1_5_v3)),
             max(max(x_1_1_v3), max(x_1_2_v3), max(x_1_3_v3), max(x_1_4_v3), max(x_1_5_v3)))
    ylims = (0, 1)

    ax[2][0].plot(x_1_1_v3, y_1_1_v3, label=line_1_1_label_v3, marker="s", ls='-', color="#599ad3",
                  markevery=markevery_1_1_v3)
    ax[2][0].fill_between(x_1_1_v3, y_1_1_v3 - std_1_1_v3, y_1_1_v3 + std_1_1_v3, alpha=0.35, color="#599ad3")

    ax[2][0].plot(x_1_2_v3, y_1_2_v3, label=line_1_2_label_v3, marker="o", ls='-', color='#f9a65a',
                  markevery=markevery_1_2_v3)
    ax[2][0].fill_between(x_1_2_v3, y_1_2_v3 - std_1_2_v3, y_1_2_v3 + std_1_2_v3, alpha=0.35, color='#f9a65a')

    ax[2][0].plot(x_1_3_v3, y_1_3_v3, label=line_1_3_label_v3, marker="p", ls='-', color="#9e66ab",
                  markevery=markevery_1_3_v3)
    ax[2][0].fill_between(x_1_3_v3, y_1_3_v3 - std_1_3_v3, y_1_3_v3 + std_1_3_v3, alpha=0.35, color="#9e66ab")

    ax[2][0].plot(x_1_4_v3, y_1_4_v3, label=line_1_4_label_v3, marker="d", ls='-', color='g',
                  markevery=markevery_1_4_v3)
    ax[2][0].fill_between(x_1_4_v3, y_1_4_v3 - std_1_4_v3, y_1_4_v3 + std_1_4_v3, alpha=0.35, color='g')

    ax[2][0].plot(x_1_5_v3, y_1_5_v3, label=line_1_5_label_v3, marker="^", ls='-', color='r',
                  markevery=markevery_1_5_v3)
    ax[2][0].fill_between(x_1_5_v3, y_1_5_v3 - std_1_5_v3, y_1_5_v3 + std_1_5_v3, alpha=0.35, color='r')

    ax[2][0].set_xlim(xlims)
    ax[2][0].set_ylim(ylims)

    ax[2][0].set_title(title_1_v3)
    ax[2][0].set_xlabel(xlabel_1_v3)
    ax[2][0].set_ylabel(ylabel_1_v3)
    # set the grid on
    ax[2][0].grid('on')

    # tweak the axis labels
    xlab = ax[2][0].xaxis.get_label()
    ylab = ax[2][0].yaxis.get_label()

    # xlab.set_style('italic')
    # xlab.set_size(8)
    # ylab.set_style('italic')
    # ylab.set_size(8)

    # change the color of the top and right spines to opaque gray
    ax[2][0].spines['right'].set_color((.8, .8, .8))
    ax[2][0].spines['top'].set_color((.8, .8, .8))

    # plt.legend(loc=legend_loc)

    # Shrink current axis's height by 10% on the bottom
    # box = ax[0].get_position()
    # ax[0].set_position([box.x0, 0.8*box.y0,
    #                     box.width, box.height * 0.99])
    fig.subplots_adjust(bottom=0.4)

    # Put a legend below current axis
    # ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
    #              fancybox=True, shadow=True, ncol=2)

    # Plot avg hack_probability eval
    xlims = (min(min(x_2_1_v3), min(x_2_2_v3), min(x_2_3_v3), min(x_2_4_v3), min(x_2_5_v3)),
             max(max(x_2_1_v3), max(x_2_2_v3), max(x_2_3_v3), max(x_2_4_v3), max(x_2_5_v3)))
    ylims = (0, 1)

    ax[2][1].plot(x_2_1_v3, y_2_1_v3, label=line_2_1_label_v3, marker="s", ls='-', color="#599ad3",
                  markevery=markevery_2_1_v3)
    ax[2][1].fill_between(x_2_1_v3, y_2_1_v3 - std_2_1_v3, y_2_1_v3 + std_2_1_v3, alpha=0.35, color="#599ad3")

    ax[2][1].plot(x_2_2_v3, y_2_2_v3, label=line_2_2_label_v3, marker="o", ls='-', color='#f9a65a',
                  markevery=markevery_2_2_v3)
    ax[2][1].fill_between(x_2_2_v3, y_2_2_v3 - std_2_2_v3, y_2_2_v3 + std_2_2_v3, alpha=0.35, color='#f9a65a')

    ax[2][1].plot(x_2_3_v3, y_2_3_v3, label=line_2_3_label_v3, marker="p", ls='-', color="#9e66ab",
                  markevery=markevery_2_3_v3)
    ax[2][1].fill_between(x_2_3_v3, y_2_3_v3 - std_2_3_v3, y_2_3_v3 + std_2_3_v3, alpha=0.35, color="#9e66ab")

    ax[2][1].plot(x_2_4_v3, y_2_4_v3, label=line_2_4_label_v3, marker="d", ls='-', color='g',
                  markevery=markevery_2_4_v3)
    ax[2][1].fill_between(x_2_4_v3, y_2_4_v3 - std_2_4_v3, y_2_4_v3 + std_2_4_v3, alpha=0.35, color='g')

    ax[2][1].plot(x_2_5_v3, y_2_5_v3, label=line_2_5_label_v3, marker="^", ls='-', color='r',
                  markevery=markevery_2_5_v3)
    ax[2][1].fill_between(x_2_5_v3, y_2_5_v3 - std_2_5_v3, y_2_5_v3 + std_2_5_v3, alpha=0.35, color='r')

    ax[2][1].set_xlim(xlims)
    ax[2][1].set_ylim(ylims)

    ax[2][1].set_title(title_2_v3)
    ax[2][1].set_xlabel(xlabel_2_v3)
    ax[2][1].set_ylabel(ylabel_2_v3)
    # set the grid on
    ax[2][1].grid('on')

    # tweak the axis labels
    xlab = ax[2][1].xaxis.get_label()
    ylab = ax[2][1].yaxis.get_label()

    # xlab.set_style('italic')
    # xlab.set_size(10)
    # ylab.set_style('italic')
    # ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax[2][1].spines['right'].set_color((.8, .8, .8))
    ax[2][1].spines['top'].set_color((.8, .8, .8))

    # plt.legend(loc=legend_loc)

    # Shrink current axis's height by 10% on the bottom
    box = ax[2][1].get_position()
    ax[2][1].set_position([box.x0, box.y0 + box.height * 0.03,
                           box.width, box.height * 0.9])

    # Put a legend below current axis
    # ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
    #              fancybox=True, shadow=True, ncol=2)

    # Plot attacker cumulative reward
    xlims = (min(min(x_3_1_v3), min(x_3_2_v3), min(x_3_3_v3), min(x_3_4_v3), min(x_3_5_v3)),
             max(max(x_3_1_v3), max(x_3_2_v3), max(x_3_3_v3), max(x_3_4_v3), max(x_3_5_v3)))
    ylims = (min(min(y_3_1_v3), min(y_3_2_v3), min(y_3_3_v3), min(y_3_4_v3), min(y_3_5_v3)),
             max(max(y_3_1_v3), max(y_3_2_v3), max(y_3_3_v3), max(y_3_4_v3), max(y_3_5_v3)))

    ax[2][2].plot(x_3_1_v3, y_3_1_v3, label=line_3_1_label_v3, marker="s", ls='-', color="#599ad3",
                  markevery=markevery_3_1_v3)
    ax[2][2].fill_between(x_3_1_v3, y_3_1_v3 - std_3_1_v3, y_3_1_v3 + std_3_1_v3, alpha=0.35, color="#599ad3")

    ax[2][2].plot(x_3_2_v3, y_3_2_v3, label=line_3_2_label_v3, marker="o", ls='-', color='#f9a65a',
                  markevery=markevery_3_2_v3)
    ax[2][2].fill_between(x_3_2_v3, y_3_2_v3 - std_3_2_v3, y_3_2_v3 + std_3_2_v3, alpha=0.35, color='#f9a65a')

    ax[2][2].plot(x_3_3_v3, y_3_3_v3, label=line_3_3_label_v3, marker="p", ls='-', color="#9e66ab",
                  markevery=markevery_3_3_v3)
    ax[2][2].fill_between(x_3_3_v3, y_3_3_v3 - std_3_3_v3, y_3_3_v3 + std_3_3_v3, alpha=0.35, color="#9e66ab")

    ax[2][2].plot(x_3_4_v3, y_3_4_v3, label=line_3_4_label_v3, marker="d", ls='-', color='g',
                  markevery=markevery_3_4_v3)
    ax[2][2].fill_between(x_3_4_v3, y_3_4_v3 - std_3_4_v3, y_3_4_v3 + std_3_4_v3, alpha=0.35, color='g')

    ax[2][2].plot(x_3_5_v3, y_3_5_v3, label=line_3_5_label_v3, marker="^", ls='-', color='r',
                  markevery=markevery_3_5_v3)
    ax[2][2].fill_between(x_3_5_v3, y_3_5_v3 - std_3_5_v3, y_3_5_v3 + std_3_5_v3, alpha=0.35, color='r')

    ax[2][2].set_xlim(xlims)
    ax[2][2].set_ylim(ylims)

    ax[2][2].set_title(title_3_v3)
    ax[2][2].set_xlabel(xlabel_3_v3)
    ax[2][2].set_ylabel(ylabel_3_v3)
    # set the grid on
    ax[2][2].grid('on')

    # tweak the axis labels
    xlab = ax[2][2].xaxis.get_label()
    ylab = ax[2][2].yaxis.get_label()

    ax[2][2].yaxis.labelpad = -5

    # xlab.set_style('italic')
    # xlab.set_size(10)
    # ylab.set_style('italic')
    # ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax[2][2].spines['right'].set_color((.8, .8, .8))
    ax[2][2].spines['top'].set_color((.8, .8, .8))

    # plt.legend(loc=legend_loc)

    # Shrink current axis's height by 10% on the bottom
    box = ax[2][2].get_position()
    ax[2][2].set_position([box.x0, box.y0 + box.height * 0.03,
                           box.width, box.height * 0.9])

    ax[2][2].get_yaxis().set_major_formatter(tkr.FuncFormatter(lambda x, p: "${:1.0f}K$".format(x * 1e-3)))

    #handles, labels = ax.get_legend_handles_labels()
    lines_labels = [ax[2][2].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    #ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    fig.legend(lines, labels, loc=(0.16, 0.01), ncol=2, borderaxespad=0.)

    #ax[2][2].legend(loc='lower center', bbox_to_anchor=(5, -1), fancybox=True, shadow=True, ncol=3)
    #fig.legend(loc='upper center', bbox_to_anchor=(-0.7, -1), fancybox=True, shadow=True, ncol=3)
    fig.tight_layout()
    plt.subplots_adjust(wspace=wspace, hspace=0.47)
    fig.subplots_adjust(bottom=0.17)
    fig.savefig(file_name + ".png", format="png", bbox_inches='tight')
    fig.savefig(file_name + ".pdf", format='pdf', dpi=500, bbox_inches='tight', transparent=True)


def plot_all_avg_summary_4(x_1_1_v0, y_1_1_v0, x_1_2_v0, y_1_2_v0, x_1_3_v0, y_1_3_v0, x_1_4_v0, y_1_4_v0,
                           x_1_5_v0, y_1_5_v0, x_1_6_v0, y_1_6_v0,
                           std_1_1_v0, std_1_2_v0, std_1_3_v0, std_1_4_v0, std_1_5_v0, std_1_6_v0,
                           line_1_1_label_v0, line_1_2_label_v0, line_1_3_label_v0, line_1_4_label_v0,
                           line_1_5_label_v0, line_1_6_label_v0, title_1_v0, xlabel_1_v0, ylabel_1_v0,
                           markevery_1_1_v0, markevery_1_2_v0, markevery_1_3_v0, markevery_1_4_v0, markevery_1_5_v0,
                           markevery_1_6_v0,

                           x_1_1_v1, y_1_1_v1, x_1_2_v1, y_1_2_v1, x_1_3_v1, y_1_3_v1, x_1_4_v1, y_1_4_v1,
                           x_1_5_v1, y_1_5_v1, x_1_6_v1, y_1_6_v1,
                           std_1_1_v1, std_1_2_v1, std_1_3_v1, std_1_4_v1, std_1_5_v1, std_1_6_v1,
                           line_1_1_label_v1, line_1_2_label_v1, line_1_3_label_v1, line_1_4_label_v1,
                           line_1_5_label_v1, line_1_6_label_v1, title_1_v1, xlabel_1_v1, ylabel_1_v1,
                           markevery_1_1_v1, markevery_1_2_v1, markevery_1_3_v1, markevery_1_4_v1, markevery_1_5_v1,
                           markevery_1_6_v1,

                           x_1_1_v2, y_1_1_v2, x_1_2_v2, y_1_2_v2, x_1_3_v2, y_1_3_v2, x_1_4_v2, y_1_4_v2,
                           x_1_5_v2, y_1_5_v2, x_1_6_v2, y_1_6_v2,
                           std_1_1_v2, std_1_2_v2, std_1_3_v2, std_1_4_v2, std_1_5_v2, std_1_6_v2,
                           line_1_1_label_v2, line_1_2_label_v2, line_1_3_label_v2, line_1_4_label_v2,
                           line_1_5_label_v2, line_1_6_label_v2, title_1_v2, xlabel_1_v2, ylabel_1_v2,
                           markevery_1_1_v2, markevery_1_2_v2, markevery_1_3_v2, markevery_1_4_v2, markevery_1_5_v2,
                           markevery_1_6_v2,

                           x_1_1_v3, y_1_1_v3, x_1_2_v3, y_1_2_v3, x_1_3_v3, y_1_3_v3, x_1_4_v3, y_1_4_v3,
                           x_1_5_v3, y_1_5_v3, x_1_6_v3, y_1_6_v3,
                           std_1_1_v3, std_1_2_v3, std_1_3_v3, std_1_4_v3, std_1_5_v3, std_1_6_v3,
                           line_1_1_label_v3, line_1_2_label_v3, line_1_3_label_v3, line_1_4_label_v3,
                           line_1_5_label_v3, line_1_6_label_v3, title_1_v3, xlabel_1_v3, ylabel_1_v3,
                           markevery_1_1_v3, markevery_1_2_v3, markevery_1_3_v3, markevery_1_4_v3, markevery_1_5_v3,
                           markevery_1_6_v3,

                           x_1_1_v4, y_1_1_v4, x_1_2_v4, y_1_2_v4, x_1_3_v4, y_1_3_v4, x_1_4_v4, y_1_4_v4,
                           x_1_5_v4, y_1_5_v4, x_1_6_v4, y_1_6_v4,
                           std_1_1_v4, std_1_2_v4, std_1_3_v4, std_1_4_v4, std_1_5_v4, std_1_6_v4,
                           line_1_1_label_v4, line_1_2_label_v4, line_1_3_label_v4, line_1_4_label_v4,
                           line_1_5_label_v4, line_1_6_label_v4, title_1_v4, xlabel_1_v4, ylabel_1_v4,
                           markevery_1_1_v4, markevery_1_2_v4, markevery_1_3_v4, markevery_1_4_v4, markevery_1_5_v4,
                           markevery_1_6_v4,

                           file_name,
                           wspace=0.28
                           ):
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')
    plt.rcParams.update({'font.size': 6})
    fig, ax = plt.subplots(nrows=5, ncols=3, figsize=(8, 7))
    #gs1 = gridspec.GridSpec(1, 4)
    #gs1.update(wspace=0.005, hspace=0.05)

    # TabQ vs TabQ
    # Plot avg hack_probability
    xlims = (min(min(x_1_1_v0), min(x_1_4_v0)),
             max(max(x_1_1_v0), max(x_1_4_v0)))
    ylims = (max(min(min(y_1_1_v0 - std_1_1_v0), min(y_1_4_v0 - std_1_4_v0)), 0),
             max(max(y_1_1_v0 + std_1_1_v0), max(y_1_4_v0 + std_1_4_v0)))

    ax[0][0].plot(x_1_1_v0, y_1_1_v0, label=line_1_1_label_v0, marker="s", ls='-', color="#599ad3", markevery=markevery_1_1_v0)
    ax[0][0].fill_between(x_1_1_v0, y_1_1_v0 - std_1_1_v0, y_1_1_v0 + std_1_1_v0, alpha=0.35, color="#599ad3")

    ax[0][0].plot(x_1_4_v0, y_1_4_v0, label=line_1_4_label_v0, marker="o", ls='-', color='#f9a65a', markevery=markevery_1_4_v0)
    ax[0][0].fill_between(x_1_4_v0, y_1_4_v0 - std_1_4_v0, y_1_4_v0 + std_1_4_v0, alpha=0.35, color='#f9a65a')

    ax[0][0].set_xlim(xlims)
    ax[0][0].set_ylim(ylims)

    ax[0][0].set_title(title_1_v0 + " v0")
    ax[0][0].set_xlabel(xlabel_1_v0)
    ax[0][0].set_ylabel(ylabel_1_v0)
    # set the grid on
    ax[0][0].grid('on')

    # tweak the axis labels
    xlab = ax[0][0].xaxis.get_label()
    ylab = ax[0][0].yaxis.get_label()

    # change the color of the top and right spines to opaque gray
    ax[0][0].spines['right'].set_color((.8, .8, .8))
    ax[0][0].spines['top'].set_color((.8, .8, .8))

    # # Plot avg hack_probability
    xlims = (min(min(x_1_2_v0), min(x_1_5_v0)),
             max(max(x_1_2_v0), max(x_1_5_v0)))
    ylims = (max(min(min(y_1_2_v0 - std_1_2_v0), min(y_1_5_v0 - std_1_5_v0)), 0),
             max(max(y_1_2_v0 + std_1_2_v0), max(y_1_5_v0 + std_1_5_v0)))

    ax[0][1].plot(x_1_2_v0, y_1_2_v0, label=line_1_2_label_v0, marker="s", ls='-', color="#599ad3", markevery=markevery_1_2_v0)
    ax[0][1].fill_between(x_1_2_v0, y_1_2_v0 - std_1_2_v0, y_1_2_v0 + std_1_2_v0, alpha=0.35, color="#599ad3")

    ax[0][1].plot(x_1_5_v0, y_1_5_v0, label=line_1_5_label_v0, marker="o", ls='-', color='#f9a65a', markevery=markevery_1_5_v0)
    ax[0][1].fill_between(x_1_5_v0, y_1_5_v0 - std_1_5_v0, y_1_5_v0 + std_1_5_v0, alpha=0.35, color='#f9a65a')

    ax[0][1].set_xlim(xlims)
    ax[0][1].set_ylim(ylims)

    ax[0][1].set_title(title_1_v0 + " v1")
    ax[0][1].set_xlabel(xlabel_1_v0)
    ax[0][1].set_ylabel(ylabel_1_v0)
    # set the grid on
    ax[0][1].grid('on')

    # tweak the axis labels
    xlab = ax[0][1].xaxis.get_label()
    ylab = ax[0][1].yaxis.get_label()

    # change the color of the top and right spines to opaque gray
    ax[0][1].spines['right'].set_color((.8, .8, .8))
    ax[0][1].spines['top'].set_color((.8, .8, .8))

    # Plot
    xlims = (min(min(x_1_3_v0), min(x_1_6_v0)),
             max(max(x_1_3_v0), max(x_1_6_v0)))
    ylims = (min(min(y_1_3_v0 - std_1_3_v0), min(y_1_6_v0 - std_1_6_v0)),
             max(max(y_1_3_v0 + std_1_3_v0), max(y_1_6_v0 + std_1_6_v0)))


    ax[0][2].plot(x_1_3_v0, y_1_3_v0, label=line_1_3_label_v0, marker="s", ls='-', color="#599ad3", markevery=markevery_1_3_v0)
    ax[0][2].fill_between(x_1_3_v0, y_1_3_v0 - std_1_3_v0, y_1_3_v0 + std_1_3_v0, alpha=0.35, color="#599ad3")

    ax[0][2].plot(x_1_6_v0, y_1_6_v0, label=line_1_6_label_v0, marker="o", ls='-', color='#f9a65a', markevery=markevery_1_4_v0)
    ax[0][2].fill_between(x_1_6_v0, y_1_6_v0 - std_1_6_v0, y_1_6_v0 + std_1_6_v0, alpha=0.35, color='#f9a65a')

    ax[0][2].set_xlim(xlims)
    ax[0][2].set_ylim(ylims)

    ax[0][2].set_title(title_1_v0 + " v2")
    ax[0][2].set_xlabel(xlabel_1_v0)
    ax[0][2].set_ylabel(ylabel_1_v0)
    # set the grid on
    ax[0][2].grid('on')

    # tweak the axis labels
    xlab = ax[0][2].xaxis.get_label()
    ylab = ax[0][2].yaxis.get_label()

    # change the color of the top and right spines to opaque gray
    ax[0][2].spines['right'].set_color((.8, .8, .8))
    ax[0][2].spines['top'].set_color((.8, .8, .8))

    # MaxA vs TabQ
    # Plot avg hack_probability
    xlims = (min(min(x_1_1_v1), min(x_1_4_v1)),
             max(max(x_1_1_v1), max(x_1_4_v1)))
    ylims = (max(min(min(y_1_1_v1 - std_1_1_v1), min(y_1_4_v1 - std_1_4_v1)), 0),
             max(max(y_1_1_v1 + std_1_1_v1), max(y_1_4_v1 + std_1_4_v1)))

    ax[1][0].plot(x_1_1_v1, y_1_1_v1, label=line_1_1_label_v1, marker="s", ls='-', color="#599ad3",
                  markevery=markevery_1_1_v1)
    ax[1][0].fill_between(x_1_1_v1, y_1_1_v1 - std_1_1_v1, y_1_1_v1 + std_1_1_v1, alpha=0.35, color="#599ad3")

    ax[1][0].plot(x_1_4_v1, y_1_4_v1, label=line_1_4_label_v1, marker="o", ls='-', color='#f9a65a',
                  markevery=markevery_1_4_v1)
    ax[1][0].fill_between(x_1_4_v1, y_1_4_v1 - std_1_4_v1, y_1_4_v1 + std_1_4_v1, alpha=0.35, color='#f9a65a')

    ax[1][0].set_xlim(xlims)
    ax[1][0].set_ylim(ylims)

    ax[1][0].set_title(title_1_v1 + " v0")
    ax[1][0].set_xlabel(xlabel_1_v1)
    ax[1][0].set_ylabel(ylabel_1_v1)
    # set the grid on
    ax[1][0].grid('on')

    # tweak the axis labels
    xlab = ax[1][0].xaxis.get_label()
    ylab = ax[1][0].yaxis.get_label()

    # change the color of the top and right spines to opaque gray
    ax[1][0].spines['right'].set_color((.8, .8, .8))
    ax[1][0].spines['top'].set_color((.8, .8, .8))

    # # Plot avg hack_probability
    xlims = (min(min(x_1_2_v1), min(x_1_5_v1)),
             max(max(x_1_2_v1), max(x_1_5_v1)))
    ylims = (max(min(min(y_1_2_v1 - std_1_2_v1), min(y_1_5_v1 - std_1_5_v1)), 0),
             max(max(y_1_2_v1 + std_1_2_v1), max(y_1_5_v1 + std_1_5_v1)))

    ax[1][1].plot(x_1_2_v1, y_1_2_v1, label=line_1_2_label_v1, marker="s", ls='-', color="#599ad3",
                  markevery=markevery_1_2_v1)
    ax[1][1].fill_between(x_1_2_v1, y_1_2_v1 - std_1_2_v1, y_1_2_v1 + std_1_2_v1, alpha=0.35, color="#599ad3")

    ax[1][1].plot(x_1_5_v1, y_1_5_v1, label=line_1_5_label_v1, marker="o", ls='-', color='#f9a65a',
                  markevery=markevery_1_5_v1)
    ax[1][1].fill_between(x_1_5_v1, y_1_5_v1 - std_1_5_v1, y_1_5_v1 + std_1_5_v1, alpha=0.35, color='#f9a65a')

    ax[1][1].set_xlim(xlims)
    ax[1][1].set_ylim(ylims)

    ax[1][1].set_title(title_1_v1 + " v1")
    ax[1][1].set_xlabel(xlabel_1_v1)
    ax[1][1].set_ylabel(ylabel_1_v1)
    # set the grid on
    ax[1][1].grid('on')

    # tweak the axis labels
    xlab = ax[1][1].xaxis.get_label()
    ylab = ax[1][1].yaxis.get_label()

    # change the color of the top and right spines to opaque gray
    ax[1][1].spines['right'].set_color((.8, .8, .8))
    ax[1][1].spines['top'].set_color((.8, .8, .8))

    # Plot
    xlims = (min(min(x_1_3_v1), min(x_1_6_v1)),
             max(max(x_1_3_v1), max(x_1_6_v1)))
    ylims = (max(min(min(y_1_3_v1 - std_1_3_v1), min(y_1_6_v1 - std_1_6_v1)), 0),
             max(max(y_1_3_v1 + std_1_3_v1), max(y_1_6_v1 + std_1_6_v1)))

    ax[1][2].plot(x_1_3_v1, y_1_3_v1, label=line_1_3_label_v1, marker="s", ls='-', color="#599ad3",
                  markevery=markevery_1_3_v1)
    ax[1][2].fill_between(x_1_3_v1, y_1_3_v1 - std_1_3_v1, y_1_3_v1 + std_1_3_v1, alpha=0.35, color="#599ad3")

    ax[1][2].plot(x_1_6_v1, y_1_6_v1, label=line_1_6_label_v1, marker="o", ls='-', color='#f9a65a',
                  markevery=markevery_1_4_v1)
    ax[1][2].fill_between(x_1_6_v1, y_1_6_v1 - std_1_6_v1, y_1_6_v1 + std_1_6_v1, alpha=0.35, color='#f9a65a')

    ax[1][2].set_xlim(xlims)
    ax[1][2].set_ylim(ylims)

    ax[1][2].set_title(title_1_v1 + " v2")
    ax[1][2].set_xlabel(xlabel_1_v1)
    ax[1][2].set_ylabel(ylabel_1_v1)
    # set the grid on
    ax[1][2].grid('on')

    # tweak the axis labels
    xlab = ax[1][2].xaxis.get_label()
    ylab = ax[1][2].yaxis.get_label()

    # change the color of the top and right spines to opaque gray
    ax[1][2].spines['right'].set_color((.8, .8, .8))
    ax[1][2].spines['top'].set_color((.8, .8, .8))

    # RandomAttack vs TabQ
    # Plot avg hack_probability
    xlims = (min(min(x_1_1_v2), min(x_1_4_v2)),
             max(max(x_1_1_v2), max(x_1_4_v2)))
    ylims = (max(0, min(min(y_1_1_v2 - std_1_1_v2), min(y_1_4_v2 - std_1_4_v2)), 0),
             max(max(y_1_1_v2 + std_1_1_v2), max(y_1_4_v2 + std_1_4_v2)))

    ax[2][0].plot(x_1_1_v2, y_1_1_v2, label=line_1_1_label_v2, marker="s", ls='-', color="#599ad3",
                  markevery=markevery_1_1_v2)
    ax[2][0].fill_between(x_1_1_v2, y_1_1_v2 - std_1_1_v2, y_1_1_v2 + std_1_1_v2, alpha=0.35, color="#599ad3")

    ax[2][0].plot(x_1_4_v2, y_1_4_v2, label=line_1_4_label_v2, marker="o", ls='-', color='#f9a65a',
                  markevery=markevery_1_4_v2)
    ax[2][0].fill_between(x_1_4_v2, y_1_4_v2 - std_1_4_v2, y_1_4_v2 + std_1_4_v2, alpha=0.35, color='#f9a65a')

    ax[2][0].set_xlim(xlims)
    ax[2][0].set_ylim(ylims)

    ax[2][0].set_title(title_1_v2 + " v0")
    ax[2][0].set_xlabel(xlabel_1_v2)
    ax[2][0].set_ylabel(ylabel_1_v2)
    # set the grid on
    ax[2][0].grid('on')

    # tweak the axis labels
    xlab = ax[2][0].xaxis.get_label()
    ylab = ax[2][0].yaxis.get_label()

    # change the color of the top and right spines to opaque gray
    ax[2][0].spines['right'].set_color((.8, .8, .8))
    ax[2][0].spines['top'].set_color((.8, .8, .8))

    # # Plot avg hack_probability
    xlims = (min(min(x_1_2_v2), min(x_1_5_v2)),
             max(max(x_1_2_v2), max(x_1_5_v2)))
    ylims = (min(min(y_1_2_v2 - std_1_2_v2), min(y_1_5_v2 - std_1_5_v2)),
             max(max(y_1_2_v2 + std_1_2_v2), max(y_1_5_v2 + std_1_5_v2)))

    ax[2][1].plot(x_1_2_v2, y_1_2_v2, label=line_1_2_label_v2, marker="s", ls='-', color="#599ad3",
                  markevery=markevery_1_2_v2)
    ax[2][1].fill_between(x_1_2_v2, y_1_2_v2 - std_1_2_v2, y_1_2_v2 + std_1_2_v2, alpha=0.35, color="#599ad3")

    ax[2][1].plot(x_1_5_v2, y_1_5_v2, label=line_1_5_label_v2, marker="o", ls='-', color='#f9a65a',
                  markevery=markevery_1_5_v2)
    ax[2][1].fill_between(x_1_5_v2, y_1_5_v2 - std_1_5_v2, y_1_5_v2 + std_1_5_v2, alpha=0.35, color='#f9a65a')

    ax[2][1].set_xlim(xlims)
    ax[2][1].set_ylim(ylims)

    ax[2][1].set_title(title_1_v2 + " v1")
    ax[2][1].set_xlabel(xlabel_1_v2)
    ax[2][1].set_ylabel(ylabel_1_v2)
    # set the grid on
    ax[2][1].grid('on')

    # tweak the axis labels
    xlab = ax[2][1].xaxis.get_label()
    ylab = ax[2][1].yaxis.get_label()

    # change the color of the top and right spines to opaque gray
    ax[2][1].spines['right'].set_color((.8, .8, .8))
    ax[2][1].spines['top'].set_color((.8, .8, .8))

    # Plot
    xlims = (min(min(x_1_3_v2), min(x_1_6_v2)),
             max(max(x_1_3_v2), max(x_1_6_v2)))
    ylims = (max(0, min(min(y_1_3_v2 - std_1_3_v2), min(y_1_6_v2 - std_1_6_v2))),
             max(max(y_1_3_v2 + std_1_3_v2), max(y_1_6_v2 + std_1_6_v2)))

    ax[2][2].plot(x_1_3_v2, y_1_3_v2, label=line_1_3_label_v2, marker="s", ls='-', color="#599ad3",
                  markevery=markevery_1_3_v2)
    ax[2][2].fill_between(x_1_3_v2, y_1_3_v2 - std_1_3_v2, y_1_3_v2 + std_1_3_v2, alpha=0.35, color="#599ad3")

    ax[2][2].plot(x_1_6_v2, y_1_6_v2, label=line_1_6_label_v2, marker="o", ls='-', color='#f9a65a',
                  markevery=markevery_1_4_v2)
    ax[2][2].fill_between(x_1_6_v2, y_1_6_v2 - std_1_6_v2, y_1_6_v2 + std_1_6_v2, alpha=0.35, color='#f9a65a')

    ax[2][2].set_xlim(xlims)
    ax[2][2].set_ylim(ylims)

    ax[2][2].set_title(title_1_v2 + " v2")
    ax[2][2].set_xlabel(xlabel_1_v2)
    ax[2][2].set_ylabel(ylabel_1_v2)
    # set the grid on
    ax[2][2].grid('on')

    # tweak the axis labels
    xlab = ax[2][2].xaxis.get_label()
    ylab = ax[2][2].yaxis.get_label()

    # change the color of the top and right spines to opaque gray
    ax[2][2].spines['right'].set_color((.8, .8, .8))
    ax[2][2].spines['top'].set_color((.8, .8, .8))

    # TabQ vs RandomDefense
    # Plot avg hack_probability
    xlims = (min(min(x_1_1_v3), min(x_1_4_v3)),
             max(max(x_1_1_v3), max(x_1_4_v3)))
    ylims = (min(min(y_1_1_v3 - std_1_1_v3), min(y_1_4_v3 - std_1_4_v3)),
             max(max(y_1_1_v3 + std_1_1_v3), max(y_1_4_v3 + std_1_4_v3)))

    ax[3][0].plot(x_1_1_v3, y_1_1_v3, label=line_1_1_label_v3, marker="s", ls='-', color="#599ad3",
                  markevery=markevery_1_1_v3)
    ax[3][0].fill_between(x_1_1_v3, y_1_1_v3 - std_1_1_v3, y_1_1_v3 + std_1_1_v3, alpha=0.35, color="#599ad3")

    ax[3][0].plot(x_1_4_v3, y_1_4_v3, label=line_1_4_label_v3, marker="o", ls='-', color='#f9a65a',
                  markevery=markevery_1_4_v3)
    ax[3][0].fill_between(x_1_4_v3, y_1_4_v3 - std_1_4_v3, y_1_4_v3 + std_1_4_v3, alpha=0.35, color='#f9a65a')

    ax[3][0].set_xlim(xlims)
    ax[3][0].set_ylim(ylims)

    ax[3][0].set_title(title_1_v3 + " v0")
    ax[3][0].set_xlabel(xlabel_1_v3)
    ax[3][0].set_ylabel(ylabel_1_v3)
    # set the grid on
    ax[3][0].grid('on')

    # tweak the axis labels
    xlab = ax[3][0].xaxis.get_label()
    ylab = ax[3][0].yaxis.get_label()

    # change the color of the top and right spines to opaque gray
    ax[3][0].spines['right'].set_color((.8, .8, .8))
    ax[3][0].spines['top'].set_color((.8, .8, .8))

    # # Plot avg hack_probability
    xlims = (min(min(x_1_2_v3), min(x_1_5_v3)),
             max(max(x_1_2_v3), max(x_1_5_v3)))
    ylims = (max(min(min(y_1_2_v3 - std_1_2_v3), min(y_1_5_v3 - std_1_5_v3)), 0),
             max(max(y_1_2_v3 + std_1_2_v3), max(y_1_5_v3 + std_1_5_v3)))

    ax[3][1].plot(x_1_2_v3, y_1_2_v3, label=line_1_2_label_v3, marker="s", ls='-', color="#599ad3",
                  markevery=markevery_1_2_v3)
    ax[3][1].fill_between(x_1_2_v3, y_1_2_v3 - std_1_2_v3, y_1_2_v3 + std_1_2_v3, alpha=0.35, color="#599ad3")

    ax[3][1].plot(x_1_5_v3, y_1_5_v3, label=line_1_5_label_v3, marker="o", ls='-', color='#f9a65a',
                  markevery=markevery_1_5_v3)
    ax[3][1].fill_between(x_1_5_v3, y_1_5_v3 - std_1_5_v3, y_1_5_v3 + std_1_5_v3, alpha=0.35, color='#f9a65a')

    ax[3][1].set_xlim(xlims)
    ax[3][1].set_ylim(ylims)

    ax[3][1].set_title(title_1_v3 + " v1")
    ax[3][1].set_xlabel(xlabel_1_v3)
    ax[3][1].set_ylabel(ylabel_1_v3)
    # set the grid on
    ax[3][1].grid('on')

    # tweak the axis labels
    xlab = ax[3][1].xaxis.get_label()
    ylab = ax[3][1].yaxis.get_label()

    # change the color of the top and right spines to opaque gray
    ax[3][1].spines['right'].set_color((.8, .8, .8))
    ax[3][1].spines['top'].set_color((.8, .8, .8))

    # Plot
    xlims = (min(min(x_1_3_v3), min(x_1_6_v3)),
             max(max(x_1_3_v3), max(x_1_6_v3)))
    ylims = (max(0, min(min(y_1_3_v3 - std_1_3_v3), min(y_1_6_v3 - std_1_6_v3))),
             max(max(y_1_3_v3 + std_1_3_v3), max(y_1_6_v3 + std_1_6_v3)))

    ax[3][2].plot(x_1_3_v3, y_1_3_v3, label=line_1_3_label_v3, marker="s", ls='-', color="#599ad3",
                  markevery=markevery_1_3_v3)
    ax[3][2].fill_between(x_1_3_v3, y_1_3_v3 - std_1_3_v3, y_1_3_v3 + std_1_3_v3, alpha=0.35, color="#599ad3")

    ax[3][2].plot(x_1_6_v3, y_1_6_v3, label=line_1_6_label_v3, marker="o", ls='-', color='#f9a65a',
                  markevery=markevery_1_4_v3)
    ax[3][2].fill_between(x_1_6_v3, y_1_6_v3 - std_1_6_v3, y_1_6_v3 + std_1_6_v3, alpha=0.35, color='#f9a65a')

    ax[3][2].set_xlim(xlims)
    ax[3][2].set_ylim(ylims)

    ax[3][2].set_title(title_1_v3 + " v3")
    ax[3][2].set_xlabel(xlabel_1_v3)
    ax[3][2].set_ylabel(ylabel_1_v3)
    # set the grid on
    ax[3][2].grid('on')

    # tweak the axis labels
    xlab = ax[3][2].xaxis.get_label()
    ylab = ax[3][2].yaxis.get_label()
    xlab.set_size(6)
    ylab.set_size(6)

    # change the color of the top and right spines to opaque gray
    ax[3][2].spines['right'].set_color((.8, .8, .8))
    ax[3][2].spines['top'].set_color((.8, .8, .8))

    # TabQ vs TabQ
    # Plot avg hack_probability
    xlims = (min(min(x_1_1_v4), min(x_1_4_v4)),
             max(max(x_1_1_v4), max(x_1_4_v4)))
    ylims = (max(min(min(y_1_1_v4 - std_1_1_v4), min(y_1_4_v4 - std_1_4_v4)), 0),
             max(max(y_1_1_v4 + std_1_1_v4), max(y_1_4_v4 + std_1_4_v4)))

    ax[4][0].plot(x_1_1_v4, y_1_1_v4, label=line_1_1_label_v4, marker="s", ls='-', color="#599ad3",
                  markevery=markevery_1_1_v4)
    ax[4][0].fill_between(x_1_1_v4, y_1_1_v4 - std_1_1_v4, y_1_1_v4 + std_1_1_v4, alpha=0.35, color="#599ad3")

    ax[4][0].plot(x_1_4_v4, y_1_4_v4, label=line_1_4_label_v4, marker="o", ls='-', color='#f9a65a',
                  markevery=markevery_1_4_v4)
    ax[4][0].fill_between(x_1_4_v4, y_1_4_v4 - std_1_4_v4, y_1_4_v4 + std_1_4_v4, alpha=0.35, color='#f9a65a')

    ax[4][0].set_xlim(xlims)
    ax[4][0].set_ylim(ylims)

    ax[4][0].set_title(title_1_v4 + " v0")
    ax[4][0].set_xlabel(xlabel_1_v4)
    ax[4][0].set_ylabel(ylabel_1_v4)
    # set the grid on
    ax[4][0].grid('on')

    # tweak the axis labels
    xlab = ax[4][0].xaxis.get_label()
    ylab = ax[4][0].yaxis.get_label()

    # change the color of the top and right spines to opaque gray
    ax[4][0].spines['right'].set_color((.8, .8, .8))
    ax[4][0].spines['top'].set_color((.8, .8, .8))

    # # Plot avg hack_probability
    xlims = (min(min(x_1_2_v4), min(x_1_5_v4)),
             max(max(x_1_2_v4), max(x_1_5_v4)))
    ylims = (min(min(y_1_2_v4 - std_1_2_v4), min(y_1_5_v4 - std_1_5_v4)),
             max(max(y_1_2_v4 + std_1_2_v4), max(y_1_5_v4 + std_1_5_v4)))

    ax[4][1].plot(x_1_2_v4, y_1_2_v4, label=line_1_2_label_v4, marker="s", ls='-', color="#599ad3",
                  markevery=markevery_1_2_v4)
    ax[4][1].fill_between(x_1_2_v4, y_1_2_v4 - std_1_2_v4, y_1_2_v4 + std_1_2_v4, alpha=0.35, color="#599ad3")

    ax[4][1].plot(x_1_5_v4, y_1_5_v4, label=line_1_5_label_v4, marker="o", ls='-', color='#f9a65a',
                  markevery=markevery_1_5_v4)
    ax[4][1].fill_between(x_1_5_v4, y_1_5_v4 - std_1_5_v4, y_1_5_v4 + std_1_5_v4, alpha=0.35, color='#f9a65a')

    ax[4][1].set_xlim(xlims)
    ax[4][1].set_ylim(ylims)

    ax[4][1].set_title(title_1_v4 + " v1")
    ax[4][1].set_xlabel(xlabel_1_v4)
    ax[4][1].set_ylabel(ylabel_1_v4)
    # set the grid on
    ax[4][1].grid('on')

    # tweak the axis labels
    xlab = ax[4][1].xaxis.get_label()
    ylab = ax[4][1].yaxis.get_label()

    # change the color of the top and right spines to opaque gray
    ax[4][1].spines['right'].set_color((.8, .8, .8))
    ax[4][1].spines['top'].set_color((.8, .8, .8))

    # Plot
    xlims = (min(min(x_1_3_v4), min(x_1_6_v4)),
             max(max(x_1_3_v4), max(x_1_6_v4)))
    ylims = (min(min(y_1_3_v4 - std_1_3_v4), min(y_1_6_v4 - std_1_6_v4)),
             max(max(y_1_3_v4 + std_1_3_v4), max(y_1_6_v4 + std_1_6_v4)))

    ax[4][2].plot(x_1_3_v4, y_1_3_v4, label=line_1_3_label_v4, marker="s", ls='-', color="#599ad3",
                  markevery=markevery_1_3_v4)
    ax[4][2].fill_between(x_1_3_v4, y_1_3_v4 - std_1_3_v4, y_1_3_v4 + std_1_3_v4, alpha=0.35, color="#599ad3")

    ax[4][2].plot(x_1_6_v4, y_1_6_v4, label=line_1_6_label_v4, marker="o", ls='-', color='#f9a65a',
                  markevery=markevery_1_4_v4)
    ax[4][2].fill_between(x_1_6_v4, y_1_6_v4 - std_1_6_v4, y_1_6_v4 + std_1_6_v4, alpha=0.35, color='#f9a65a')

    ax[4][2].set_xlim(xlims)
    ax[4][2].set_ylim(ylims)

    ax[4][2].set_title(title_1_v4 + " v3")
    ax[4][2].set_xlabel(xlabel_1_v4)
    ax[4][2].set_ylabel(ylabel_1_v4)
    # set the grid on
    ax[4][2].grid('on')

    # tweak the axis labels
    xlab = ax[4][2].xaxis.get_label()
    ylab = ax[4][2].yaxis.get_label()

    # change the color of the top and right spines to opaque gray
    ax[4][2].spines['right'].set_color((.8, .8, .8))
    ax[4][2].spines['top'].set_color((.8, .8, .8))

    lines_labels = [ax[4][1].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    fig.legend(lines, labels, loc=(0.35, 0.012), ncol=2, borderaxespad=0.)

    #fig.subplots_adjust(bottom=0.2)
    fig.tight_layout()
    plt.subplots_adjust(wspace=wspace, hspace=0.7)
    fig.subplots_adjust(bottom=0.08)
    fig.savefig(file_name + ".png", format="png", bbox_inches='tight')
    fig.savefig(file_name + ".pdf", format='pdf', dpi=500, bbox_inches='tight', transparent=True)


def plot_sparse_dense_difference(maximal_attack_train_csv_paths_v0, maximal_attack_eval_csv_paths_v0,
                                        minimal_defense_train_csv_paths_v0, minimal_defense_eval_csv_paths_v0,
                                        random_attack_train_csv_paths_v0, random_attack_eval_csv_paths_v0,
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

                                        maximal_attack_train_csv_paths_v8, maximal_attack_eval_csv_paths_v8,
                                        minimal_defense_train_csv_paths_v8, minimal_defense_eval_csv_paths_v8,
                                        random_attack_train_csv_paths_v8, random_attack_eval_csv_paths_v8,
                                        random_defense_train_csv_paths_v8, random_defense_eval_csv_paths_v8,
                                        two_agents_train_csv_paths_v8, two_agents_eval_csv_paths_v8,

                                        maximal_attack_train_csv_paths_v9, maximal_attack_eval_csv_paths_v9,
                                        minimal_defense_train_csv_paths_v9, minimal_defense_eval_csv_paths_v9,
                                        random_attack_train_csv_paths_v9, random_attack_eval_csv_paths_v9,
                                        random_defense_train_csv_paths_v9, random_defense_eval_csv_paths_v9,
                                        two_agents_train_csv_paths_v9, two_agents_eval_csv_paths_v9,

                                        maximal_attack_train_csv_paths_v7, maximal_attack_eval_csv_paths_v7,
                                        minimal_defense_train_csv_paths_v7, minimal_defense_eval_csv_paths_v7,
                                        random_attack_train_csv_paths_v7, random_attack_eval_csv_paths_v7,
                                        random_defense_train_csv_paths_v7, random_defense_eval_csv_paths_v7,
                                        two_agents_train_csv_paths_v7, two_agents_eval_csv_paths_v7,

                                        algorithm, output_dir, eval_freq : int, train_log_freq : int, versions: list,
                                        wspace=0.28, file_name = "combined_plot_mult_versions_"):
    # V0
    train_max_attack_dfs_v0 = []
    eval_max_attack_dfs_v0 = []
    for csv_path in maximal_attack_train_csv_paths_v0:
        df = read_data(csv_path)
        train_max_attack_dfs_v0.append(df)

    for csv_path in maximal_attack_eval_csv_paths_v0:
        df = read_data(csv_path)
        eval_max_attack_dfs_v0.append(df)
    hack_prob_eval_max_attack_data_v0 = list(map(lambda df: df["hack_probability"].values, eval_max_attack_dfs_v0))
    hack_prob_eval_max_attack_means_v0 = np.mean(tuple(hack_prob_eval_max_attack_data_v0), axis=0)
    hack_prob_eval_max_attack_stds_v0 = np.std(tuple(hack_prob_eval_max_attack_data_v0), axis=0, ddof=1)

    train_min_defense_dfs_v0 = []
    eval_min_defense_dfs_v0 = []
    for csv_path in minimal_defense_train_csv_paths_v0:
        df = read_data(csv_path)
        train_min_defense_dfs_v0.append(df)

    for csv_path in minimal_defense_eval_csv_paths_v0:
        df = read_data(csv_path)
        eval_min_defense_dfs_v0.append(df)

    hack_prob_eval_min_defense_data_v0 = list(map(lambda df: df["hack_probability"].values, eval_min_defense_dfs_v0))
    hack_prob_eval_min_defense_means_v0 = np.mean(tuple(hack_prob_eval_min_defense_data_v0), axis=0)
    hack_prob_eval_min_defense_stds_v0 = np.std(tuple(hack_prob_eval_min_defense_data_v0), axis=0, ddof=1)

    train_random_attack_dfs_v0 = []
    eval_random_attack_dfs_v0 = []
    for csv_path in random_attack_train_csv_paths_v0:
        df = read_data(csv_path)
        train_random_attack_dfs_v0.append(df)

    for csv_path in random_attack_eval_csv_paths_v0:
        df = read_data(csv_path)
        eval_random_attack_dfs_v0.append(df)

    hack_prob_eval_random_attack_data_v0 = list(
        map(lambda df: df["hack_probability"].values, eval_random_attack_dfs_v0))
    hack_prob_eval_random_attack_means_v0 = np.mean(tuple(hack_prob_eval_random_attack_data_v0), axis=0)
    hack_prob_eval_random_attack_stds_v0 = np.std(tuple(hack_prob_eval_random_attack_data_v0), axis=0, ddof=1)

    train_random_defense_dfs_v0 = []
    eval_random_defense_dfs_v0 = []
    for csv_path in random_defense_train_csv_paths_v0:
        df = read_data(csv_path)
        train_random_defense_dfs_v0.append(df)

    for csv_path in random_defense_eval_csv_paths_v0:
        df = read_data(csv_path)
        eval_random_defense_dfs_v0.append(df)

    hack_prob_eval_random_defense_data_v0 = list(
        map(lambda df: df["hack_probability"].values, eval_random_defense_dfs_v0))
    hack_prob_eval_random_defense_means_v0 = np.mean(tuple(hack_prob_eval_random_defense_data_v0), axis=0)
    hack_prob_eval_random_defense_stds_v0 = np.std(tuple(hack_prob_eval_random_defense_data_v0), axis=0, ddof=1)

    train_two_agents_dfs_v0 = []
    eval_two_agents_dfs_v0 = []
    for csv_path in two_agents_train_csv_paths_v0:
        df = read_data(csv_path)
        train_two_agents_dfs_v0.append(df)

    for csv_path in two_agents_eval_csv_paths_v0:
        df = read_data(csv_path)
        eval_two_agents_dfs_v0.append(df)

    hack_prob_eval_two_agents_data_v0 = list(map(lambda df: df["hack_probability"].values, eval_two_agents_dfs_v0))
    hack_prob_eval_two_agents_means_v0 = np.mean(tuple(hack_prob_eval_two_agents_data_v0), axis=0)
    hack_prob_eval_two_agents_stds_v0 = np.std(tuple(hack_prob_eval_two_agents_data_v0), axis=0, ddof=1)

    # V2
    train_max_attack_dfs_v2 = []
    eval_max_attack_dfs_v2 = []
    for csv_path in maximal_attack_train_csv_paths_v2:
        df = read_data(csv_path)
        train_max_attack_dfs_v2.append(df)

    for csv_path in maximal_attack_eval_csv_paths_v2:
        df = read_data(csv_path)
        eval_max_attack_dfs_v2.append(df)

    hack_prob_eval_max_attack_data_v2 = list(map(lambda df: df["hack_probability"].values, eval_max_attack_dfs_v2))
    hack_prob_eval_max_attack_means_v2 = np.mean(tuple(hack_prob_eval_max_attack_data_v2), axis=0)
    hack_prob_eval_max_attack_stds_v2 = np.std(tuple(hack_prob_eval_max_attack_data_v2), axis=0, ddof=1)

    train_min_defense_dfs_v2 = []
    eval_min_defense_dfs_v2 = []
    for csv_path in minimal_defense_train_csv_paths_v2:
        df = read_data(csv_path)
        train_min_defense_dfs_v2.append(df)

    for csv_path in minimal_defense_eval_csv_paths_v2:
        df = read_data(csv_path)
        eval_min_defense_dfs_v2.append(df)

    hack_prob_eval_min_defense_data_v2 = list(map(lambda df: df["hack_probability"].values, eval_min_defense_dfs_v2))
    hack_prob_eval_min_defense_means_v2 = np.mean(tuple(hack_prob_eval_min_defense_data_v2), axis=0)
    hack_prob_eval_min_defense_stds_v2 = np.std(tuple(hack_prob_eval_min_defense_data_v2), axis=0, ddof=1)

    train_random_attack_dfs_v2 = []
    eval_random_attack_dfs_v2 = []
    for csv_path in random_attack_train_csv_paths_v2:
        df = read_data(csv_path)
        train_random_attack_dfs_v2.append(df)

    for csv_path in random_attack_eval_csv_paths_v2:
        df = read_data(csv_path)
        eval_random_attack_dfs_v2.append(df)

    hack_prob_eval_random_attack_data_v2 = list(
        map(lambda df: df["hack_probability"].values, eval_random_attack_dfs_v2))
    hack_prob_eval_random_attack_means_v2 = np.mean(tuple(hack_prob_eval_random_attack_data_v2), axis=0)
    hack_prob_eval_random_attack_stds_v2 = np.std(tuple(hack_prob_eval_random_attack_data_v2), axis=0, ddof=1)

    train_random_defense_dfs_v2 = []
    eval_random_defense_dfs_v2 = []
    for csv_path in random_defense_train_csv_paths_v2:
        df = read_data(csv_path)
        train_random_defense_dfs_v2.append(df)

    for csv_path in random_defense_eval_csv_paths_v2:
        df = read_data(csv_path)
        eval_random_defense_dfs_v2.append(df)

    hack_prob_eval_random_defense_data_v2 = list(
        map(lambda df: df["hack_probability"].values, eval_random_defense_dfs_v2))
    hack_prob_eval_random_defense_means_v2 = np.mean(tuple(hack_prob_eval_random_defense_data_v2), axis=0)
    hack_prob_eval_random_defense_stds_v2 = np.std(tuple(hack_prob_eval_random_defense_data_v2), axis=0, ddof=1)

    train_two_agents_dfs_v2 = []
    eval_two_agents_dfs_v2 = []
    for csv_path in two_agents_train_csv_paths_v2:
        df = read_data(csv_path)
        train_two_agents_dfs_v2.append(df)

    for csv_path in two_agents_eval_csv_paths_v2:
        df = read_data(csv_path)
        eval_two_agents_dfs_v2.append(df)

    hack_prob_eval_two_agents_data_v2 = list(map(lambda df: df["hack_probability"].values, eval_two_agents_dfs_v2))
    hack_prob_eval_two_agents_means_v2 = np.mean(tuple(hack_prob_eval_two_agents_data_v2), axis=0)
    hack_prob_eval_two_agents_stds_v2 = np.std(tuple(hack_prob_eval_two_agents_data_v2), axis=0, ddof=1)

    # V3
    train_max_attack_dfs_v3 = []
    eval_max_attack_dfs_v3 = []
    for csv_path in maximal_attack_train_csv_paths_v3:
        df = read_data(csv_path)
        train_max_attack_dfs_v3.append(df)

    for csv_path in maximal_attack_eval_csv_paths_v3:
        df = read_data(csv_path)
        eval_max_attack_dfs_v3.append(df)

    hack_prob_eval_max_attack_data_v3 = list(map(lambda df: df["hack_probability"].values, eval_max_attack_dfs_v3))
    hack_prob_eval_max_attack_means_v3 = np.mean(tuple(hack_prob_eval_max_attack_data_v3), axis=0)
    hack_prob_eval_max_attack_stds_v3 = np.std(tuple(hack_prob_eval_max_attack_data_v3), axis=0, ddof=1)

    train_min_defense_dfs_v3 = []
    eval_min_defense_dfs_v3 = []
    for csv_path in minimal_defense_train_csv_paths_v3:
        df = read_data(csv_path)
        train_min_defense_dfs_v3.append(df)

    for csv_path in minimal_defense_eval_csv_paths_v3:
        df = read_data(csv_path)
        eval_min_defense_dfs_v3.append(df)

    hack_prob_eval_min_defense_data_v3 = list(map(lambda df: df["hack_probability"].values, eval_min_defense_dfs_v3))
    hack_prob_eval_min_defense_means_v3 = np.mean(tuple(hack_prob_eval_min_defense_data_v3), axis=0)
    hack_prob_eval_min_defense_stds_v3 = np.std(tuple(hack_prob_eval_min_defense_data_v3), axis=0, ddof=1)

    train_random_attack_dfs_v3 = []
    eval_random_attack_dfs_v3 = []
    for csv_path in random_attack_train_csv_paths_v3:
        df = read_data(csv_path)
        train_random_attack_dfs_v3.append(df)

    for csv_path in random_attack_eval_csv_paths_v3:
        df = read_data(csv_path)
        eval_random_attack_dfs_v3.append(df)

    hack_prob_eval_random_attack_data_v3 = list(
        map(lambda df: df["hack_probability"].values, eval_random_attack_dfs_v3))
    hack_prob_eval_random_attack_means_v3 = np.mean(tuple(hack_prob_eval_random_attack_data_v3), axis=0)
    hack_prob_eval_random_attack_stds_v3 = np.std(tuple(hack_prob_eval_random_attack_data_v3), axis=0, ddof=1)

    train_random_defense_dfs_v3 = []
    eval_random_defense_dfs_v3 = []
    for csv_path in random_defense_train_csv_paths_v3:
        df = read_data(csv_path)
        train_random_defense_dfs_v3.append(df)

    for csv_path in random_defense_eval_csv_paths_v3:
        df = read_data(csv_path)
        eval_random_defense_dfs_v3.append(df)

    hack_prob_eval_random_defense_data_v3 = list(
        map(lambda df: df["hack_probability"].values, eval_random_defense_dfs_v3))
    hack_prob_eval_random_defense_means_v3 = np.mean(tuple(hack_prob_eval_random_defense_data_v3), axis=0)
    hack_prob_eval_random_defense_stds_v3 = np.std(tuple(hack_prob_eval_random_defense_data_v3), axis=0, ddof=1)

    train_two_agents_dfs_v3 = []
    eval_two_agents_dfs_v3 = []
    for csv_path in two_agents_train_csv_paths_v3:
        df = read_data(csv_path)
        train_two_agents_dfs_v3.append(df)

    for csv_path in two_agents_eval_csv_paths_v3:
        df = read_data(csv_path)
        eval_two_agents_dfs_v3.append(df)

    hack_prob_eval_two_agents_data_v3 = list(map(lambda df: df["hack_probability"].values, eval_two_agents_dfs_v3))
    hack_prob_eval_two_agents_means_v3 = np.mean(tuple(hack_prob_eval_two_agents_data_v3), axis=0)
    hack_prob_eval_two_agents_stds_v3 = np.std(tuple(hack_prob_eval_two_agents_data_v3), axis=0, ddof=1)

    # V8
    train_max_attack_dfs_v8 = []
    eval_max_attack_dfs_v8 = []
    for csv_path in maximal_attack_train_csv_paths_v8:
        df = read_data(csv_path)
        train_max_attack_dfs_v8.append(df)

    for csv_path in maximal_attack_eval_csv_paths_v8:
        df = read_data(csv_path)
        eval_max_attack_dfs_v8.append(df)
    hack_prob_eval_max_attack_data_v8 = list(map(lambda df: df["hack_probability"].values, eval_max_attack_dfs_v8))
    hack_prob_eval_max_attack_means_v8 = np.mean(tuple(hack_prob_eval_max_attack_data_v8), axis=0)
    hack_prob_eval_max_attack_stds_v8 = np.std(tuple(hack_prob_eval_max_attack_data_v8), axis=0, ddof=1)

    train_min_defense_dfs_v8 = []
    eval_min_defense_dfs_v8 = []
    for csv_path in minimal_defense_train_csv_paths_v8:
        df = read_data(csv_path)
        train_min_defense_dfs_v8.append(df)

    for csv_path in minimal_defense_eval_csv_paths_v8:
        df = read_data(csv_path)
        eval_min_defense_dfs_v8.append(df)

    hack_prob_eval_min_defense_data_v8 = list(map(lambda df: df["hack_probability"].values, eval_min_defense_dfs_v8))
    hack_prob_eval_min_defense_means_v8 = np.mean(tuple(hack_prob_eval_min_defense_data_v8), axis=0)
    hack_prob_eval_min_defense_stds_v8 = np.std(tuple(hack_prob_eval_min_defense_data_v8), axis=0, ddof=1)

    train_random_attack_dfs_v8 = []
    eval_random_attack_dfs_v8 = []
    for csv_path in random_attack_train_csv_paths_v8:
        df = read_data(csv_path)
        train_random_attack_dfs_v8.append(df)

    for csv_path in random_attack_eval_csv_paths_v8:
        df = read_data(csv_path)
        eval_random_attack_dfs_v8.append(df)

    hack_prob_eval_random_attack_data_v8 = list(
        map(lambda df: df["hack_probability"].values, eval_random_attack_dfs_v8))
    hack_prob_eval_random_attack_means_v8 = np.mean(tuple(hack_prob_eval_random_attack_data_v8), axis=0)
    hack_prob_eval_random_attack_stds_v8 = np.std(tuple(hack_prob_eval_random_attack_data_v8), axis=0, ddof=1)

    train_random_defense_dfs_v8 = []
    eval_random_defense_dfs_v8 = []
    for csv_path in random_defense_train_csv_paths_v8:
        df = read_data(csv_path)
        train_random_defense_dfs_v8.append(df)

    for csv_path in random_defense_eval_csv_paths_v8:
        df = read_data(csv_path)
        eval_random_defense_dfs_v8.append(df)

    hack_prob_eval_random_defense_data_v8 = list(
        map(lambda df: df["hack_probability"].values, eval_random_defense_dfs_v8))
    hack_prob_eval_random_defense_means_v8 = np.mean(tuple(hack_prob_eval_random_defense_data_v8), axis=0)
    hack_prob_eval_random_defense_stds_v8 = np.std(tuple(hack_prob_eval_random_defense_data_v8), axis=0, ddof=1)

    train_two_agents_dfs_v8 = []
    eval_two_agents_dfs_v8 = []
    for csv_path in two_agents_train_csv_paths_v8:
        df = read_data(csv_path)
        train_two_agents_dfs_v8.append(df)

    for csv_path in two_agents_eval_csv_paths_v8:
        df = read_data(csv_path)
        eval_two_agents_dfs_v8.append(df)

    hack_prob_eval_two_agents_data_v8 = list(map(lambda df: df["hack_probability"].values, eval_two_agents_dfs_v8))
    hack_prob_eval_two_agents_means_v8 = np.mean(tuple(hack_prob_eval_two_agents_data_v8), axis=0)
    hack_prob_eval_two_agents_stds_v8 = np.std(tuple(hack_prob_eval_two_agents_data_v8), axis=0, ddof=1)

    # V9
    train_max_attack_dfs_v9 = []
    eval_max_attack_dfs_v9 = []
    for csv_path in maximal_attack_train_csv_paths_v9:
        df = read_data(csv_path)
        train_max_attack_dfs_v9.append(df)

    for csv_path in maximal_attack_eval_csv_paths_v9:
        df = read_data(csv_path)
        eval_max_attack_dfs_v9.append(df)

    hack_prob_eval_max_attack_data_v9 = list(map(lambda df: df["hack_probability"].values, eval_max_attack_dfs_v9))
    hack_prob_eval_max_attack_means_v9 = np.mean(tuple(hack_prob_eval_max_attack_data_v9), axis=0)
    hack_prob_eval_max_attack_stds_v9 = np.std(tuple(hack_prob_eval_max_attack_data_v9), axis=0, ddof=1)

    train_min_defense_dfs_v9 = []
    eval_min_defense_dfs_v9 = []
    for csv_path in minimal_defense_train_csv_paths_v9:
        df = read_data(csv_path)
        train_min_defense_dfs_v9.append(df)

    for csv_path in minimal_defense_eval_csv_paths_v9:
        df = read_data(csv_path)
        eval_min_defense_dfs_v9.append(df)

    hack_prob_eval_min_defense_data_v9 = list(map(lambda df: df["hack_probability"].values, eval_min_defense_dfs_v9))
    hack_prob_eval_min_defense_means_v9 = np.mean(tuple(hack_prob_eval_min_defense_data_v9), axis=0)
    hack_prob_eval_min_defense_stds_v9 = np.std(tuple(hack_prob_eval_min_defense_data_v9), axis=0, ddof=1)

    train_random_attack_dfs_v9 = []
    eval_random_attack_dfs_v9 = []
    for csv_path in random_attack_train_csv_paths_v9:
        df = read_data(csv_path)
        train_random_attack_dfs_v9.append(df)

    for csv_path in random_attack_eval_csv_paths_v9:
        df = read_data(csv_path)
        eval_random_attack_dfs_v9.append(df)

    hack_prob_eval_random_attack_data_v9 = list(
        map(lambda df: df["hack_probability"].values, eval_random_attack_dfs_v9))
    hack_prob_eval_random_attack_means_v9 = np.mean(tuple(hack_prob_eval_random_attack_data_v9), axis=0)
    hack_prob_eval_random_attack_stds_v9 = np.std(tuple(hack_prob_eval_random_attack_data_v9), axis=0, ddof=1)

    train_random_defense_dfs_v9 = []
    eval_random_defense_dfs_v9 = []
    for csv_path in random_defense_train_csv_paths_v9:
        df = read_data(csv_path)
        train_random_defense_dfs_v9.append(df)

    for csv_path in random_defense_eval_csv_paths_v9:
        df = read_data(csv_path)
        eval_random_defense_dfs_v9.append(df)

    hack_prob_eval_random_defense_data_v9 = list(
        map(lambda df: df["hack_probability"].values, eval_random_defense_dfs_v9))
    hack_prob_eval_random_defense_means_v9 = np.mean(tuple(hack_prob_eval_random_defense_data_v9), axis=0)
    hack_prob_eval_random_defense_stds_v9 = np.std(tuple(hack_prob_eval_random_defense_data_v9), axis=0, ddof=1)

    train_two_agents_dfs_v9 = []
    eval_two_agents_dfs_v9 = []
    for csv_path in two_agents_train_csv_paths_v9:
        df = read_data(csv_path)
        train_two_agents_dfs_v9.append(df)

    for csv_path in two_agents_eval_csv_paths_v9:
        df = read_data(csv_path)
        eval_two_agents_dfs_v9.append(df)

    hack_prob_eval_two_agents_data_v9 = list(map(lambda df: df["hack_probability"].values, eval_two_agents_dfs_v9))
    hack_prob_eval_two_agents_means_v9 = np.mean(tuple(hack_prob_eval_two_agents_data_v9), axis=0)
    hack_prob_eval_two_agents_stds_v9 = np.std(tuple(hack_prob_eval_two_agents_data_v9), axis=0, ddof=1)

    # V7
    train_max_attack_dfs_v7 = []
    eval_max_attack_dfs_v7 = []
    for csv_path in maximal_attack_train_csv_paths_v7:
        df = read_data(csv_path)
        train_max_attack_dfs_v7.append(df)

    for csv_path in maximal_attack_eval_csv_paths_v7:
        df = read_data(csv_path)
        eval_max_attack_dfs_v7.append(df)

    hack_prob_eval_max_attack_data_v7 = list(map(lambda df: df["hack_probability"].values, eval_max_attack_dfs_v7))
    hack_prob_eval_max_attack_means_v7 = np.mean(tuple(hack_prob_eval_max_attack_data_v7), axis=0)
    hack_prob_eval_max_attack_stds_v7 = np.std(tuple(hack_prob_eval_max_attack_data_v7), axis=0, ddof=1)

    train_min_defense_dfs_v7 = []
    eval_min_defense_dfs_v7 = []
    for csv_path in minimal_defense_train_csv_paths_v7:
        df = read_data(csv_path)
        train_min_defense_dfs_v7.append(df)

    for csv_path in minimal_defense_eval_csv_paths_v7:
        df = read_data(csv_path)
        eval_min_defense_dfs_v7.append(df)

    hack_prob_eval_min_defense_data_v7 = list(map(lambda df: df["hack_probability"].values, eval_min_defense_dfs_v7))
    hack_prob_eval_min_defense_means_v7 = np.mean(tuple(hack_prob_eval_min_defense_data_v7), axis=0)
    hack_prob_eval_min_defense_stds_v7 = np.std(tuple(hack_prob_eval_min_defense_data_v7), axis=0, ddof=1)

    train_random_attack_dfs_v7 = []
    eval_random_attack_dfs_v7 = []
    for csv_path in random_attack_train_csv_paths_v7:
        df = read_data(csv_path)
        train_random_attack_dfs_v7.append(df)

    for csv_path in random_attack_eval_csv_paths_v7:
        df = read_data(csv_path)
        eval_random_attack_dfs_v7.append(df)

    hack_prob_eval_random_attack_data_v7 = list(
        map(lambda df: df["hack_probability"].values, eval_random_attack_dfs_v7))
    hack_prob_eval_random_attack_means_v7 = np.mean(tuple(hack_prob_eval_random_attack_data_v7), axis=0)
    hack_prob_eval_random_attack_stds_v7 = np.std(tuple(hack_prob_eval_random_attack_data_v7), axis=0, ddof=1)

    train_random_defense_dfs_v7 = []
    eval_random_defense_dfs_v7 = []
    for csv_path in random_defense_train_csv_paths_v7:
        df = read_data(csv_path)
        train_random_defense_dfs_v7.append(df)

    for csv_path in random_defense_eval_csv_paths_v7:
        df = read_data(csv_path)
        eval_random_defense_dfs_v7.append(df)

    hack_prob_eval_random_defense_data_v7 = list(
        map(lambda df: df["hack_probability"].values, eval_random_defense_dfs_v7))
    hack_prob_eval_random_defense_means_v7 = np.mean(tuple(hack_prob_eval_random_defense_data_v7), axis=0)
    hack_prob_eval_random_defense_stds_v7 = np.std(tuple(hack_prob_eval_random_defense_data_v7), axis=0, ddof=1)

    train_two_agents_dfs_v7 = []
    eval_two_agents_dfs_v7 = []
    for csv_path in two_agents_train_csv_paths_v7:
        df = read_data(csv_path)
        train_two_agents_dfs_v7.append(df)

    for csv_path in two_agents_eval_csv_paths_v7:
        df = read_data(csv_path)
        eval_two_agents_dfs_v7.append(df)

    hack_prob_eval_two_agents_data_v7 = list(map(lambda df: df["hack_probability"].values, eval_two_agents_dfs_v7))
    hack_prob_eval_two_agents_means_v7 = np.mean(tuple(hack_prob_eval_two_agents_data_v7), axis=0)
    hack_prob_eval_two_agents_stds_v7 = np.std(tuple(hack_prob_eval_two_agents_data_v7), axis=0, ddof=1)

    plot_all_avg_summary_4(np.array(list(range(len(hack_prob_eval_min_defense_data_v0[0])))) * eval_freq,
                           hack_prob_eval_min_defense_means_v0,
                           np.array(list(range(len(hack_prob_eval_min_defense_data_v2[0])))) * eval_freq,
                           hack_prob_eval_min_defense_means_v2,
                           np.array(list(range(len(hack_prob_eval_min_defense_data_v3[0])))) * eval_freq,
                           hack_prob_eval_min_defense_means_v3,
                           np.array(list(range(len(hack_prob_eval_min_defense_data_v8[0])))) * eval_freq,
                           hack_prob_eval_min_defense_means_v8,
                           np.array(list(range(len(hack_prob_eval_min_defense_data_v9[0])))) * eval_freq,
                           hack_prob_eval_min_defense_means_v9,
                           np.array(list(range(len(hack_prob_eval_min_defense_data_v7[0])))) * eval_freq,
                           hack_prob_eval_min_defense_means_v7,
                           hack_prob_eval_min_defense_stds_v0, hack_prob_eval_min_defense_stds_v2,
                           hack_prob_eval_min_defense_stds_v3, hack_prob_eval_min_defense_stds_v8,
                           hack_prob_eval_min_defense_stds_v9, hack_prob_eval_min_defense_stds_v7,
                           r"Sparse $\mathcal{R}$", r"Sparse $\mathcal{R}$",
                           r"Sparse $\mathcal{R}$", r"Dense $\mathcal{R}$",
                           r"Dense $\mathcal{R}$", r"Dense $\mathcal{R}$",
                           r"\textsc{TabularQLearning} vs \textsc{MinDefense}",
                           r"Episode \#", r"$\mathbb{P}[Hacked]$", 1, 1, 1, 1, 1, 1,

                           np.array(list(range(len(hack_prob_eval_max_attack_data_v0[0])))) * eval_freq,
                           hack_prob_eval_max_attack_means_v0,
                           np.array(list(range(len(hack_prob_eval_max_attack_data_v2[0])))) * eval_freq,
                           hack_prob_eval_max_attack_means_v2,
                           np.array(list(range(len(hack_prob_eval_max_attack_data_v3[0])))) * eval_freq,
                           hack_prob_eval_max_attack_means_v3,
                           np.array(list(range(len(hack_prob_eval_max_attack_data_v8[0])))) * eval_freq,
                           hack_prob_eval_max_attack_means_v8,
                           np.array(list(range(len(hack_prob_eval_max_attack_data_v9[0])))) * eval_freq,
                           hack_prob_eval_max_attack_means_v9,
                           np.array(list(range(len(hack_prob_eval_max_attack_data_v7[0])))) * eval_freq,
                           hack_prob_eval_max_attack_means_v7,
                           hack_prob_eval_max_attack_stds_v0, hack_prob_eval_max_attack_stds_v2,
                           hack_prob_eval_max_attack_stds_v3, hack_prob_eval_max_attack_stds_v8,
                           hack_prob_eval_max_attack_stds_v9, hack_prob_eval_max_attack_stds_v7,
                           r"Sparse $\mathcal{R}$", r"Sparse $\mathcal{R}$",
                           r"Sparse $\mathcal{R}$", r"Dense $\mathcal{R}$",
                           r"Dense $\mathcal{R}$", r"Dense $\mathcal{R}$",
                           r"\textsc{MaxAttack} vs \textsc{TabularQLearning}",
                           r"Episode \#", r"$\mathbb{P}[Hacked]$", 1, 1, 1, 1, 1, 1,

                           np.array(list(range(len(hack_prob_eval_random_attack_data_v0[0])))) * eval_freq,
                           hack_prob_eval_random_attack_means_v0,
                           np.array(list(range(len(hack_prob_eval_random_attack_data_v2[0])))) * eval_freq,
                           hack_prob_eval_random_attack_means_v2,
                           np.array(list(range(len(hack_prob_eval_random_attack_data_v3[0])))) * eval_freq,
                           hack_prob_eval_random_attack_means_v3,
                           np.array(list(range(len(hack_prob_eval_random_attack_data_v8[0])))) * eval_freq,
                           hack_prob_eval_random_attack_means_v8,
                           np.array(list(range(len(hack_prob_eval_random_attack_data_v9[0])))) * eval_freq,
                           hack_prob_eval_random_attack_means_v9,
                           np.array(list(range(len(hack_prob_eval_random_attack_data_v7[0])))) * eval_freq,
                           hack_prob_eval_random_attack_means_v7,
                           hack_prob_eval_random_attack_stds_v0, hack_prob_eval_random_attack_stds_v2,
                           hack_prob_eval_random_attack_stds_v3, hack_prob_eval_random_attack_stds_v8,
                           hack_prob_eval_random_attack_stds_v9, hack_prob_eval_random_attack_stds_v7,
                           r"Sparse $\mathcal{R}$", r"Sparse $\mathcal{R}$",
                           r"Sparse $\mathcal{R}$", r"Dense $\mathcal{R}$",
                           r"Dense $\mathcal{R}$", r"Dense $\mathcal{R}$",
                           r"\textsc{RandomAttack} vs \textsc{TabularQLearning}",
                           r"Episode \#", r"$\mathbb{P}[Hacked]$", 1, 1, 1, 1, 1, 1,

                           np.array(list(range(len(hack_prob_eval_random_defense_data_v0[0])))) * eval_freq,
                           hack_prob_eval_random_defense_means_v0,
                           np.array(list(range(len(hack_prob_eval_random_defense_data_v2[0])))) * eval_freq,
                           hack_prob_eval_random_defense_means_v2,
                           np.array(list(range(len(hack_prob_eval_random_defense_data_v3[0])))) * eval_freq,
                           hack_prob_eval_random_defense_means_v3,
                           np.array(list(range(len(hack_prob_eval_random_defense_data_v8[0])))) * eval_freq,
                           hack_prob_eval_random_defense_means_v8,
                           np.array(list(range(len(hack_prob_eval_random_defense_data_v9[0])))) * eval_freq,
                           hack_prob_eval_random_defense_means_v9,
                           np.array(list(range(len(hack_prob_eval_random_defense_data_v7[0])))) * eval_freq,
                           hack_prob_eval_random_defense_means_v7,
                           hack_prob_eval_random_defense_stds_v0, hack_prob_eval_random_defense_stds_v2,
                           hack_prob_eval_random_defense_stds_v3, hack_prob_eval_random_defense_stds_v8,
                           hack_prob_eval_random_defense_stds_v9, hack_prob_eval_random_defense_stds_v7,
                           r"Sparse $\mathcal{R}$", r"Sparse $\mathcal{R}$",
                           r"Sparse $\mathcal{R}$", r"Dense $\mathcal{R}$",
                           r"Dense $\mathcal{R}$", r"Dense $\mathcal{R}$",
                           r"\textsc{TabularQLearning} vs \textsc{RandomDefense}",
                           r"Episode \#", r"$\mathbb{P}[Hacked]$", 1, 1, 1, 1, 1, 1,

                           np.array(list(range(len(hack_prob_eval_two_agents_data_v0[0])))) * eval_freq,
                           hack_prob_eval_two_agents_means_v0,
                           np.array(list(range(len(hack_prob_eval_two_agents_data_v2[0])))) * eval_freq,
                           hack_prob_eval_two_agents_means_v2,
                           np.array(list(range(len(hack_prob_eval_two_agents_data_v3[0])))) * eval_freq,
                           hack_prob_eval_two_agents_means_v3,
                           np.array(list(range(len(hack_prob_eval_two_agents_data_v8[0])))) * eval_freq,
                           hack_prob_eval_two_agents_means_v8,
                           np.array(list(range(len(hack_prob_eval_two_agents_data_v9[0])))) * eval_freq,
                           hack_prob_eval_two_agents_means_v9,
                           np.array(list(range(len(hack_prob_eval_two_agents_data_v7[0])))) * eval_freq,
                           hack_prob_eval_two_agents_means_v7,
                           hack_prob_eval_two_agents_stds_v0, hack_prob_eval_two_agents_stds_v2,
                           hack_prob_eval_two_agents_stds_v3, hack_prob_eval_two_agents_stds_v8,
                           hack_prob_eval_two_agents_stds_v9, hack_prob_eval_two_agents_stds_v7,
                           r"Sparse Reward Function $\mathcal{R}$", r"Sparse Reward Function $\mathcal{R}$",
                           r"Sparse Reward Function $\mathcal{R}$", r"Dense Reward Function $\mathcal{R}$",
                           r"Dense Reward Function $\mathcal{R}$", r"Dense Reward Function $\mathcal{R}$",
                           r"\textsc{TabularQLearning} vs \textsc{TabularQLearning}",
                           r"Episode \#", r"$\mathbb{P}[Hacked]$", 1, 1, 1, 1, 1, 1,

                           output_dir + "/" + file_name + algorithm + "_" + str(0),
                           wspace=wspace
                           )



def plot_all_averages_multiple_versions(maximal_attack_train_csv_paths_v0, maximal_attack_eval_csv_paths_v0,
                                        minimal_defense_train_csv_paths_v0, minimal_defense_eval_csv_paths_v0,
                                        random_attack_train_csv_paths_v0, random_attack_eval_csv_paths_v0,
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

                                        algorithm, output_dir, eval_freq : int, train_log_freq : int, versions: list,
                                        wspace=0.28, file_name = "combined_plot_mult_versions_"):

    # V0
    train_max_attack_dfs_v0 = []
    eval_max_attack_dfs_v0 = []
    for csv_path in maximal_attack_train_csv_paths_v0:
        df = read_data(csv_path)
        train_max_attack_dfs_v0.append(df)

    for csv_path in maximal_attack_eval_csv_paths_v0:
        df = read_data(csv_path)
        eval_max_attack_dfs_v0.append(df)

    hack_prob_train_max_attack_data_v0 = list(map(lambda df: df["hack_probability"].values, train_max_attack_dfs_v0))
    hack_prob_train_max_attack_means_v0 = np.mean(tuple(hack_prob_train_max_attack_data_v0), axis=0)
    hack_prob_train_max_attack_stds_v0 = np.std(tuple(hack_prob_train_max_attack_data_v0), axis=0, ddof=1)
    hack_prob_eval_max_attack_data_v0 = list(map(lambda df: df["hack_probability"].values, eval_max_attack_dfs_v0))
    hack_prob_eval_max_attack_means_v0 = np.mean(tuple(hack_prob_eval_max_attack_data_v0), axis=0)
    hack_prob_eval_max_attack_stds_v0 = np.std(tuple(hack_prob_eval_max_attack_data_v0), axis=0, ddof=1)

    a_cum_reward_train_max_attack_data_v0 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, train_max_attack_dfs_v0))
    a_cum_reward_train_max_attack_means_v0 = np.mean(tuple(a_cum_reward_train_max_attack_data_v0), axis=0)
    a_cum_reward_train_max_attack_stds_v0 = np.std(tuple(a_cum_reward_train_max_attack_data_v0), axis=0, ddof=1)
    a_cum_reward_eval_max_attack_data_v0 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, eval_max_attack_dfs_v0))
    a_cum_reward_eval_max_attack_means_v0 = np.mean(tuple(a_cum_reward_eval_max_attack_data_v0), axis=0)
    a_cum_reward_eval_max_attack_stds_v0 = np.std(tuple(a_cum_reward_eval_max_attack_data_v0), axis=0, ddof=1)

    d_cum_reward_train_max_attack_data_v0 = list(
        map(lambda df: df["defender_cumulative_reward"].values, train_max_attack_dfs_v0))
    d_cum_reward_train_max_attack_means_v0 = np.mean(tuple(d_cum_reward_train_max_attack_data_v0), axis=0)
    d_cum_reward_train_max_attack_stds_v0 = np.std(tuple(d_cum_reward_train_max_attack_data_v0), axis=0, ddof=1)
    d_cum_reward_eval_max_attack_data_v0 = list(
        map(lambda df: df["defender_cumulative_reward"].values, eval_max_attack_dfs_v0))
    d_cum_reward_eval_max_attack_means_v0 = np.mean(tuple(d_cum_reward_eval_max_attack_data_v0), axis=0)
    d_cum_reward_eval_max_attack_stds_v0 = np.std(tuple(d_cum_reward_eval_max_attack_data_v0), axis=0, ddof=1)

    episode_len_train_max_attack_data_v0 = list(map(lambda df: df["avg_episode_steps"].values, train_max_attack_dfs_v0))
    episode_len_train_max_attack_means_v0 = np.mean(tuple(episode_len_train_max_attack_data_v0), axis=0)
    episode_len_train_max_attack_stds_v0 = np.std(tuple(episode_len_train_max_attack_data_v0), axis=0, ddof=1)
    episode_len_eval_max_attack_data_v0 = list(map(lambda df: df["avg_episode_steps"].values, eval_max_attack_dfs_v0))
    episode_len_eval_max_attack_means_v0 = np.mean(tuple(episode_len_eval_max_attack_data_v0), axis=0)
    episode_len_eval_max_attack_stds_v0 = np.std(tuple(episode_len_eval_max_attack_data_v0), axis=0, ddof=1)

    train_min_defense_dfs_v0 = []
    eval_min_defense_dfs_v0 = []
    for csv_path in minimal_defense_train_csv_paths_v0:
        df = read_data(csv_path)
        train_min_defense_dfs_v0.append(df)

    for csv_path in minimal_defense_eval_csv_paths_v0:
        df = read_data(csv_path)
        eval_min_defense_dfs_v0.append(df)

    hack_prob_train_min_defense_data_v0 = list(map(lambda df: df["hack_probability"].values, train_min_defense_dfs_v0))
    hack_prob_train_min_defense_means_v0 = np.mean(tuple(hack_prob_train_min_defense_data_v0), axis=0)
    hack_prob_train_min_defense_stds_v0 = np.std(tuple(hack_prob_train_min_defense_data_v0), axis=0, ddof=1)
    hack_prob_eval_min_defense_data_v0 = list(map(lambda df: df["hack_probability"].values, eval_min_defense_dfs_v0))
    hack_prob_eval_min_defense_means_v0 = np.mean(tuple(hack_prob_eval_min_defense_data_v0), axis=0)
    hack_prob_eval_min_defense_stds_v0 = np.std(tuple(hack_prob_eval_min_defense_data_v0), axis=0, ddof=1)

    a_cum_reward_train_min_defense_data_v0 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, train_min_defense_dfs_v0))
    a_cum_reward_train_min_defense_means_v0 = np.mean(tuple(a_cum_reward_train_min_defense_data_v0), axis=0)
    a_cum_reward_train_min_defense_stds_v0 = np.std(tuple(a_cum_reward_train_min_defense_data_v0), axis=0, ddof=1)
    a_cum_reward_eval_min_defense_data_v0 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, eval_min_defense_dfs_v0))
    a_cum_reward_eval_min_defense_means_v0 = np.mean(tuple(a_cum_reward_eval_min_defense_data_v0), axis=0)
    a_cum_reward_eval_min_defense_stds_v0 = np.std(tuple(a_cum_reward_eval_min_defense_data_v0), axis=0, ddof=1)

    d_cum_reward_train_min_defense_data_v0 = list(
        map(lambda df: df["defender_cumulative_reward"].values, train_min_defense_dfs_v0))
    d_cum_reward_train_min_defense_means_v0 = np.mean(tuple(d_cum_reward_train_min_defense_data_v0), axis=0)
    d_cum_reward_train_min_defense_stds_v0 = np.std(tuple(d_cum_reward_train_min_defense_data_v0), axis=0, ddof=1)
    d_cum_reward_eval_min_defense_data_v0 = list(
        map(lambda df: df["defender_cumulative_reward"].values, eval_min_defense_dfs_v0))
    d_cum_reward_eval_min_defense_means_v0 = np.mean(tuple(d_cum_reward_eval_min_defense_data_v0), axis=0)
    d_cum_reward_eval_min_defense_stds_v0 = np.std(tuple(d_cum_reward_eval_min_defense_data_v0), axis=0, ddof=1)

    episode_len_train_min_defense_data_v0 = list(map(lambda df: df["avg_episode_steps"].values, train_min_defense_dfs_v0))
    episode_len_train_min_defense_means_v0 = np.mean(tuple(episode_len_train_min_defense_data_v0), axis=0)
    episode_len_train_min_defense_stds_v0 = np.std(tuple(episode_len_train_min_defense_data_v0), axis=0, ddof=1)
    episode_len_eval_min_defense_data_v0 = list(map(lambda df: df["avg_episode_steps"].values, eval_min_defense_dfs_v0))
    episode_len_eval_min_defense_means_v0 = np.mean(tuple(episode_len_eval_min_defense_data_v0), axis=0)
    episode_len_eval_min_defense_stds_v0 = np.std(tuple(episode_len_eval_min_defense_data_v0), axis=0, ddof=1)

    train_random_attack_dfs_v0 = []
    eval_random_attack_dfs_v0 = []
    for csv_path in random_attack_train_csv_paths_v0:
        df = read_data(csv_path)
        train_random_attack_dfs_v0.append(df)

    for csv_path in random_attack_eval_csv_paths_v0:
        df = read_data(csv_path)
        eval_random_attack_dfs_v0.append(df)

    hack_prob_train_random_attack_data_v0 = list(map(lambda df: df["hack_probability"].values, train_random_attack_dfs_v0))
    hack_prob_train_random_attack_means_v0 = np.mean(tuple(hack_prob_train_random_attack_data_v0), axis=0)
    hack_prob_train_random_attack_stds_v0 = np.std(tuple(hack_prob_train_random_attack_data_v0), axis=0, ddof=1)
    hack_prob_eval_random_attack_data_v0 = list(map(lambda df: df["hack_probability"].values, eval_random_attack_dfs_v0))
    hack_prob_eval_random_attack_means_v0 = np.mean(tuple(hack_prob_eval_random_attack_data_v0), axis=0)
    hack_prob_eval_random_attack_stds_v0 = np.std(tuple(hack_prob_eval_random_attack_data_v0), axis=0, ddof=1)

    a_cum_reward_train_random_attack_data_v0 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, train_random_attack_dfs_v0))
    a_cum_reward_train_random_attack_means_v0 = np.mean(tuple(a_cum_reward_train_random_attack_data_v0), axis=0)
    a_cum_reward_train_random_attack_stds_v0 = np.std(tuple(a_cum_reward_train_random_attack_data_v0), axis=0, ddof=1)
    a_cum_reward_eval_random_attack_data_v0 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, eval_random_attack_dfs_v0))
    a_cum_reward_eval_random_attack_means_v0 = np.mean(tuple(a_cum_reward_eval_random_attack_data_v0), axis=0)
    a_cum_reward_eval_random_attack_stds_v0 = np.std(tuple(a_cum_reward_eval_random_attack_data_v0), axis=0, ddof=1)

    d_cum_reward_train_random_attack_data_v0 = list(
        map(lambda df: df["defender_cumulative_reward"].values, train_random_attack_dfs_v0))
    d_cum_reward_train_random_attack_means_v0 = np.mean(tuple(d_cum_reward_train_random_attack_data_v0), axis=0)
    d_cum_reward_train_random_attack_stds_v0 = np.std(tuple(d_cum_reward_train_random_attack_data_v0), axis=0, ddof=1)
    d_cum_reward_eval_random_attack_data_v0 = list(
        map(lambda df: df["defender_cumulative_reward"].values, eval_random_attack_dfs_v0))
    d_cum_reward_eval_random_attack_means_v0 = np.mean(tuple(d_cum_reward_eval_random_attack_data_v0), axis=0)
    d_cum_reward_eval_random_attack_stds_v0 = np.std(tuple(d_cum_reward_eval_random_attack_data_v0), axis=0, ddof=1)

    episode_len_train_random_attack_data_v0 = list(
        map(lambda df: df["avg_episode_steps"].values, train_random_attack_dfs_v0))
    episode_len_train_random_attack_means_v0 = np.mean(tuple(episode_len_train_random_attack_data_v0), axis=0)
    episode_len_train_random_attack_stds_v0 = np.std(tuple(episode_len_train_random_attack_data_v0), axis=0, ddof=1)
    episode_len_eval_random_attack_data_v0 = list(
        map(lambda df: df["avg_episode_steps"].values, eval_random_attack_dfs_v0))
    episode_len_eval_random_attack_means_v0 = np.mean(tuple(episode_len_eval_random_attack_data_v0), axis=0)
    episode_len_eval_random_attack_stds_v0 = np.std(tuple(episode_len_eval_random_attack_data_v0), axis=0, ddof=1)

    train_random_defense_dfs_v0 = []
    eval_random_defense_dfs_v0 = []
    for csv_path in random_defense_train_csv_paths_v0:
        df = read_data(csv_path)
        train_random_defense_dfs_v0.append(df)

    for csv_path in random_defense_eval_csv_paths_v0:
        df = read_data(csv_path)
        eval_random_defense_dfs_v0.append(df)

    hack_prob_train_random_defense_data_v0 = list(map(lambda df: df["hack_probability"].values, train_random_defense_dfs_v0))
    hack_prob_train_random_defense_means_v0 = np.mean(tuple(hack_prob_train_random_defense_data_v0), axis=0)
    hack_prob_train_random_defense_stds_v0 = np.std(tuple(hack_prob_train_random_defense_data_v0), axis=0, ddof=1)
    hack_prob_eval_random_defense_data_v0 = list(map(lambda df: df["hack_probability"].values, eval_random_defense_dfs_v0))
    hack_prob_eval_random_defense_means_v0 = np.mean(tuple(hack_prob_eval_random_defense_data_v0), axis=0)
    hack_prob_eval_random_defense_stds_v0 = np.std(tuple(hack_prob_eval_random_defense_data_v0), axis=0, ddof=1)

    a_cum_reward_train_random_defense_data_v0 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, train_random_defense_dfs_v0))
    a_cum_reward_train_random_defense_means_v0 = np.mean(tuple(a_cum_reward_train_random_defense_data_v0), axis=0)
    a_cum_reward_train_random_defense_stds_v0 = np.std(tuple(a_cum_reward_train_random_defense_data_v0), axis=0, ddof=1)
    a_cum_reward_eval_random_defense_data_v0 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, eval_random_defense_dfs_v0))
    a_cum_reward_eval_random_defense_means_v0 = np.mean(tuple(a_cum_reward_eval_random_defense_data_v0), axis=0)
    a_cum_reward_eval_random_defense_stds_v0 = np.std(tuple(a_cum_reward_eval_random_defense_data_v0), axis=0, ddof=1)

    d_cum_reward_train_random_defense_data_v0 = list(
        map(lambda df: df["defender_cumulative_reward"].values, train_random_defense_dfs_v0))
    d_cum_reward_train_random_defense_means_v0 = np.mean(tuple(d_cum_reward_train_random_defense_data_v0), axis=0)
    d_cum_reward_train_random_defense_stds_v0 = np.std(tuple(d_cum_reward_train_random_defense_data_v0), axis=0, ddof=1)
    d_cum_reward_eval_random_defense_data_v0 = list(
        map(lambda df: df["defender_cumulative_reward"].values, eval_random_defense_dfs_v0))
    d_cum_reward_eval_random_defense_means_v0 = np.mean(tuple(d_cum_reward_eval_random_defense_data_v0), axis=0)
    d_cum_reward_eval_random_defense_stds_v0 = np.std(tuple(d_cum_reward_eval_random_defense_data_v0), axis=0, ddof=1)

    episode_len_train_random_defense_data_v0 = list(
        map(lambda df: df["avg_episode_steps"].values, train_random_defense_dfs_v0))
    episode_len_train_random_defense_means_v0 = np.mean(tuple(episode_len_train_random_defense_data_v0), axis=0)
    episode_len_train_random_defense_stds_v0 = np.std(tuple(episode_len_train_random_defense_data_v0), axis=0, ddof=1)
    episode_len_eval_random_defense_data_v0 = list(
        map(lambda df: df["avg_episode_steps"].values, eval_random_defense_dfs_v0))
    episode_len_eval_random_defense_means_v0 = np.mean(tuple(episode_len_eval_random_defense_data_v0), axis=0)
    episode_len_eval_random_defense_stds_v0 = np.std(tuple(episode_len_eval_random_defense_data_v0), axis=0, ddof=1)

    train_two_agents_dfs_v0 = []
    eval_two_agents_dfs_v0 = []
    for csv_path in two_agents_train_csv_paths_v0:
        df = read_data(csv_path)
        train_two_agents_dfs_v0.append(df)

    for csv_path in two_agents_eval_csv_paths_v0:
        df = read_data(csv_path)
        eval_two_agents_dfs_v0.append(df)

    hack_prob_train_two_agents_data_v0 = list(map(lambda df: df["hack_probability"].values, train_two_agents_dfs_v0))
    hack_prob_train_two_agents_means_v0 = np.mean(tuple(hack_prob_train_two_agents_data_v0), axis=0)
    hack_prob_train_two_agents_stds_v0 = np.std(tuple(hack_prob_train_two_agents_data_v0), axis=0, ddof=1)
    hack_prob_eval_two_agents_data_v0 = list(map(lambda df: df["hack_probability"].values, eval_two_agents_dfs_v0))
    hack_prob_eval_two_agents_means_v0 = np.mean(tuple(hack_prob_eval_two_agents_data_v0), axis=0)
    hack_prob_eval_two_agents_stds_v0 = np.std(tuple(hack_prob_eval_two_agents_data_v0), axis=0, ddof=1)

    a_cum_reward_train_two_agents_data_v0 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, train_two_agents_dfs_v0))
    a_cum_reward_train_two_agents_means_v0 = np.mean(tuple(a_cum_reward_train_two_agents_data_v0), axis=0)
    a_cum_reward_train_two_agents_stds_v0 = np.std(tuple(a_cum_reward_train_two_agents_data_v0), axis=0, ddof=1)
    a_cum_reward_eval_two_agents_data = list(
        map(lambda df: df["attacker_cumulative_reward"].values, eval_two_agents_dfs_v0))
    a_cum_reward_eval_two_agents_means_v0 = np.mean(tuple(a_cum_reward_eval_two_agents_data), axis=0)
    a_cum_reward_eval_two_agents_stds_v0 = np.std(tuple(a_cum_reward_eval_two_agents_data), axis=0, ddof=1)

    d_cum_reward_train_two_agents_data_v0 = list(
        map(lambda df: df["defender_cumulative_reward"].values, train_two_agents_dfs_v0))
    d_cum_reward_train_two_agents_means_v0 = np.mean(tuple(d_cum_reward_train_two_agents_data_v0), axis=0)
    d_cum_reward_train_two_agents_stds_v0 = np.std(tuple(d_cum_reward_train_two_agents_data_v0), axis=0, ddof=1)
    d_cum_reward_eval_two_agents_data_v0 = list(
        map(lambda df: df["defender_cumulative_reward"].values, eval_two_agents_dfs_v0))
    d_cum_reward_eval_two_agents_means_v0 = np.mean(tuple(d_cum_reward_eval_two_agents_data_v0), axis=0)
    d_cum_reward_eval_two_agents_stds_v0 = np.std(tuple(d_cum_reward_eval_two_agents_data_v0), axis=0, ddof=1)

    episode_len_train_two_agents_data_v0 = list(map(lambda df: df["avg_episode_steps"].values, train_two_agents_dfs_v0))
    episode_len_train_two_agents_means_v0 = np.mean(tuple(episode_len_train_two_agents_data_v0), axis=0)
    episode_len_train_two_agents_stds_v0 = np.std(tuple(episode_len_train_two_agents_data_v0), axis=0, ddof=1)
    episode_len_eval_two_agents_data_v0 = list(map(lambda df: df["avg_episode_steps"].values, eval_two_agents_dfs_v0))
    episode_len_eval_two_agents_means_v0 = np.mean(tuple(episode_len_eval_two_agents_data_v0), axis=0)
    episode_len_eval_two_agents_stds_v0 = np.std(tuple(episode_len_eval_two_agents_data_v0), axis=0, ddof=1)



    # V2
    train_max_attack_dfs_v2 = []
    eval_max_attack_dfs_v2 = []
    for csv_path in maximal_attack_train_csv_paths_v2:
        df = read_data(csv_path)
        train_max_attack_dfs_v2.append(df)

    for csv_path in maximal_attack_eval_csv_paths_v2:
        df = read_data(csv_path)
        eval_max_attack_dfs_v2.append(df)

    hack_prob_train_max_attack_data_v2 = list(map(lambda df: df["hack_probability"].values, train_max_attack_dfs_v2))
    hack_prob_train_max_attack_means_v2 = np.mean(tuple(hack_prob_train_max_attack_data_v2), axis=0)
    hack_prob_train_max_attack_stds_v2 = np.std(tuple(hack_prob_train_max_attack_data_v2), axis=0, ddof=1)
    hack_prob_eval_max_attack_data_v2 = list(map(lambda df: df["hack_probability"].values, eval_max_attack_dfs_v2))
    hack_prob_eval_max_attack_means_v2 = np.mean(tuple(hack_prob_eval_max_attack_data_v2), axis=0)
    hack_prob_eval_max_attack_stds_v2 = np.std(tuple(hack_prob_eval_max_attack_data_v2), axis=0, ddof=1)

    a_cum_reward_train_max_attack_data_v2 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, train_max_attack_dfs_v2))
    a_cum_reward_train_max_attack_means_v2 = np.mean(tuple(a_cum_reward_train_max_attack_data_v2), axis=0)
    a_cum_reward_train_max_attack_stds_v2 = np.std(tuple(a_cum_reward_train_max_attack_data_v2), axis=0, ddof=1)
    a_cum_reward_eval_max_attack_data_v2 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, eval_max_attack_dfs_v2))
    a_cum_reward_eval_max_attack_means_v2 = np.mean(tuple(a_cum_reward_eval_max_attack_data_v2), axis=0)
    a_cum_reward_eval_max_attack_stds_v2 = np.std(tuple(a_cum_reward_eval_max_attack_data_v2), axis=0, ddof=1)

    d_cum_reward_train_max_attack_data_v2 = list(
        map(lambda df: df["defender_cumulative_reward"].values, train_max_attack_dfs_v2))
    d_cum_reward_train_max_attack_means_v2 = np.mean(tuple(d_cum_reward_train_max_attack_data_v2), axis=0)
    d_cum_reward_train_max_attack_stds_v2 = np.std(tuple(d_cum_reward_train_max_attack_data_v2), axis=0, ddof=1)
    d_cum_reward_eval_max_attack_data_v2 = list(
        map(lambda df: df["defender_cumulative_reward"].values, eval_max_attack_dfs_v2))
    d_cum_reward_eval_max_attack_means_v2 = np.mean(tuple(d_cum_reward_eval_max_attack_data_v2), axis=0)
    d_cum_reward_eval_max_attack_stds_v2 = np.std(tuple(d_cum_reward_eval_max_attack_data_v2), axis=0, ddof=1)

    episode_len_train_max_attack_data_v2 = list(map(lambda df: df["avg_episode_steps"].values, train_max_attack_dfs_v2))
    episode_len_train_max_attack_means_v2 = np.mean(tuple(episode_len_train_max_attack_data_v2), axis=0)
    episode_len_train_max_attack_stds_v2 = np.std(tuple(episode_len_train_max_attack_data_v2), axis=0, ddof=1)
    episode_len_eval_max_attack_data_v2 = list(map(lambda df: df["avg_episode_steps"].values, eval_max_attack_dfs_v2))
    episode_len_eval_max_attack_means_v2 = np.mean(tuple(episode_len_eval_max_attack_data_v2), axis=0)
    episode_len_eval_max_attack_stds_v2 = np.std(tuple(episode_len_eval_max_attack_data_v2), axis=0, ddof=1)

    train_min_defense_dfs_v2 = []
    eval_min_defense_dfs_v2 = []
    for csv_path in minimal_defense_train_csv_paths_v2:
        df = read_data(csv_path)
        train_min_defense_dfs_v2.append(df)

    for csv_path in minimal_defense_eval_csv_paths_v2:
        df = read_data(csv_path)
        eval_min_defense_dfs_v2.append(df)

    hack_prob_train_min_defense_data_v2 = list(map(lambda df: df["hack_probability"].values, train_min_defense_dfs_v2))
    hack_prob_train_min_defense_means_v2 = np.mean(tuple(hack_prob_train_min_defense_data_v2), axis=0)
    hack_prob_train_min_defense_stds_v2 = np.std(tuple(hack_prob_train_min_defense_data_v2), axis=0, ddof=1)
    hack_prob_eval_min_defense_data_v2 = list(map(lambda df: df["hack_probability"].values, eval_min_defense_dfs_v2))
    hack_prob_eval_min_defense_means_v2 = np.mean(tuple(hack_prob_eval_min_defense_data_v2), axis=0)
    hack_prob_eval_min_defense_stds_v2 = np.std(tuple(hack_prob_eval_min_defense_data_v2), axis=0, ddof=1)

    a_cum_reward_train_min_defense_data_v2 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, train_min_defense_dfs_v2))
    a_cum_reward_train_min_defense_means_v2 = np.mean(tuple(a_cum_reward_train_min_defense_data_v2), axis=0)
    a_cum_reward_train_min_defense_stds_v2 = np.std(tuple(a_cum_reward_train_min_defense_data_v2), axis=0, ddof=1)
    a_cum_reward_eval_min_defense_data_v2 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, eval_min_defense_dfs_v2))
    a_cum_reward_eval_min_defense_means_v2 = np.mean(tuple(a_cum_reward_eval_min_defense_data_v2), axis=0)
    a_cum_reward_eval_min_defense_stds_v2 = np.std(tuple(a_cum_reward_eval_min_defense_data_v2), axis=0, ddof=1)

    d_cum_reward_train_min_defense_data_v2 = list(
        map(lambda df: df["defender_cumulative_reward"].values, train_min_defense_dfs_v2))
    d_cum_reward_train_min_defense_means_v2 = np.mean(tuple(d_cum_reward_train_min_defense_data_v2), axis=0)
    d_cum_reward_train_min_defense_stds_v2 = np.std(tuple(d_cum_reward_train_min_defense_data_v2), axis=0, ddof=1)
    d_cum_reward_eval_min_defense_data_v2 = list(
        map(lambda df: df["defender_cumulative_reward"].values, eval_min_defense_dfs_v2))
    d_cum_reward_eval_min_defense_means_v2 = np.mean(tuple(d_cum_reward_eval_min_defense_data_v2), axis=0)
    d_cum_reward_eval_min_defense_stds_v2 = np.std(tuple(d_cum_reward_eval_min_defense_data_v2), axis=0, ddof=1)

    episode_len_train_min_defense_data_v2 = list(map(lambda df: df["avg_episode_steps"].values, train_min_defense_dfs_v2))
    episode_len_train_min_defense_means_v2 = np.mean(tuple(episode_len_train_min_defense_data_v2), axis=0)
    episode_len_train_min_defense_stds_v2 = np.std(tuple(episode_len_train_min_defense_data_v2), axis=0, ddof=1)
    episode_len_eval_min_defense_data_v2 = list(map(lambda df: df["avg_episode_steps"].values, eval_min_defense_dfs_v2))
    episode_len_eval_min_defense_means_v2 = np.mean(tuple(episode_len_eval_min_defense_data_v2), axis=0)
    episode_len_eval_min_defense_stds_v2 = np.std(tuple(episode_len_eval_min_defense_data_v2), axis=0, ddof=1)

    train_random_attack_dfs_v2 = []
    eval_random_attack_dfs_v2 = []
    for csv_path in random_attack_train_csv_paths_v2:
        df = read_data(csv_path)
        train_random_attack_dfs_v2.append(df)

    for csv_path in random_attack_eval_csv_paths_v2:
        df = read_data(csv_path)
        eval_random_attack_dfs_v2.append(df)

    hack_prob_train_random_attack_data_v2 = list(
        map(lambda df: df["hack_probability"].values, train_random_attack_dfs_v2))
    hack_prob_train_random_attack_means_v2 = np.mean(tuple(hack_prob_train_random_attack_data_v2), axis=0)
    hack_prob_train_random_attack_stds_v2 = np.std(tuple(hack_prob_train_random_attack_data_v2), axis=0, ddof=1)
    hack_prob_eval_random_attack_data_v2 = list(
        map(lambda df: df["hack_probability"].values, eval_random_attack_dfs_v2))
    hack_prob_eval_random_attack_means_v2 = np.mean(tuple(hack_prob_eval_random_attack_data_v2), axis=0)
    hack_prob_eval_random_attack_stds_v2 = np.std(tuple(hack_prob_eval_random_attack_data_v2), axis=0, ddof=1)

    a_cum_reward_train_random_attack_data_v2 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, train_random_attack_dfs_v2))
    a_cum_reward_train_random_attack_means_v2 = np.mean(tuple(a_cum_reward_train_random_attack_data_v2), axis=0)
    a_cum_reward_train_random_attack_stds_v2 = np.std(tuple(a_cum_reward_train_random_attack_data_v2), axis=0, ddof=1)
    a_cum_reward_eval_random_attack_data_v2 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, eval_random_attack_dfs_v2))
    a_cum_reward_eval_random_attack_means_v2 = np.mean(tuple(a_cum_reward_eval_random_attack_data_v2), axis=0)
    a_cum_reward_eval_random_attack_stds_v2 = np.std(tuple(a_cum_reward_eval_random_attack_data_v2), axis=0, ddof=1)

    d_cum_reward_train_random_attack_data_v2 = list(
        map(lambda df: df["defender_cumulative_reward"].values, train_random_attack_dfs_v2))
    d_cum_reward_train_random_attack_means_v2 = np.mean(tuple(d_cum_reward_train_random_attack_data_v2), axis=0)
    d_cum_reward_train_random_attack_stds_v2 = np.std(tuple(d_cum_reward_train_random_attack_data_v2), axis=0, ddof=1)
    d_cum_reward_eval_random_attack_data_v2 = list(
        map(lambda df: df["defender_cumulative_reward"].values, eval_random_attack_dfs_v2))
    d_cum_reward_eval_random_attack_means_v2 = np.mean(tuple(d_cum_reward_eval_random_attack_data_v2), axis=0)
    d_cum_reward_eval_random_attack_stds_v2 = np.std(tuple(d_cum_reward_eval_random_attack_data_v2), axis=0, ddof=1)

    episode_len_train_random_attack_data_v2 = list(
        map(lambda df: df["avg_episode_steps"].values, train_random_attack_dfs_v2))
    episode_len_train_random_attack_means_v2 = np.mean(tuple(episode_len_train_random_attack_data_v2), axis=0)
    episode_len_train_random_attack_stds_v2 = np.std(tuple(episode_len_train_random_attack_data_v2), axis=0, ddof=1)
    episode_len_eval_random_attack_data_v2 = list(
        map(lambda df: df["avg_episode_steps"].values, eval_random_attack_dfs_v2))
    episode_len_eval_random_attack_means_v2 = np.mean(tuple(episode_len_eval_random_attack_data_v2), axis=0)
    episode_len_eval_random_attack_stds_v2 = np.std(tuple(episode_len_eval_random_attack_data_v2), axis=0, ddof=1)

    train_random_defense_dfs_v2 = []
    eval_random_defense_dfs_v2 = []
    for csv_path in random_defense_train_csv_paths_v2:
        df = read_data(csv_path)
        train_random_defense_dfs_v2.append(df)

    for csv_path in random_defense_eval_csv_paths_v2:
        df = read_data(csv_path)
        eval_random_defense_dfs_v2.append(df)

    hack_prob_train_random_defense_data_v2 = list(
        map(lambda df: df["hack_probability"].values, train_random_defense_dfs_v2))
    hack_prob_train_random_defense_means_v2 = np.mean(tuple(hack_prob_train_random_defense_data_v2), axis=0)
    hack_prob_train_random_defense_stds_v2 = np.std(tuple(hack_prob_train_random_defense_data_v2), axis=0, ddof=1)
    hack_prob_eval_random_defense_data_v2 = list(
        map(lambda df: df["hack_probability"].values, eval_random_defense_dfs_v2))
    hack_prob_eval_random_defense_means_v2 = np.mean(tuple(hack_prob_eval_random_defense_data_v2), axis=0)
    hack_prob_eval_random_defense_stds_v2 = np.std(tuple(hack_prob_eval_random_defense_data_v2), axis=0, ddof=1)

    a_cum_reward_train_random_defense_data_v2 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, train_random_defense_dfs_v2))
    a_cum_reward_train_random_defense_means_v2 = np.mean(tuple(a_cum_reward_train_random_defense_data_v2), axis=0)
    a_cum_reward_train_random_defense_stds_v2 = np.std(tuple(a_cum_reward_train_random_defense_data_v2), axis=0, ddof=1)
    a_cum_reward_eval_random_defense_data_v2 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, eval_random_defense_dfs_v2))
    a_cum_reward_eval_random_defense_means_v2 = np.mean(tuple(a_cum_reward_eval_random_defense_data_v2), axis=0)
    a_cum_reward_eval_random_defense_stds_v2 = np.std(tuple(a_cum_reward_eval_random_defense_data_v2), axis=0, ddof=1)

    d_cum_reward_train_random_defense_data_v2 = list(
        map(lambda df: df["defender_cumulative_reward"].values, train_random_defense_dfs_v2))
    d_cum_reward_train_random_defense_means_v2 = np.mean(tuple(d_cum_reward_train_random_defense_data_v2), axis=0)
    d_cum_reward_train_random_defense_stds_v2 = np.std(tuple(d_cum_reward_train_random_defense_data_v2), axis=0, ddof=1)
    d_cum_reward_eval_random_defense_data_v2 = list(
        map(lambda df: df["defender_cumulative_reward"].values, eval_random_defense_dfs_v2))
    d_cum_reward_eval_random_defense_means_v2 = np.mean(tuple(d_cum_reward_eval_random_defense_data_v2), axis=0)
    d_cum_reward_eval_random_defense_stds_v2 = np.std(tuple(d_cum_reward_eval_random_defense_data_v2), axis=0, ddof=1)

    episode_len_train_random_defense_data_v2 = list(
        map(lambda df: df["avg_episode_steps"].values, train_random_defense_dfs_v2))
    episode_len_train_random_defense_means_v2 = np.mean(tuple(episode_len_train_random_defense_data_v2), axis=0)
    episode_len_train_random_defense_stds_v2 = np.std(tuple(episode_len_train_random_defense_data_v2), axis=0, ddof=1)
    episode_len_eval_random_defense_data_v2 = list(
        map(lambda df: df["avg_episode_steps"].values, eval_random_defense_dfs_v2))
    episode_len_eval_random_defense_means_v2 = np.mean(tuple(episode_len_eval_random_defense_data_v2), axis=0)
    episode_len_eval_random_defense_stds_v2 = np.std(tuple(episode_len_eval_random_defense_data_v2), axis=0, ddof=1)

    train_two_agents_dfs_v2 = []
    eval_two_agents_dfs_v2 = []
    for csv_path in two_agents_train_csv_paths_v2:
        df = read_data(csv_path)
        train_two_agents_dfs_v2.append(df)

    for csv_path in two_agents_eval_csv_paths_v2:
        df = read_data(csv_path)
        eval_two_agents_dfs_v2.append(df)

    hack_prob_train_two_agents_data_v2 = list(map(lambda df: df["hack_probability"].values, train_two_agents_dfs_v2))
    hack_prob_train_two_agents_means_v2 = np.mean(tuple(hack_prob_train_two_agents_data_v2), axis=0)
    hack_prob_train_two_agents_stds_v2 = np.std(tuple(hack_prob_train_two_agents_data_v2), axis=0, ddof=1)
    hack_prob_eval_two_agents_data_v2 = list(map(lambda df: df["hack_probability"].values, eval_two_agents_dfs_v2))
    hack_prob_eval_two_agents_means_v2 = np.mean(tuple(hack_prob_eval_two_agents_data_v2), axis=0)
    hack_prob_eval_two_agents_stds_v2 = np.std(tuple(hack_prob_eval_two_agents_data_v2), axis=0, ddof=1)

    a_cum_reward_train_two_agents_data_v2 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, train_two_agents_dfs_v2))
    a_cum_reward_train_two_agents_means_v2 = np.mean(tuple(a_cum_reward_train_two_agents_data_v2), axis=0)
    a_cum_reward_train_two_agents_stds_v2 = np.std(tuple(a_cum_reward_train_two_agents_data_v2), axis=0, ddof=1)
    a_cum_reward_eval_two_agents_data = list(
        map(lambda df: df["attacker_cumulative_reward"].values, eval_two_agents_dfs_v2))
    a_cum_reward_eval_two_agents_means_v2 = np.mean(tuple(a_cum_reward_eval_two_agents_data), axis=0)
    a_cum_reward_eval_two_agents_stds_v2 = np.std(tuple(a_cum_reward_eval_two_agents_data), axis=0, ddof=1)

    d_cum_reward_train_two_agents_data_v2 = list(
        map(lambda df: df["defender_cumulative_reward"].values, train_two_agents_dfs_v2))
    d_cum_reward_train_two_agents_means_v2 = np.mean(tuple(d_cum_reward_train_two_agents_data_v2), axis=0)
    d_cum_reward_train_two_agents_stds_v2 = np.std(tuple(d_cum_reward_train_two_agents_data_v2), axis=0, ddof=1)
    d_cum_reward_eval_two_agents_data_v2 = list(
        map(lambda df: df["defender_cumulative_reward"].values, eval_two_agents_dfs_v2))
    d_cum_reward_eval_two_agents_means_v2 = np.mean(tuple(d_cum_reward_eval_two_agents_data_v2), axis=0)
    d_cum_reward_eval_two_agents_stds_v2 = np.std(tuple(d_cum_reward_eval_two_agents_data_v2), axis=0, ddof=1)

    episode_len_train_two_agents_data_v2 = list(map(lambda df: df["avg_episode_steps"].values, train_two_agents_dfs_v2))
    episode_len_train_two_agents_means_v2 = np.mean(tuple(episode_len_train_two_agents_data_v2), axis=0)
    episode_len_train_two_agents_stds_v2 = np.std(tuple(episode_len_train_two_agents_data_v2), axis=0, ddof=1)
    episode_len_eval_two_agents_data_v2 = list(map(lambda df: df["avg_episode_steps"].values, eval_two_agents_dfs_v2))
    episode_len_eval_two_agents_means_v2 = np.mean(tuple(episode_len_eval_two_agents_data_v2), axis=0)
    episode_len_eval_two_agents_stds_v2 = np.std(tuple(episode_len_eval_two_agents_data_v2), axis=0, ddof=1)

    # V3
    train_max_attack_dfs_v3 = []
    eval_max_attack_dfs_v3 = []
    for csv_path in maximal_attack_train_csv_paths_v3:
        df = read_data(csv_path)
        train_max_attack_dfs_v3.append(df)

    for csv_path in maximal_attack_eval_csv_paths_v3:
        df = read_data(csv_path)
        eval_max_attack_dfs_v3.append(df)

    hack_prob_train_max_attack_data_v3 = list(map(lambda df: df["hack_probability"].values, train_max_attack_dfs_v3))
    hack_prob_train_max_attack_means_v3 = np.mean(tuple(hack_prob_train_max_attack_data_v3), axis=0)
    hack_prob_train_max_attack_stds_v3 = np.std(tuple(hack_prob_train_max_attack_data_v3), axis=0, ddof=1)
    hack_prob_eval_max_attack_data_v3 = list(map(lambda df: df["hack_probability"].values, eval_max_attack_dfs_v3))
    hack_prob_eval_max_attack_means_v3 = np.mean(tuple(hack_prob_eval_max_attack_data_v3), axis=0)
    hack_prob_eval_max_attack_stds_v3 = np.std(tuple(hack_prob_eval_max_attack_data_v3), axis=0, ddof=1)

    a_cum_reward_train_max_attack_data_v3 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, train_max_attack_dfs_v3))
    a_cum_reward_train_max_attack_means_v3 = np.mean(tuple(a_cum_reward_train_max_attack_data_v3), axis=0)
    a_cum_reward_train_max_attack_stds_v3 = np.std(tuple(a_cum_reward_train_max_attack_data_v3), axis=0, ddof=1)
    a_cum_reward_eval_max_attack_data_v3 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, eval_max_attack_dfs_v3))
    a_cum_reward_eval_max_attack_means_v3 = np.mean(tuple(a_cum_reward_eval_max_attack_data_v3), axis=0)
    a_cum_reward_eval_max_attack_stds_v3 = np.std(tuple(a_cum_reward_eval_max_attack_data_v3), axis=0, ddof=1)

    d_cum_reward_train_max_attack_data_v3 = list(
        map(lambda df: df["defender_cumulative_reward"].values, train_max_attack_dfs_v3))
    d_cum_reward_train_max_attack_means_v3 = np.mean(tuple(d_cum_reward_train_max_attack_data_v3), axis=0)
    d_cum_reward_train_max_attack_stds_v3 = np.std(tuple(d_cum_reward_train_max_attack_data_v3), axis=0, ddof=1)
    d_cum_reward_eval_max_attack_data_v3 = list(
        map(lambda df: df["defender_cumulative_reward"].values, eval_max_attack_dfs_v3))
    d_cum_reward_eval_max_attack_means_v3 = np.mean(tuple(d_cum_reward_eval_max_attack_data_v3), axis=0)
    d_cum_reward_eval_max_attack_stds_v3 = np.std(tuple(d_cum_reward_eval_max_attack_data_v3), axis=0, ddof=1)

    episode_len_train_max_attack_data_v3 = list(map(lambda df: df["avg_episode_steps"].values, train_max_attack_dfs_v3))
    episode_len_train_max_attack_means_v3 = np.mean(tuple(episode_len_train_max_attack_data_v3), axis=0)
    episode_len_train_max_attack_stds_v3 = np.std(tuple(episode_len_train_max_attack_data_v3), axis=0, ddof=1)
    episode_len_eval_max_attack_data_v3 = list(map(lambda df: df["avg_episode_steps"].values, eval_max_attack_dfs_v3))
    episode_len_eval_max_attack_means_v3 = np.mean(tuple(episode_len_eval_max_attack_data_v3), axis=0)
    episode_len_eval_max_attack_stds_v3 = np.std(tuple(episode_len_eval_max_attack_data_v3), axis=0, ddof=1)

    train_min_defense_dfs_v3 = []
    eval_min_defense_dfs_v3 = []
    for csv_path in minimal_defense_train_csv_paths_v3:
        df = read_data(csv_path)
        train_min_defense_dfs_v3.append(df)

    for csv_path in minimal_defense_eval_csv_paths_v3:
        df = read_data(csv_path)
        eval_min_defense_dfs_v3.append(df)

    hack_prob_train_min_defense_data_v3 = list(map(lambda df: df["hack_probability"].values, train_min_defense_dfs_v3))
    hack_prob_train_min_defense_means_v3 = np.mean(tuple(hack_prob_train_min_defense_data_v3), axis=0)
    hack_prob_train_min_defense_stds_v3 = np.std(tuple(hack_prob_train_min_defense_data_v3), axis=0, ddof=1)
    hack_prob_eval_min_defense_data_v3 = list(map(lambda df: df["hack_probability"].values, eval_min_defense_dfs_v3))
    hack_prob_eval_min_defense_means_v3 = np.mean(tuple(hack_prob_eval_min_defense_data_v3), axis=0)
    hack_prob_eval_min_defense_stds_v3 = np.std(tuple(hack_prob_eval_min_defense_data_v3), axis=0, ddof=1)

    a_cum_reward_train_min_defense_data_v3 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, train_min_defense_dfs_v3))
    a_cum_reward_train_min_defense_means_v3 = np.mean(tuple(a_cum_reward_train_min_defense_data_v3), axis=0)
    a_cum_reward_train_min_defense_stds_v3 = np.std(tuple(a_cum_reward_train_min_defense_data_v3), axis=0, ddof=1)
    a_cum_reward_eval_min_defense_data_v3 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, eval_min_defense_dfs_v3))
    a_cum_reward_eval_min_defense_means_v3 = np.mean(tuple(a_cum_reward_eval_min_defense_data_v3), axis=0)
    a_cum_reward_eval_min_defense_stds_v3 = np.std(tuple(a_cum_reward_eval_min_defense_data_v3), axis=0, ddof=1)

    d_cum_reward_train_min_defense_data_v3 = list(
        map(lambda df: df["defender_cumulative_reward"].values, train_min_defense_dfs_v3))
    d_cum_reward_train_min_defense_means_v3 = np.mean(tuple(d_cum_reward_train_min_defense_data_v3), axis=0)
    d_cum_reward_train_min_defense_stds_v3 = np.std(tuple(d_cum_reward_train_min_defense_data_v3), axis=0, ddof=1)
    d_cum_reward_eval_min_defense_data_v3 = list(
        map(lambda df: df["defender_cumulative_reward"].values, eval_min_defense_dfs_v3))
    d_cum_reward_eval_min_defense_means_v3 = np.mean(tuple(d_cum_reward_eval_min_defense_data_v3), axis=0)
    d_cum_reward_eval_min_defense_stds_v3 = np.std(tuple(d_cum_reward_eval_min_defense_data_v3), axis=0, ddof=1)

    episode_len_train_min_defense_data_v3 = list(map(lambda df: df["avg_episode_steps"].values, train_min_defense_dfs_v3))
    episode_len_train_min_defense_means_v3 = np.mean(tuple(episode_len_train_min_defense_data_v3), axis=0)
    episode_len_train_min_defense_stds_v3 = np.std(tuple(episode_len_train_min_defense_data_v3), axis=0, ddof=1)
    episode_len_eval_min_defense_data_v3 = list(map(lambda df: df["avg_episode_steps"].values, eval_min_defense_dfs_v3))
    episode_len_eval_min_defense_means_v3 = np.mean(tuple(episode_len_eval_min_defense_data_v3), axis=0)
    episode_len_eval_min_defense_stds_v3 = np.std(tuple(episode_len_eval_min_defense_data_v3), axis=0, ddof=1)

    train_random_attack_dfs_v3 = []
    eval_random_attack_dfs_v3 = []
    for csv_path in random_attack_train_csv_paths_v3:
        df = read_data(csv_path)
        train_random_attack_dfs_v3.append(df)

    for csv_path in random_attack_eval_csv_paths_v3:
        df = read_data(csv_path)
        eval_random_attack_dfs_v3.append(df)

    hack_prob_train_random_attack_data_v3 = list(
        map(lambda df: df["hack_probability"].values, train_random_attack_dfs_v3))
    hack_prob_train_random_attack_means_v3 = np.mean(tuple(hack_prob_train_random_attack_data_v3), axis=0)
    hack_prob_train_random_attack_stds_v3 = np.std(tuple(hack_prob_train_random_attack_data_v3), axis=0, ddof=1)
    hack_prob_eval_random_attack_data_v3 = list(
        map(lambda df: df["hack_probability"].values, eval_random_attack_dfs_v3))
    hack_prob_eval_random_attack_means_v3 = np.mean(tuple(hack_prob_eval_random_attack_data_v3), axis=0)
    hack_prob_eval_random_attack_stds_v3 = np.std(tuple(hack_prob_eval_random_attack_data_v3), axis=0, ddof=1)

    a_cum_reward_train_random_attack_data_v3 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, train_random_attack_dfs_v3))
    a_cum_reward_train_random_attack_means_v3 = np.mean(tuple(a_cum_reward_train_random_attack_data_v3), axis=0)
    a_cum_reward_train_random_attack_stds_v3 = np.std(tuple(a_cum_reward_train_random_attack_data_v3), axis=0, ddof=1)
    a_cum_reward_eval_random_attack_data_v3 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, eval_random_attack_dfs_v3))
    a_cum_reward_eval_random_attack_means_v3 = np.mean(tuple(a_cum_reward_eval_random_attack_data_v3), axis=0)
    a_cum_reward_eval_random_attack_stds_v3 = np.std(tuple(a_cum_reward_eval_random_attack_data_v3), axis=0, ddof=1)

    d_cum_reward_train_random_attack_data_v3 = list(
        map(lambda df: df["defender_cumulative_reward"].values, train_random_attack_dfs_v3))
    d_cum_reward_train_random_attack_means_v3 = np.mean(tuple(d_cum_reward_train_random_attack_data_v3), axis=0)
    d_cum_reward_train_random_attack_stds_v3 = np.std(tuple(d_cum_reward_train_random_attack_data_v3), axis=0, ddof=1)
    d_cum_reward_eval_random_attack_data_v3 = list(
        map(lambda df: df["defender_cumulative_reward"].values, eval_random_attack_dfs_v3))
    d_cum_reward_eval_random_attack_means_v3 = np.mean(tuple(d_cum_reward_eval_random_attack_data_v3), axis=0)
    d_cum_reward_eval_random_attack_stds_v3 = np.std(tuple(d_cum_reward_eval_random_attack_data_v3), axis=0, ddof=1)

    episode_len_train_random_attack_data_v3 = list(
        map(lambda df: df["avg_episode_steps"].values, train_random_attack_dfs_v3))
    episode_len_train_random_attack_means_v3 = np.mean(tuple(episode_len_train_random_attack_data_v3), axis=0)
    episode_len_train_random_attack_stds_v3 = np.std(tuple(episode_len_train_random_attack_data_v3), axis=0, ddof=1)
    episode_len_eval_random_attack_data_v3 = list(
        map(lambda df: df["avg_episode_steps"].values, eval_random_attack_dfs_v3))
    episode_len_eval_random_attack_means_v3 = np.mean(tuple(episode_len_eval_random_attack_data_v3), axis=0)
    episode_len_eval_random_attack_stds_v3 = np.std(tuple(episode_len_eval_random_attack_data_v3), axis=0, ddof=1)

    train_random_defense_dfs_v3 = []
    eval_random_defense_dfs_v3 = []
    for csv_path in random_defense_train_csv_paths_v3:
        df = read_data(csv_path)
        train_random_defense_dfs_v3.append(df)

    for csv_path in random_defense_eval_csv_paths_v3:
        df = read_data(csv_path)
        eval_random_defense_dfs_v3.append(df)

    hack_prob_train_random_defense_data_v3 = list(
        map(lambda df: df["hack_probability"].values, train_random_defense_dfs_v3))
    hack_prob_train_random_defense_means_v3 = np.mean(tuple(hack_prob_train_random_defense_data_v3), axis=0)
    hack_prob_train_random_defense_stds_v3 = np.std(tuple(hack_prob_train_random_defense_data_v3), axis=0, ddof=1)
    hack_prob_eval_random_defense_data_v3 = list(
        map(lambda df: df["hack_probability"].values, eval_random_defense_dfs_v3))
    hack_prob_eval_random_defense_means_v3 = np.mean(tuple(hack_prob_eval_random_defense_data_v3), axis=0)
    hack_prob_eval_random_defense_stds_v3 = np.std(tuple(hack_prob_eval_random_defense_data_v3), axis=0, ddof=1)

    a_cum_reward_train_random_defense_data_v3 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, train_random_defense_dfs_v3))
    a_cum_reward_train_random_defense_means_v3 = np.mean(tuple(a_cum_reward_train_random_defense_data_v3), axis=0)
    a_cum_reward_train_random_defense_stds_v3 = np.std(tuple(a_cum_reward_train_random_defense_data_v3), axis=0, ddof=1)
    a_cum_reward_eval_random_defense_data_v3 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, eval_random_defense_dfs_v3))
    a_cum_reward_eval_random_defense_means_v3 = np.mean(tuple(a_cum_reward_eval_random_defense_data_v3), axis=0)
    a_cum_reward_eval_random_defense_stds_v3 = np.std(tuple(a_cum_reward_eval_random_defense_data_v3), axis=0, ddof=1)

    d_cum_reward_train_random_defense_data_v3 = list(
        map(lambda df: df["defender_cumulative_reward"].values, train_random_defense_dfs_v3))
    d_cum_reward_train_random_defense_means_v3 = np.mean(tuple(d_cum_reward_train_random_defense_data_v3), axis=0)
    d_cum_reward_train_random_defense_stds_v3 = np.std(tuple(d_cum_reward_train_random_defense_data_v3), axis=0, ddof=1)
    d_cum_reward_eval_random_defense_data_v3 = list(
        map(lambda df: df["defender_cumulative_reward"].values, eval_random_defense_dfs_v3))
    d_cum_reward_eval_random_defense_means_v3 = np.mean(tuple(d_cum_reward_eval_random_defense_data_v3), axis=0)
    d_cum_reward_eval_random_defense_stds_v3 = np.std(tuple(d_cum_reward_eval_random_defense_data_v3), axis=0, ddof=1)

    episode_len_train_random_defense_data_v3 = list(
        map(lambda df: df["avg_episode_steps"].values, train_random_defense_dfs_v3))
    episode_len_train_random_defense_means_v3 = np.mean(tuple(episode_len_train_random_defense_data_v3), axis=0)
    episode_len_train_random_defense_stds_v3 = np.std(tuple(episode_len_train_random_defense_data_v3), axis=0, ddof=1)
    episode_len_eval_random_defense_data_v3 = list(
        map(lambda df: df["avg_episode_steps"].values, eval_random_defense_dfs_v3))
    episode_len_eval_random_defense_means_v3 = np.mean(tuple(episode_len_eval_random_defense_data_v3), axis=0)
    episode_len_eval_random_defense_stds_v3 = np.std(tuple(episode_len_eval_random_defense_data_v3), axis=0, ddof=1)

    train_two_agents_dfs_v3 = []
    eval_two_agents_dfs_v3 = []
    for csv_path in two_agents_train_csv_paths_v3:
        df = read_data(csv_path)
        train_two_agents_dfs_v3.append(df)

    for csv_path in two_agents_eval_csv_paths_v3:
        df = read_data(csv_path)
        eval_two_agents_dfs_v3.append(df)

    hack_prob_train_two_agents_data_v3 = list(map(lambda df: df["hack_probability"].values, train_two_agents_dfs_v3))
    hack_prob_train_two_agents_means_v3 = np.mean(tuple(hack_prob_train_two_agents_data_v3), axis=0)
    hack_prob_train_two_agents_stds_v3 = np.std(tuple(hack_prob_train_two_agents_data_v3), axis=0, ddof=1)
    hack_prob_eval_two_agents_data_v3 = list(map(lambda df: df["hack_probability"].values, eval_two_agents_dfs_v3))
    hack_prob_eval_two_agents_means_v3 = np.mean(tuple(hack_prob_eval_two_agents_data_v3), axis=0)
    hack_prob_eval_two_agents_stds_v3 = np.std(tuple(hack_prob_eval_two_agents_data_v3), axis=0, ddof=1)

    a_cum_reward_train_two_agents_data_v3 = list(
        map(lambda df: df["attacker_cumulative_reward"].values, train_two_agents_dfs_v3))
    a_cum_reward_train_two_agents_means_v3 = np.mean(tuple(a_cum_reward_train_two_agents_data_v3), axis=0)
    a_cum_reward_train_two_agents_stds_v3 = np.std(tuple(a_cum_reward_train_two_agents_data_v3), axis=0, ddof=1)
    a_cum_reward_eval_two_agents_data = list(
        map(lambda df: df["attacker_cumulative_reward"].values, eval_two_agents_dfs_v3))
    a_cum_reward_eval_two_agents_means_v3 = np.mean(tuple(a_cum_reward_eval_two_agents_data), axis=0)
    a_cum_reward_eval_two_agents_stds_v3 = np.std(tuple(a_cum_reward_eval_two_agents_data), axis=0, ddof=1)

    d_cum_reward_train_two_agents_data_v3 = list(
        map(lambda df: df["defender_cumulative_reward"].values, train_two_agents_dfs_v3))
    d_cum_reward_train_two_agents_means_v3 = np.mean(tuple(d_cum_reward_train_two_agents_data_v3), axis=0)
    d_cum_reward_train_two_agents_stds_v3 = np.std(tuple(d_cum_reward_train_two_agents_data_v3), axis=0, ddof=1)
    d_cum_reward_eval_two_agents_data_v3 = list(
        map(lambda df: df["defender_cumulative_reward"].values, eval_two_agents_dfs_v3))
    d_cum_reward_eval_two_agents_means_v3 = np.mean(tuple(d_cum_reward_eval_two_agents_data_v3), axis=0)
    d_cum_reward_eval_two_agents_stds_v3 = np.std(tuple(d_cum_reward_eval_two_agents_data_v3), axis=0, ddof=1)

    episode_len_train_two_agents_data_v3 = list(map(lambda df: df["avg_episode_steps"].values, train_two_agents_dfs_v3))
    episode_len_train_two_agents_means_v3 = np.mean(tuple(episode_len_train_two_agents_data_v3), axis=0)
    episode_len_train_two_agents_stds_v3 = np.std(tuple(episode_len_train_two_agents_data_v3), axis=0, ddof=1)
    episode_len_eval_two_agents_data_v3 = list(map(lambda df: df["avg_episode_steps"].values, eval_two_agents_dfs_v3))
    episode_len_eval_two_agents_means_v3 = np.mean(tuple(episode_len_eval_two_agents_data_v3), axis=0)
    episode_len_eval_two_agents_stds_v3 = np.std(tuple(episode_len_eval_two_agents_data_v3), axis=0, ddof=1)

    plot_all_avg_summary_3(np.array(list(range(len(hack_prob_train_min_defense_data_v0[0])))) * train_log_freq,
                           hack_prob_train_min_defense_means_v0,
                           np.array(list(range(len(hack_prob_train_random_defense_data_v0[0])))) * train_log_freq,
                           hack_prob_train_random_defense_means_v0,
                           np.array(list(range(len(hack_prob_train_max_attack_data_v0[0])))) * train_log_freq,
                           hack_prob_train_max_attack_means_v0,
                           np.array(list(range(len(hack_prob_train_random_attack_data_v0[0])))) * train_log_freq,
                           hack_prob_train_random_attack_means_v0,
                           np.array(list(range(len(hack_prob_train_two_agents_data_v0[0])))) * train_log_freq,
                           hack_prob_train_two_agents_means_v0, hack_prob_train_min_defense_stds_v0,
                           hack_prob_train_random_defense_stds_v0, hack_prob_train_max_attack_stds_v0,
                           hack_prob_train_random_attack_stds_v0, hack_prob_train_two_agents_stds_v0,
                           r"\textsc{TabularQLearning} vs \textsc{MinDefense}", r"\textsc{TabularQLearning} vs \textsc{RandomDefense}",
                           r"\textsc{MaxAttack} vs \textsc{TabularQLearning}", r"\textsc{RandomAttack} vs \textsc{TabularQLearning}",
                           r"\textsc{TabularQLearning} vs \textsc{TabularQLearning}", r"Hack probability (train, v" + str(versions[0]) + ")",
                           r"Episode \#", r"$\mathbb{P}[Hacked]$", eval_freq, eval_freq, eval_freq, eval_freq, eval_freq,

                           np.array(list(range(len(hack_prob_eval_min_defense_data_v0[0])))) * eval_freq,
                           hack_prob_eval_min_defense_means_v0,
                           np.array(list(range(len(hack_prob_eval_random_defense_data_v0[0])))) * eval_freq,
                           hack_prob_eval_random_defense_means_v0,
                           np.array(list(range(len(hack_prob_eval_max_attack_data_v0[0])))) * eval_freq,
                           hack_prob_eval_max_attack_means_v0,
                           np.array(list(range(len(hack_prob_eval_random_attack_data_v0[0])))) * eval_freq,
                           hack_prob_eval_random_attack_means_v0,
                           np.array(list(range(len(hack_prob_eval_two_agents_data_v0[0])))) * eval_freq,
                           hack_prob_eval_two_agents_means_v0, hack_prob_eval_min_defense_stds_v0,
                           hack_prob_eval_random_defense_stds_v0, hack_prob_eval_max_attack_stds_v0,
                           hack_prob_eval_random_attack_stds_v0, hack_prob_eval_two_agents_stds_v0,
                           r"\textsc{TabularQLearning} vs \textsc{MinDefense}", r"\textsc{TabularQLearning} vs \textsc{RandomDefense}",
                           r"\textsc{MaxAttack} vs \textsc{TabularQLearning}", r"\textsc{RandomAttack} vs \textsc{TabularQLearning}",
                           r"\textsc{TabularQLearning} vs \textsc{TabularQLearning}", r"Hack probability (eval v" + str(versions[0]) + ")",
                           r"Episode \#", r"$\mathbb{P}[Hacked]$", 1, 1, 1, 1, 1,

                           np.array(list(range(len(a_cum_reward_train_min_defense_data_v0[0])))) * train_log_freq,
                           a_cum_reward_train_min_defense_means_v0,
                           np.array(list(range(len(a_cum_reward_train_random_defense_data_v0[0])))) * train_log_freq,
                           a_cum_reward_train_random_defense_means_v0,
                           np.array(list(range(len(a_cum_reward_train_max_attack_data_v0[0])))) * train_log_freq,
                           a_cum_reward_train_max_attack_means_v0,
                           np.array(list(range(len(a_cum_reward_train_random_attack_data_v0[0])))) * train_log_freq,
                           a_cum_reward_train_random_attack_means_v0,
                           np.array(list(range(len(a_cum_reward_train_two_agents_data_v0[0])))) * train_log_freq,
                           a_cum_reward_train_two_agents_means_v0, a_cum_reward_train_min_defense_stds_v0,
                           a_cum_reward_train_random_defense_stds_v0, a_cum_reward_train_max_attack_stds_v0,
                           a_cum_reward_train_random_attack_stds_v0, a_cum_reward_train_two_agents_stds_v0,
                           r"\textsc{TabularQLearning} vs \textsc{MinDefense}", r"\textsc{TabularQLearning} vs \textsc{RandomDefense}",
                           r"\textsc{MaxAttack} vs \textsc{TabularQLearning}", r"\textsc{RandomAttack} vs \textsc{TabularQLearning}",
                           r"\textsc{TabularQLearning} vs \textsc{TabularQLearning}", r"Attacker reward (train v" + str(versions[0]) + ")",
                           r"Episode \#", r"Cumulative Reward", eval_freq, eval_freq, eval_freq, eval_freq, eval_freq,


                           np.array(list(range(len(hack_prob_train_min_defense_data_v2[0])))) * train_log_freq,
                           hack_prob_train_min_defense_means_v2,
                           np.array(list(range(len(hack_prob_train_random_defense_data_v2[0])))) * train_log_freq,
                           hack_prob_train_random_defense_means_v2,
                           np.array(list(range(len(hack_prob_train_max_attack_data_v2[0])))) * train_log_freq,
                           hack_prob_train_max_attack_means_v2,
                           np.array(list(range(len(hack_prob_train_random_attack_data_v2[0])))) * train_log_freq,
                           hack_prob_train_random_attack_means_v2,
                           np.array(list(range(len(hack_prob_train_two_agents_data_v2[0])))) * train_log_freq,
                           hack_prob_train_two_agents_means_v2, hack_prob_train_min_defense_stds_v2,
                           hack_prob_train_random_defense_stds_v2, hack_prob_train_max_attack_stds_v2,
                           hack_prob_train_random_attack_stds_v2, hack_prob_train_two_agents_stds_v2,
                           r"\textsc{TabularQLearning} vs \textsc{MinDefense}", r"\textsc{TabularQLearning} vs \textsc{RandomDefense}",
                           r"\textsc{MaxAttack} vs \textsc{TabularQLearning}", r"\textsc{RandomAttack} vs \textsc{TabularQLearning}",
                           r"\textsc{TabularQLearning} vs \textsc{TabularQLearning}", r"Hack probability (train, v" + str(versions[1]) + ")",
                           r"Episode \#", r"$\mathbb{P}[Hacked]$", eval_freq, eval_freq, eval_freq, eval_freq, eval_freq,

                           np.array(list(range(len(hack_prob_eval_min_defense_data_v2[0])))) * eval_freq,
                           hack_prob_eval_min_defense_means_v2,
                           np.array(list(range(len(hack_prob_eval_random_defense_data_v2[0])))) * eval_freq,
                           hack_prob_eval_random_defense_means_v2,
                           np.array(list(range(len(hack_prob_eval_max_attack_data_v2[0])))) * eval_freq,
                           hack_prob_eval_max_attack_means_v2,
                           np.array(list(range(len(hack_prob_eval_random_attack_data_v2[0])))) * eval_freq,
                           hack_prob_eval_random_attack_means_v2,
                           np.array(list(range(len(hack_prob_eval_two_agents_data_v2[0])))) * eval_freq,
                           hack_prob_eval_two_agents_means_v2, hack_prob_eval_min_defense_stds_v2,
                           hack_prob_eval_random_defense_stds_v2, hack_prob_eval_max_attack_stds_v2,
                           hack_prob_eval_random_attack_stds_v2, hack_prob_eval_two_agents_stds_v2,
                           r"\textsc{TabularQLearning} vs \textsc{MinDefense}", r"\textsc{TabularQLearning} vs \textsc{RandomDefense}",
                           r"\textsc{MaxAttack} vs \textsc{TabularQLearning}", r"\textsc{RandomAttack} vs \textsc{TabularQLearning}",
                           r"\textsc{TabularQLearning} vs \textsc{TabularQLearning}", r"Hack probability (eval v" + str(versions[1]) + ")",
                           r"Episode \#", r"$\mathbb{P}[Hacked]$", 1, 1, 1, 1, 1,

                           np.array(list(range(len(a_cum_reward_train_min_defense_data_v2[0])))) * train_log_freq,
                           a_cum_reward_train_min_defense_means_v2,
                           np.array(list(range(len(a_cum_reward_train_random_defense_data_v2[0])))) * train_log_freq,
                           a_cum_reward_train_random_defense_means_v2,
                           np.array(list(range(len(a_cum_reward_train_max_attack_data_v2[0])))) * train_log_freq,
                           a_cum_reward_train_max_attack_means_v2,
                           np.array(list(range(len(a_cum_reward_train_random_attack_data_v2[0])))) * train_log_freq,
                           a_cum_reward_train_random_attack_means_v2,
                           np.array(list(range(len(a_cum_reward_train_two_agents_data_v2[0])))) * train_log_freq,
                           a_cum_reward_train_two_agents_means_v2, a_cum_reward_train_min_defense_stds_v2,
                           a_cum_reward_train_random_defense_stds_v2, a_cum_reward_train_max_attack_stds_v2,
                           a_cum_reward_train_random_attack_stds_v2, a_cum_reward_train_two_agents_stds_v2,
                           r"\textsc{TabularQLearning} vs \textsc{MinDefense}", r"\textsc{TabularQLearning} vs \textsc{RandomDefense}",
                           r"\textsc{MaxAttack} vs \textsc{TabularQLearning}", r"\textsc{RandomAttack} vs \textsc{TabularQLearning}",
                           r"\textsc{TabularQLearning} vs \textsc{TabularQLearning}", r"Attacker reward (train v" + str(versions[1]) + ")",
                           r"Episode \#", r"Cumulative Reward", eval_freq, eval_freq, eval_freq, eval_freq, eval_freq,

                           np.array(list(range(len(hack_prob_train_min_defense_data_v3[0])))) * train_log_freq,
                           hack_prob_train_min_defense_means_v3,
                           np.array(list(range(len(hack_prob_train_random_defense_data_v3[0])))) * train_log_freq,
                           hack_prob_train_random_defense_means_v3,
                           np.array(list(range(len(hack_prob_train_max_attack_data_v3[0])))) * train_log_freq,
                           hack_prob_train_max_attack_means_v3,
                           np.array(list(range(len(hack_prob_train_random_attack_data_v3[0])))) * train_log_freq,
                           hack_prob_train_random_attack_means_v3,
                           np.array(list(range(len(hack_prob_train_two_agents_data_v3[0])))) * train_log_freq,
                           hack_prob_train_two_agents_means_v3, hack_prob_train_min_defense_stds_v3,
                           hack_prob_train_random_defense_stds_v3, hack_prob_train_max_attack_stds_v3,
                           hack_prob_train_random_attack_stds_v3, hack_prob_train_two_agents_stds_v3,
                           r"\textsc{TabularQLearning} vs \textsc{MinDefense}", r"\textsc{TabularQLearning} vs \textsc{RandomDefense}",
                           r"\textsc{MaxAttack} vs \textsc{TabularQLearning}", r"\textsc{RandomAttack} vs \textsc{TabularQLearning}",
                           r"\textsc{TabularQLearning} vs \textsc{TabularQLearning}", r"Hack probability (train, v" + str(versions[2]) + ")",
                           r"Episode \#", r"$\mathbb{P}[Hacked]$", eval_freq, eval_freq, eval_freq, eval_freq, eval_freq,

                           np.array(list(range(len(hack_prob_eval_min_defense_data_v3[0])))) * eval_freq,
                           hack_prob_eval_min_defense_means_v3,
                           np.array(list(range(len(hack_prob_eval_random_defense_data_v3[0])))) * eval_freq,
                           hack_prob_eval_random_defense_means_v3,
                           np.array(list(range(len(hack_prob_eval_max_attack_data_v3[0])))) * eval_freq,
                           hack_prob_eval_max_attack_means_v3,
                           np.array(list(range(len(hack_prob_eval_random_attack_data_v3[0])))) * eval_freq,
                           hack_prob_eval_random_attack_means_v3,
                           np.array(list(range(len(hack_prob_eval_two_agents_data_v3[0])))) * eval_freq,
                           hack_prob_eval_two_agents_means_v3, hack_prob_eval_min_defense_stds_v3,
                           hack_prob_eval_random_defense_stds_v3, hack_prob_eval_max_attack_stds_v3,
                           hack_prob_eval_random_attack_stds_v3, hack_prob_eval_two_agents_stds_v3,
                           r"\textsc{TabularQLearning} vs \textsc{MinDefense}", r"\textsc{TabularQLearning} vs \textsc{RandomDefense}",
                           r"\textsc{MaxAttack} vs \textsc{TabularQLearning}", r"\textsc{RandomAttack} vs \textsc{TabularQLearning}",
                           r"\textsc{TabularQLearning} vs \textsc{TabularQLearning}", r"Hack probability (eval v" + str(versions[2]) + ")",
                           r"Episode \#", r"$\mathbb{P}[Hacked]$", 1, 1, 1, 1, 1,

                           np.array(list(range(len(a_cum_reward_train_min_defense_data_v3[0])))) * train_log_freq,
                           a_cum_reward_train_min_defense_means_v3,
                           np.array(list(range(len(a_cum_reward_train_random_defense_data_v3[0])))) * train_log_freq,
                           a_cum_reward_train_random_defense_means_v3,
                           np.array(list(range(len(a_cum_reward_train_max_attack_data_v3[0])))) * train_log_freq,
                           a_cum_reward_train_max_attack_means_v3,
                           np.array(list(range(len(a_cum_reward_train_random_attack_data_v3[0])))) * train_log_freq,
                           a_cum_reward_train_random_attack_means_v3,
                           np.array(list(range(len(a_cum_reward_train_two_agents_data_v3[0])))) * train_log_freq,
                           a_cum_reward_train_two_agents_means_v3, a_cum_reward_train_min_defense_stds_v3,
                           a_cum_reward_train_random_defense_stds_v3, a_cum_reward_train_max_attack_stds_v3,
                           a_cum_reward_train_random_attack_stds_v3, a_cum_reward_train_two_agents_stds_v3,
                           r'\textsc{TabularQLearning} vs \textsc{MinDefense}', r"\textsc{TabularQLearning} vs \textsc{RandomDefense}",
                           r"\textsc{MaxAttack} vs \textsc{TabularQLearning}", r"\textsc{RandomAttack} vs \textsc{TabularQLearning}",
                           r"\textsc{TabularQLearning} vs \textsc{TabularQLearning}", r"Attacker reward (train v" + str(versions[2]) + ")",
                           r"Episode \#", r"Cumulative Reward", eval_freq, eval_freq, eval_freq, eval_freq, eval_freq,


                           output_dir + "/" + file_name + algorithm + "_" + str(0),
                           wspace=wspace
                           )
    five_line_plot_w_shades(np.array(list(range(len(episode_len_eval_max_attack_data_v0[0])))) * eval_freq,
                            episode_len_eval_max_attack_means_v0,
                            np.array(list(range(len(episode_len_eval_min_defense_data_v0[0])))) * eval_freq,
                            episode_len_eval_min_defense_means_v0,
                            np.array(list(range(len(episode_len_eval_random_attack_data_v0[0])))) * eval_freq,
                            episode_len_eval_random_attack_means_v0,
                            np.array(list(range(len(episode_len_eval_random_defense_data_v0[0])))) * eval_freq,
                            episode_len_eval_random_defense_means_v0,
                            np.array(list(range(len(episode_len_eval_two_agents_data_v0[0])))) * eval_freq,
                            episode_len_eval_two_agents_means_v0, stds_1=episode_len_eval_max_attack_stds_v0,
                            stds_2=episode_len_eval_min_defense_stds_v0, stds_3=episode_len_eval_random_attack_stds_v0,
                            stds_4=episode_len_eval_random_defense_stds_v0, stds_5=episode_len_eval_two_agents_stds_v0,
                            title=r"Avg Episode Lengths [Eval] (v" + str(versions[0]) + ")",
                            xlabel=r"Episode \#", ylabel=r"Avg Length (num steps)",
                            file_name=output_dir + "/avg_episode_length_eval_" + algorithm + "_" + str(0),
                            line1_label=r"\textsc{MaxAttack} vs \textsc{TabularQLearning}",
                            line2_label=r"\textsc{TabularQLearning} vs \textsc{MinDefense}",
                            line3_label=r"\textsc{RandomAttack} vs \textsc{TabularQLearning}",
                            line4_label=r"\textsc{TabularQLearning} vs \textsc{RandomDefense}",
                            line5_label=r"\textsc{TabularQLearning} vs \textsc{TabularQLearning}", legend_loc="lower right",
                            markevery_1=1, markevery_2=1, markevery_3=1,
                            markevery_4=1, markevery_5=1)

    five_line_plot_w_shades(np.array(list(range(len(episode_len_eval_max_attack_data_v2[0])))) * eval_freq,
                            episode_len_eval_max_attack_means_v2,
                            np.array(list(range(len(episode_len_eval_min_defense_data_v2[0])))) * eval_freq,
                            episode_len_eval_min_defense_means_v2,
                            np.array(list(range(len(episode_len_eval_random_attack_data_v2[0])))) * eval_freq,
                            episode_len_eval_random_attack_means_v2,
                            np.array(list(range(len(episode_len_eval_random_defense_data_v2[0])))) * eval_freq,
                            episode_len_eval_random_defense_means_v2,
                            np.array(list(range(len(episode_len_eval_two_agents_data_v2[0])))) * eval_freq,
                            episode_len_eval_two_agents_means_v2, stds_1=episode_len_eval_max_attack_stds_v2,
                            stds_2=episode_len_eval_min_defense_stds_v2, stds_3=episode_len_eval_random_attack_stds_v2,
                            stds_4=episode_len_eval_random_defense_stds_v2, stds_5=episode_len_eval_two_agents_stds_v2,
                            title=r"Avg Episode Lengths [Eval] (v" + str(versions[1]) + ")",
                            xlabel=r"Episode \#", ylabel=r"Avg Length (num steps)",
                            file_name=output_dir + "/avg_episode_length_eval_" + algorithm + "_" + str(1),
                            line1_label=r"\textsc{MaxAttack} vs \textsc{TabularQLearning}",
                            line2_label=r"\textsc{TabularQLearning} vs \textsc{MinDefense}",
                            line3_label=r"\textsc{RandomAttack} vs \textsc{TabularQLearning}",
                            line4_label=r"\textsc{TabularQLearning} vs \textsc{RandomDefense}",
                            line5_label=r"\textsc{TabularQLearning} vs \textsc{TabularQLearning}", legend_loc="lower right",
                            markevery_1=1, markevery_2=1, markevery_3=1,
                            markevery_4=1, markevery_5=1)

    five_line_plot_w_shades(np.array(list(range(len(episode_len_eval_max_attack_data_v3[0])))) * eval_freq,
                            episode_len_eval_max_attack_means_v3,
                            np.array(list(range(len(episode_len_eval_min_defense_data_v3[0])))) * eval_freq,
                            episode_len_eval_min_defense_means_v3,
                            np.array(list(range(len(episode_len_eval_random_attack_data_v3[0])))) * eval_freq,
                            episode_len_eval_random_attack_means_v3,
                            np.array(list(range(len(episode_len_eval_random_defense_data_v3[0])))) * eval_freq,
                            episode_len_eval_random_defense_means_v3,
                            np.array(list(range(len(episode_len_eval_two_agents_data_v3[0])))) * eval_freq,
                            episode_len_eval_two_agents_means_v3, stds_1=episode_len_eval_max_attack_stds_v3,
                            stds_2=episode_len_eval_min_defense_stds_v3, stds_3=episode_len_eval_random_attack_stds_v3,
                            stds_4=episode_len_eval_random_defense_stds_v3, stds_5=episode_len_eval_two_agents_stds_v3,
                            title=r"Avg Episode Lengths [Eval] (v" + str(versions[2]) + ")",
                            xlabel=r"Episode \#", ylabel=r"Avg Length (num steps)",
                            file_name=output_dir + "/avg_episode_length_eval_" + algorithm + "_" + str(2),
                            line1_label=r"\textsc{MaxAttack} vs \textsc{TabularQLearning}",
                            line2_label=r"\textsc{TabularQLearning} vs \textsc{MinDefense}",
                            line3_label=r"\textsc{RandomAttack} vs \textsc{TabularQLearning}",
                            line4_label=r"\textsc{TabularQLearning} vs \textsc{RandomDefense}",
                            line5_label=r"\textsc{TabularQLearning} vs \textsc{TabularQLearning}",
                            legend_loc="lower right",
                            markevery_1=1, markevery_2=1, markevery_3=1,
                            markevery_4=1, markevery_5=1)

    plot_all_avg_summary_2(np.array(list(range(len(episode_len_eval_min_defense_data_v0[0])))) * eval_freq,
                           episode_len_eval_min_defense_means_v0,
                           np.array(list(range(len(episode_len_eval_random_defense_data_v0[0])))) * eval_freq,
                           episode_len_eval_random_defense_means_v0,
                           np.array(list(range(len(episode_len_eval_max_attack_data_v0[0])))) * eval_freq,
                           episode_len_eval_max_attack_means_v0,
                           np.array(list(range(len(episode_len_eval_random_attack_data_v0[0])))) * eval_freq,
                           episode_len_eval_random_attack_means_v0,
                           np.array(list(range(len(episode_len_eval_two_agents_data_v0[0])))) * eval_freq,
                           episode_len_eval_two_agents_means_v0, episode_len_eval_min_defense_stds_v0,
                           episode_len_eval_random_defense_stds_v0, episode_len_eval_max_attack_stds_v0,
                           episode_len_eval_random_attack_stds_v0, episode_len_eval_two_agents_stds_v0,
                           r"\textsc{TabularQLearning} vs \textsc{MinDefense}", r"\textsc{TabularQLearning} vs \textsc{RandomDefense}",
                           r"\textsc{MaxAttack} vs \textsc{TabularQLearning}", r"\textsc{RandomAttack} vs \textsc{TabularQLearning}",
                           r"\textsc{TabularQLearning} vs \textsc{TabularQLearning}", r"Avg Episode Lengths [Eval] (v" + str(versions[0]) + ")",
                           r"Episode \#", r"Avg Length (num steps)", 1, 1, 1, 1, 1,

                           np.array(list(range(len(episode_len_eval_min_defense_data_v2[0])))) * eval_freq,
                           episode_len_eval_min_defense_means_v2,
                           np.array(list(range(len(episode_len_eval_random_defense_data_v2[0])))) * eval_freq,
                           episode_len_eval_random_defense_means_v2,
                           np.array(list(range(len(episode_len_eval_max_attack_data_v2[0])))) * eval_freq,
                           episode_len_eval_max_attack_means_v2,
                           np.array(list(range(len(episode_len_eval_random_attack_data_v2[0])))) * eval_freq,
                           episode_len_eval_random_attack_means_v2,
                           np.array(list(range(len(episode_len_eval_two_agents_data_v2[0])))) * eval_freq,
                           episode_len_eval_two_agents_means_v2, episode_len_eval_min_defense_stds_v2,
                           episode_len_eval_random_defense_stds_v2, episode_len_eval_max_attack_stds_v2,
                           episode_len_eval_random_attack_stds_v2, episode_len_eval_two_agents_stds_v2,
                           r"\textsc{TabularQLearning} vs \textsc{MinDefense}", r"\textsc{TabularQLearning} vs \textsc{RandomDefense}",
                           r"\textsc{MaxAttack} vs \textsc{TabularQLearning}", r"\textsc{RandomAttack} vs \textsc{TabularQLearning}",
                           r"\textsc{TabularQLearning} vs \textsc{TabularQLearning}", r"Avg Episode Lengths [Eval] (v" + str(versions[1]) + ")",
                           r"Episode \#", r"Avg Length (num steps)", 1, 1, 1, 1, 1,

                           np.array(list(range(len(episode_len_eval_min_defense_data_v3[0])))) * eval_freq,
                           episode_len_eval_min_defense_means_v3,
                           np.array(list(range(len(episode_len_eval_random_defense_data_v3[0])))) * eval_freq,
                           episode_len_eval_random_defense_means_v3,
                           np.array(list(range(len(episode_len_eval_max_attack_data_v3[0])))) * eval_freq,
                           episode_len_eval_max_attack_means_v3,
                           np.array(list(range(len(episode_len_eval_random_attack_data_v3[0])))) * eval_freq,
                           episode_len_eval_random_attack_means_v3,
                           np.array(list(range(len(episode_len_eval_two_agents_data_v3[0])))) * eval_freq,
                           episode_len_eval_two_agents_means_v3, episode_len_eval_min_defense_stds_v3,
                           episode_len_eval_random_defense_stds_v3, episode_len_eval_max_attack_stds_v3,
                           episode_len_eval_random_attack_stds_v3, episode_len_eval_two_agents_stds_v3,
                           r"\textsc{TabularQLearning} vs \textsc{MinDefense}", r"\textsc{TabularQLearning} vs \textsc{RandomDefense}",
                           r"\textsc{MaxAttack} vs \textsc{TabularQLearning}", r"\textsc{RandomAttack} vs \textsc{TabularQLearning}",
                           r"\textsc{TabularQLearning} vs \textsc{TabularQLearning}", r"Avg Episode Lengths [Eval] (v" + str(versions[2]) + ")",
                           r"Episode \#", r"Avg Length (num steps)", 1, 1, 1, 1, 1,

                           np.array(list(range(len(episode_len_eval_min_defense_data_v3[0])))) * eval_freq,
                           episode_len_eval_min_defense_means_v3,
                           np.array(list(range(len(episode_len_eval_random_defense_data_v3[0])))) * eval_freq,
                           episode_len_eval_random_defense_means_v3,
                           np.array(list(range(len(episode_len_eval_max_attack_data_v3[0])))) * eval_freq,
                           episode_len_eval_max_attack_means_v3,
                           np.array(list(range(len(episode_len_eval_random_attack_data_v3[0])))) * eval_freq,
                           episode_len_eval_random_attack_means_v3,
                           np.array(list(range(len(episode_len_eval_two_agents_data_v3[0])))) * eval_freq,
                           episode_len_eval_two_agents_means_v3, episode_len_eval_min_defense_stds_v3,
                           episode_len_eval_random_defense_stds_v3, episode_len_eval_max_attack_stds_v3,
                           episode_len_eval_random_attack_stds_v3, episode_len_eval_two_agents_stds_v3,
                           r"\textsc{TabularQLearning} vs \textsc{MinDefense}", r"\textsc{TabularQLearning} vs \textsc{RandomDefense}",
                           r"\textsc{MaxAttack} vs \textsc{TabularQLearning}", r"\textsc{RandomAttack} vs \textsc{TabularQLearning}",
                           r"\textsc{TabularQLearning} vs \textsc{TabularQLearning}", r"Avg Episode Lengths [Eval] (v" + str(versions[2]) + ")",
                           r"Episode \#", r"Avg Length (num steps)", 1, 1, 1, 1, 1,

                           output_dir + "/" + file_name + "_avg_length_" + algorithm
                           )

    # five_line_plot_w_shades(np.array(list(range(len(episode_len_train_max_attack_data_v0[0])))) * train_log_freq,
    #                         episode_len_train_max_attack_means_v0,
    #                         np.array(list(range(len(episode_len_train_min_defense_data_v0[0])))) * train_log_freq,
    #                         episode_len_train_min_defense_means_v0,
    #                         np.array(list(range(len(episode_len_train_random_attack_data_v0[0])))) * train_log_freq,
    #                         episode_len_train_random_attack_means_v0,
    #                         np.array(list(range(len(episode_len_train_random_defense_data_v0[0])))) * train_log_freq,
    #                         episode_len_train_random_defense_means_v0,
    #                         np.array(list(range(len(episode_len_train_two_agents_data_v0[0])))) * train_log_freq,
    #                         episode_len_train_two_agents_means_v0, stds_1=episode_len_train_max_attack_stds_v0,
    #                         stds_2=episode_len_train_min_defense_stds_v0, stds_3=episode_len_train_random_attack_stds_v0,
    #                         stds_4=episode_len_train_random_defense_stds_v0, stds_5=episode_len_train_two_agents_stds_v0,
    #                         title="Avg Episode Lengths [Train] (v" + str(0) + ")",
    #                         xlabel="Episode \#", ylabel="Avg Length (num steps)",
    #                         file_name=output_dir + "/avg_episode_length_train_" + algorithm + "_" + str(0),
    #                         line1_label="maximal attack vs Q-learning", line2_label="Q-learning vs minimal defense",
    #                         line3_label="\textsc{RandomAttack} vs Q-learning", line4_label="Q-learning vs random defense",
    #                         line5_label="Q-learning vs Q-learning", legend_loc="lower right",
    #                         markevery_1=eval_freq, markevery_2=eval_freq, markevery_3=eval_freq,
    #                         markevery_4=eval_freq, markevery_5=eval_freq)



def plot_all_averages(maximal_attack_train_csv_paths, maximal_attack_eval_csv_paths,
                      minimal_defense_train_csv_paths, minimal_defense_eval_csv_paths,
                      random_attack_train_csv_paths, random_attack_eval_csv_paths,
                      random_defense_train_csv_paths, random_defense_eval_csv_paths,
                      two_agents_train_csv_paths, two_agents_eval_csv_paths,
                      version, algorithm, output_dir, eval_freq : int, train_log_freq : int):
    train_max_attack_dfs = []
    eval_max_attack_dfs = []
    for csv_path in maximal_attack_train_csv_paths:
        df = read_data(csv_path)
        train_max_attack_dfs.append(df)

    for csv_path in maximal_attack_eval_csv_paths:
        df = read_data(csv_path)
        eval_max_attack_dfs.append(df)

    hack_prob_train_max_attack_data = list(map(lambda df: df["hack_probability"].values, train_max_attack_dfs))
    hack_prob_train_max_attack_means = np.mean(tuple(hack_prob_train_max_attack_data), axis=0)
    hack_prob_train_max_attack_stds = np.std(tuple(hack_prob_train_max_attack_data), axis=0, ddof=1)
    hack_prob_eval_max_attack_data = list(map(lambda df: df["hack_probability"].values, eval_max_attack_dfs))
    hack_prob_eval_max_attack_means = np.mean(tuple(hack_prob_eval_max_attack_data), axis=0)
    hack_prob_eval_max_attack_stds = np.std(tuple(hack_prob_eval_max_attack_data), axis=0, ddof=1)

    a_cum_reward_train_max_attack_data = list(map(lambda df: df["attacker_cumulative_reward"].values, train_max_attack_dfs))
    a_cum_reward_train_max_attack_means = np.mean(tuple(a_cum_reward_train_max_attack_data), axis=0)
    a_cum_reward_train_max_attack_stds = np.std(tuple(a_cum_reward_train_max_attack_data), axis=0, ddof=1)
    a_cum_reward_eval_max_attack_data = list(map(lambda df: df["attacker_cumulative_reward"].values, eval_max_attack_dfs))
    a_cum_reward_eval_max_attack_means = np.mean(tuple(a_cum_reward_eval_max_attack_data), axis=0)
    a_cum_reward_eval_max_attack_stds = np.std(tuple(a_cum_reward_eval_max_attack_data), axis=0, ddof=1)

    d_cum_reward_train_max_attack_data = list(
        map(lambda df: df["defender_cumulative_reward"].values, train_max_attack_dfs))
    d_cum_reward_train_max_attack_means = np.mean(tuple(d_cum_reward_train_max_attack_data), axis=0)
    d_cum_reward_train_max_attack_stds = np.std(tuple(d_cum_reward_train_max_attack_data), axis=0, ddof=1)
    d_cum_reward_eval_max_attack_data = list(
        map(lambda df: df["defender_cumulative_reward"].values, eval_max_attack_dfs))
    d_cum_reward_eval_max_attack_means = np.mean(tuple(d_cum_reward_eval_max_attack_data), axis=0)
    d_cum_reward_eval_max_attack_stds = np.std(tuple(d_cum_reward_eval_max_attack_data), axis=0, ddof=1)

    train_min_defense_dfs = []
    eval_min_defense_dfs = []
    for csv_path in minimal_defense_train_csv_paths:
        df = read_data(csv_path)
        train_min_defense_dfs.append(df)

    for csv_path in minimal_defense_eval_csv_paths:
        df = read_data(csv_path)
        eval_min_defense_dfs.append(df)

    hack_prob_train_min_defense_data = list(map(lambda df: df["hack_probability"].values, train_min_defense_dfs))
    hack_prob_train_min_defense_means = np.mean(tuple(hack_prob_train_min_defense_data), axis=0)
    hack_prob_train_min_defense_stds = np.std(tuple(hack_prob_train_min_defense_data), axis=0, ddof=1)
    hack_prob_eval_min_defense_data = list(map(lambda df: df["hack_probability"].values, eval_min_defense_dfs))
    hack_prob_eval_min_defense_means = np.mean(tuple(hack_prob_eval_min_defense_data), axis=0)
    hack_prob_eval_min_defense_stds = np.std(tuple(hack_prob_eval_min_defense_data), axis=0, ddof=1)

    a_cum_reward_train_min_defense_data = list(
        map(lambda df: df["attacker_cumulative_reward"].values, train_min_defense_dfs))
    a_cum_reward_train_min_defense_means = np.mean(tuple(a_cum_reward_train_min_defense_data), axis=0)
    a_cum_reward_train_min_defense_stds = np.std(tuple(a_cum_reward_train_min_defense_data), axis=0, ddof=1)
    a_cum_reward_eval_min_defense_data = list(
        map(lambda df: df["attacker_cumulative_reward"].values, eval_min_defense_dfs))
    a_cum_reward_eval_min_defense_means = np.mean(tuple(a_cum_reward_eval_min_defense_data), axis=0)
    a_cum_reward_eval_min_defense_stds = np.std(tuple(a_cum_reward_eval_min_defense_data), axis=0, ddof=1)

    d_cum_reward_train_min_defense_data = list(
        map(lambda df: df["defender_cumulative_reward"].values, train_min_defense_dfs))
    d_cum_reward_train_min_defense_means = np.mean(tuple(d_cum_reward_train_min_defense_data), axis=0)
    d_cum_reward_train_min_defense_stds = np.std(tuple(d_cum_reward_train_min_defense_data), axis=0, ddof=1)
    d_cum_reward_eval_min_defense_data = list(
        map(lambda df: df["defender_cumulative_reward"].values, eval_min_defense_dfs))
    d_cum_reward_eval_min_defense_means = np.mean(tuple(d_cum_reward_eval_min_defense_data), axis=0)
    d_cum_reward_eval_min_defense_stds = np.std(tuple(d_cum_reward_eval_min_defense_data), axis=0, ddof=1)

    train_random_attack_dfs = []
    eval_random_attack_dfs = []
    for csv_path in random_attack_train_csv_paths:
        df = read_data(csv_path)
        train_random_attack_dfs.append(df)

    for csv_path in random_attack_eval_csv_paths:
        df = read_data(csv_path)
        eval_random_attack_dfs.append(df)

    hack_prob_train_random_attack_data = list(map(lambda df: df["hack_probability"].values, train_random_attack_dfs))
    hack_prob_train_random_attack_means = np.mean(tuple(hack_prob_train_random_attack_data), axis=0)
    hack_prob_train_random_attack_stds = np.std(tuple(hack_prob_train_random_attack_data), axis=0, ddof=1)
    hack_prob_eval_random_attack_data = list(map(lambda df: df["hack_probability"].values, eval_random_attack_dfs))
    hack_prob_eval_random_attack_means = np.mean(tuple(hack_prob_eval_random_attack_data), axis=0)
    hack_prob_eval_random_attack_stds = np.std(tuple(hack_prob_eval_random_attack_data), axis=0, ddof=1)

    a_cum_reward_train_random_attack_data = list(
        map(lambda df: df["attacker_cumulative_reward"].values, train_random_attack_dfs))
    a_cum_reward_train_random_attack_means = np.mean(tuple(a_cum_reward_train_random_attack_data), axis=0)
    a_cum_reward_train_random_attack_stds = np.std(tuple(a_cum_reward_train_random_attack_data), axis=0, ddof=1)
    a_cum_reward_eval_random_attack_data = list(
        map(lambda df: df["attacker_cumulative_reward"].values, eval_random_attack_dfs))
    a_cum_reward_eval_random_attack_means = np.mean(tuple(a_cum_reward_eval_random_attack_data), axis=0)
    a_cum_reward_eval_random_attack_stds = np.std(tuple(a_cum_reward_eval_random_attack_data), axis=0, ddof=1)

    d_cum_reward_train_random_attack_data = list(
        map(lambda df: df["defender_cumulative_reward"].values, train_random_attack_dfs))
    d_cum_reward_train_random_attack_means = np.mean(tuple(d_cum_reward_train_random_attack_data), axis=0)
    d_cum_reward_train_random_attack_stds = np.std(tuple(d_cum_reward_train_random_attack_data), axis=0, ddof=1)
    d_cum_reward_eval_random_attack_data = list(
        map(lambda df: df["defender_cumulative_reward"].values, eval_random_attack_dfs))
    d_cum_reward_eval_random_attack_means = np.mean(tuple(d_cum_reward_eval_random_attack_data), axis=0)
    d_cum_reward_eval_random_attack_stds = np.std(tuple(d_cum_reward_eval_random_attack_data), axis=0, ddof=1)

    train_random_defense_dfs = []
    eval_random_defense_dfs = []
    for csv_path in random_defense_train_csv_paths:
        df = read_data(csv_path)
        train_random_defense_dfs.append(df)

    for csv_path in random_defense_eval_csv_paths:
        df = read_data(csv_path)
        eval_random_defense_dfs.append(df)

    hack_prob_train_random_defense_data = list(map(lambda df: df["hack_probability"].values, train_random_defense_dfs))
    hack_prob_train_random_defense_means = np.mean(tuple(hack_prob_train_random_defense_data), axis=0)
    hack_prob_train_random_defense_stds = np.std(tuple(hack_prob_train_random_defense_data), axis=0, ddof=1)
    hack_prob_eval_random_defense_data = list(map(lambda df: df["hack_probability"].values, eval_random_defense_dfs))
    hack_prob_eval_random_defense_means = np.mean(tuple(hack_prob_eval_random_defense_data), axis=0)
    hack_prob_eval_random_defense_stds = np.std(tuple(hack_prob_eval_random_defense_data), axis=0, ddof=1)

    a_cum_reward_train_random_defense_data = list(
        map(lambda df: df["attacker_cumulative_reward"].values, train_random_defense_dfs))
    a_cum_reward_train_random_defense_means = np.mean(tuple(a_cum_reward_train_random_defense_data), axis=0)
    a_cum_reward_train_random_defense_stds = np.std(tuple(a_cum_reward_train_random_defense_data), axis=0, ddof=1)
    a_cum_reward_eval_random_defense_data = list(
        map(lambda df: df["attacker_cumulative_reward"].values, eval_random_defense_dfs))
    a_cum_reward_eval_random_defense_means = np.mean(tuple(a_cum_reward_eval_random_defense_data), axis=0)
    a_cum_reward_eval_random_defense_stds = np.std(tuple(a_cum_reward_eval_random_defense_data), axis=0, ddof=1)

    d_cum_reward_train_random_defense_data = list(
        map(lambda df: df["defender_cumulative_reward"].values, train_random_defense_dfs))
    d_cum_reward_train_random_defense_means = np.mean(tuple(d_cum_reward_train_random_defense_data), axis=0)
    d_cum_reward_train_random_defense_stds = np.std(tuple(d_cum_reward_train_random_defense_data), axis=0, ddof=1)
    d_cum_reward_eval_random_defense_data = list(
        map(lambda df: df["defender_cumulative_reward"].values, eval_random_defense_dfs))
    d_cum_reward_eval_random_defense_means = np.mean(tuple(d_cum_reward_eval_random_defense_data), axis=0)
    d_cum_reward_eval_random_defense_stds = np.std(tuple(d_cum_reward_eval_random_defense_data), axis=0, ddof=1)

    train_two_agents_dfs = []
    eval_two_agents_dfs = []
    for csv_path in two_agents_train_csv_paths:
        df = read_data(csv_path)
        train_two_agents_dfs.append(df)

    for csv_path in two_agents_eval_csv_paths:
        df = read_data(csv_path)
        eval_two_agents_dfs.append(df)

    hack_prob_train_two_agents_data = list(map(lambda df: df["hack_probability"].values, train_two_agents_dfs))
    hack_prob_train_two_agents_means = np.mean(tuple(hack_prob_train_two_agents_data), axis=0)
    hack_prob_train_two_agents_stds = np.std(tuple(hack_prob_train_two_agents_data), axis=0, ddof=1)
    hack_prob_eval_two_agents_data = list(map(lambda df: df["hack_probability"].values, eval_two_agents_dfs))
    hack_prob_eval_two_agents_means = np.mean(tuple(hack_prob_eval_two_agents_data), axis=0)
    hack_prob_eval_two_agents_stds = np.std(tuple(hack_prob_eval_two_agents_data), axis=0, ddof=1)

    a_cum_reward_train_two_agents_data = list(
        map(lambda df: df["attacker_cumulative_reward"].values, train_two_agents_dfs))
    a_cum_reward_train_two_agents_means = np.mean(tuple(a_cum_reward_train_two_agents_data), axis=0)
    a_cum_reward_train_two_agents_stds = np.std(tuple(a_cum_reward_train_two_agents_data), axis=0, ddof=1)
    a_cum_reward_eval_two_agents_data = list(
        map(lambda df: df["attacker_cumulative_reward"].values, eval_two_agents_dfs))
    a_cum_reward_eval_two_agents_means = np.mean(tuple(a_cum_reward_eval_two_agents_data), axis=0)
    a_cum_reward_eval_two_agents_stds = np.std(tuple(a_cum_reward_eval_two_agents_data), axis=0, ddof=1)

    d_cum_reward_train_two_agents_data = list(
        map(lambda df: df["defender_cumulative_reward"].values, train_two_agents_dfs))
    d_cum_reward_train_two_agents_means = np.mean(tuple(d_cum_reward_train_two_agents_data), axis=0)
    d_cum_reward_train_two_agents_stds = np.std(tuple(d_cum_reward_train_two_agents_data), axis=0, ddof=1)
    d_cum_reward_eval_two_agents_data = list(
        map(lambda df: df["defender_cumulative_reward"].values, eval_two_agents_dfs))
    d_cum_reward_eval_two_agents_means = np.mean(tuple(d_cum_reward_eval_two_agents_data), axis=0)
    d_cum_reward_eval_two_agents_stds = np.std(tuple(d_cum_reward_eval_two_agents_data), axis=0, ddof=1)

    five_line_plot_w_shades(np.array(list(range(len(hack_prob_train_min_defense_data[0])))) * train_log_freq,
                            hack_prob_train_min_defense_means,
                            np.array(list(range(len(hack_prob_train_random_defense_data[0])))) * train_log_freq,
                            hack_prob_train_random_defense_means,
                            np.array(list(range(len(hack_prob_train_max_attack_data[0])))) * train_log_freq,
                            hack_prob_train_max_attack_means,
                            np.array(list(range(len(hack_prob_train_random_attack_data[0])))) * train_log_freq,
                            hack_prob_train_random_attack_means,
                            np.array(list(range(len(hack_prob_train_two_agents_data[0])))) * train_log_freq,
                            hack_prob_train_two_agents_means, stds_1=hack_prob_train_min_defense_stds,
                            stds_2=hack_prob_train_random_defense_stds, stds_3=hack_prob_train_max_attack_stds,
                            stds_4=hack_prob_train_random_attack_stds, stds_5=hack_prob_train_two_agents_stds,
                            title="Likelihood of Successful Hack [Train] (v" + str(version) + ")",
                            xlabel="Episode \#", ylabel="$\mathbb{P}[Hacked]$",
                            file_name=output_dir + "/avg_hack_prob_train_" + algorithm + "_" + str(version),
                            line1_label="Q-learning vs minimal defense", line2_label="Q-learning vs random defense",
                            line3_label="maximal attack vs Q-learning", line4_label="random attack vs Q-learning",
                            line5_label="Q-learning vs Q-learning", legend_loc="lower right",
                            ylims=(0, 1), markevery_1=eval_freq, markevery_2=eval_freq, markevery_3=eval_freq,
                            markevery_4=eval_freq, markevery_5=eval_freq)

    five_line_plot_w_shades(np.array(list(range(len(hack_prob_eval_min_defense_data[0])))) * eval_freq,
                            hack_prob_eval_min_defense_means,
                            np.array(list(range(len(hack_prob_eval_random_defense_data[0])))) * eval_freq,
                            hack_prob_eval_random_defense_means,
                            np.array(list(range(len(hack_prob_eval_max_attack_data[0])))) * eval_freq,
                            hack_prob_eval_max_attack_means,
                            np.array(list(range(len(hack_prob_eval_random_attack_data[0])))) * eval_freq,
                            hack_prob_eval_random_attack_means,
                            np.array(list(range(len(hack_prob_eval_two_agents_data[0])))) * eval_freq,
                            hack_prob_eval_two_agents_means, stds_1=hack_prob_eval_min_defense_stds,
                            stds_2=hack_prob_eval_random_defense_stds, stds_3=hack_prob_eval_max_attack_stds,
                            stds_4=hack_prob_eval_random_attack_stds, stds_5=hack_prob_eval_two_agents_stds,
                            title="Likelihood of Successful Hack [Eval] (v" + str(version) + ")",
                            xlabel="Episode \#", ylabel="$\mathbb{P}[Hacked]$",
                            file_name=output_dir + "/avg_hack_prob_eval_" + algorithm + "_" + str(version),
                            line1_label="Q-learning vs minimal defense", line2_label="Q-learning vs random defense",
                            line3_label="maximal attack vs Q-learning", line4_label="random attack vs Q-learning",
                            line5_label="Q-learning vs Q-learning", legend_loc="lower right",
                            ylims=(0, 1), markevery_1=1, markevery_2=1, markevery_3=1, markevery_4=1,
                            markevery_5=1)

    five_line_plot_w_shades(np.array(list(range(len(a_cum_reward_train_min_defense_data[0])))) * train_log_freq,
                            a_cum_reward_train_min_defense_means,
                            np.array(list(range(len(a_cum_reward_train_random_defense_data[0])))) * train_log_freq,
                            a_cum_reward_train_random_defense_means,
                            np.array(list(range(len(a_cum_reward_train_max_attack_data[0])))) * train_log_freq,
                            a_cum_reward_train_max_attack_means,
                            np.array(list(range(len(a_cum_reward_train_random_attack_data[0])))) * train_log_freq,
                            a_cum_reward_train_random_attack_means,
                            np.array(list(range(len(a_cum_reward_train_two_agents_data[0])))) * train_log_freq,
                            a_cum_reward_train_two_agents_means, stds_1=a_cum_reward_train_min_defense_stds,
                            stds_2=a_cum_reward_train_random_defense_stds, stds_3=a_cum_reward_train_max_attack_stds,
                            stds_4=a_cum_reward_train_random_attack_stds, stds_5=a_cum_reward_train_two_agents_stds,
                            title="Cumulative Reward for Attacker [Train] (v" + str(version) + ")",
                            xlabel="Episode \#", ylabel="Cumulative Reward",
                            file_name=output_dir + "/avg_a_cum_reward_train_" + algorithm + "_" + str(version),
                            line1_label="Q-learning vs minimal defense", line2_label="Q-learning vs random defense",
                            line3_label="maximal attack vs Q-learning", line4_label="random attack vs Q-learning",
                            line5_label="Q-learning vs Q-learning", legend_loc="lower right",
                            markevery_1=eval_freq, markevery_2=eval_freq, markevery_3=eval_freq,
                            markevery_4=eval_freq, markevery_5=eval_freq)

    five_line_plot_w_shades(np.array(list(range(len(d_cum_reward_train_min_defense_data[0])))) * train_log_freq,
                            d_cum_reward_train_min_defense_means,
                            np.array(list(range(len(d_cum_reward_train_random_defense_data[0])))) * train_log_freq,
                            d_cum_reward_train_random_defense_means,
                            np.array(list(range(len(d_cum_reward_train_max_attack_data[0])))) * train_log_freq,
                            d_cum_reward_train_max_attack_means,
                            np.array(list(range(len(d_cum_reward_train_random_attack_data[0])))) * train_log_freq,
                            d_cum_reward_train_random_attack_means,
                            np.array(list(range(len(d_cum_reward_train_two_agents_data[0])))) * train_log_freq,
                            d_cum_reward_train_two_agents_means, stds_1=d_cum_reward_train_min_defense_stds,
                            stds_2=d_cum_reward_train_random_defense_stds, stds_3=d_cum_reward_train_max_attack_stds,
                            stds_4=d_cum_reward_train_random_attack_stds, stds_5=d_cum_reward_train_two_agents_stds,
                            title="Cumulative Reward for Defender [Train] (v" + str(version) + ")",
                            xlabel="Episode \#", ylabel="Cumulative Reward",
                            file_name=output_dir + "/avg_d_cum_reward_train_" + algorithm + "_" + str(version),
                            line1_label="Q-learning vs minimal defense", line2_label="Q-learning vs random defense",
                            line3_label="maximal attack vs Q-learning", line4_label="random attack vs Q-learning",
                            line5_label="Q-learning vs Q-learning", legend_loc="lower right",
                            markevery_1=eval_freq, markevery_2=eval_freq, markevery_3=eval_freq,
                            markevery_4=eval_freq, markevery_5=eval_freq)

    plot_all_avg_summary_1(np.array(list(range(len(hack_prob_train_min_defense_data[0])))) * train_log_freq,
                            hack_prob_train_min_defense_means,
                            np.array(list(range(len(hack_prob_train_random_defense_data[0])))) * train_log_freq,
                            hack_prob_train_random_defense_means,
                            np.array(list(range(len(hack_prob_train_max_attack_data[0])))) * train_log_freq,
                            hack_prob_train_max_attack_means,
                            np.array(list(range(len(hack_prob_train_random_attack_data[0])))) * train_log_freq,
                            hack_prob_train_random_attack_means,
                            np.array(list(range(len(hack_prob_train_two_agents_data[0])))) * train_log_freq,
                            hack_prob_train_two_agents_means, hack_prob_train_min_defense_stds,
                            hack_prob_train_random_defense_stds, hack_prob_train_max_attack_stds,
                            hack_prob_train_random_attack_stds, hack_prob_train_two_agents_stds,
                            "Q-learning vs minimal defense", "Q-learning vs random defense",
                            "maximal attack vs Q-learning", "random attack vs Q-learning",
                            "Q-learning vs Q-learning", "Hack probability (train, v" + str(version) + ")",
                           "Episode \#", "$\mathbb{P}[Hacked]$", eval_freq, eval_freq, eval_freq, eval_freq, eval_freq,

                           np.array(list(range(len(hack_prob_eval_min_defense_data[0])))) * eval_freq,
                           hack_prob_eval_min_defense_means,
                           np.array(list(range(len(hack_prob_eval_random_defense_data[0])))) * eval_freq,
                           hack_prob_eval_random_defense_means,
                           np.array(list(range(len(hack_prob_eval_max_attack_data[0])))) * eval_freq,
                           hack_prob_eval_max_attack_means,
                           np.array(list(range(len(hack_prob_eval_random_attack_data[0])))) * eval_freq,
                           hack_prob_eval_random_attack_means,
                           np.array(list(range(len(hack_prob_eval_two_agents_data[0])))) * eval_freq,
                           hack_prob_eval_two_agents_means, hack_prob_eval_min_defense_stds,
                           hack_prob_eval_random_defense_stds, hack_prob_eval_max_attack_stds,
                           hack_prob_eval_random_attack_stds, hack_prob_eval_two_agents_stds,
                           "Q-learning vs minimal defense", "Q-learning vs random defense",
                           "maximal attack vs Q-learning", "random attack vs Q-learning",
                           "Q-learning vs Q-learning", "Hack probability (eval v" + str(version) + ")",
                            "Episode \#", "$\mathbb{P}[Hacked]$", 1,1, 1, 1,1,

                           np.array(list(range(len(a_cum_reward_train_min_defense_data[0])))) * train_log_freq,
                           a_cum_reward_train_min_defense_means,
                           np.array(list(range(len(a_cum_reward_train_random_defense_data[0])))) * train_log_freq,
                           a_cum_reward_train_random_defense_means,
                           np.array(list(range(len(a_cum_reward_train_max_attack_data[0])))) * train_log_freq,
                           a_cum_reward_train_max_attack_means,
                           np.array(list(range(len(a_cum_reward_train_random_attack_data[0])))) * train_log_freq,
                           a_cum_reward_train_random_attack_means,
                           np.array(list(range(len(a_cum_reward_train_two_agents_data[0])))) * train_log_freq,
                           a_cum_reward_train_two_agents_means, a_cum_reward_train_min_defense_stds,
                           a_cum_reward_train_random_defense_stds, a_cum_reward_train_max_attack_stds,
                           a_cum_reward_train_random_attack_stds, a_cum_reward_train_two_agents_stds,
                           "Q-learning vs minimal defense", "Q-learning vs random defense",
                           "maximal attack vs Q-learning", "random attack vs Q-learning",
                           "Q-learning vs Q-learning", "Attacker reward (train v" + str(version) + ")",
                           "Episode \#", "Cumulative Reward", eval_freq, eval_freq, eval_freq, eval_freq, eval_freq,

                           np.array(list(range(len(d_cum_reward_train_min_defense_data[0])))) * train_log_freq,
                           d_cum_reward_train_min_defense_means,
                           np.array(list(range(len(d_cum_reward_train_random_defense_data[0])))) * train_log_freq,
                           d_cum_reward_train_random_defense_means,
                           np.array(list(range(len(d_cum_reward_train_max_attack_data[0])))) * train_log_freq,
                           d_cum_reward_train_max_attack_means,
                           np.array(list(range(len(d_cum_reward_train_random_attack_data[0])))) * train_log_freq,
                           d_cum_reward_train_random_attack_means,
                           np.array(list(range(len(d_cum_reward_train_two_agents_data[0])))) * train_log_freq,
                           d_cum_reward_train_two_agents_means, d_cum_reward_train_min_defense_stds,
                           d_cum_reward_train_random_defense_stds, d_cum_reward_train_max_attack_stds,
                           d_cum_reward_train_random_attack_stds, d_cum_reward_train_two_agents_stds,
                           "Q-learning vs minimal defense", "Q-learning vs random defense",
                           "maximal attack vs Q-learning", "random attack vs Q-learning", "Q-learning vs Q-learning",
                           "Defender reward (train v" + str(version) + ")", "Episode \#",
                           "Cumulative Reward", eval_freq, eval_freq, eval_freq, eval_freq, eval_freq,
                           output_dir + "/combined_plot_" + algorithm + "_" + str(version)
                           )

    # Save mean and std data
    with open(output_dir + "/data/hack_prob_train_min_defense_means.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["hack_prob_train_min_defense_means"])
        for row in hack_prob_train_min_defense_means.tolist():
            writer.writerow([row])

    with open(output_dir + "/data/hack_prob_train_min_defense_stds.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["hack_prob_train_min_defense_stds"])
        for row in hack_prob_train_min_defense_stds.tolist():
            writer.writerow([row])

    with open(output_dir + "/data/hack_prob_train_random_defense_means.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["hack_prob_train_random_defense_means"])
        for row in hack_prob_train_random_defense_means.tolist():
            writer.writerow([row])

    with open(output_dir + "/data/hack_prob_train_random_defense_stds.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["hack_prob_train_random_defense_stds"])
        for row in hack_prob_train_random_defense_stds.tolist():
            writer.writerow([row])

    with open(output_dir + "/data/hack_prob_train_max_attack_means.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["hack_prob_train_max_attack_means"])
        for row in hack_prob_train_max_attack_means.tolist():
            writer.writerow([row])

    with open(output_dir + "/data/hack_prob_train_max_attack_stds.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["hack_prob_train_max_attack_stds"])
        for row in hack_prob_train_max_attack_stds.tolist():
            writer.writerow([row])

    with open(output_dir + "/data/hack_prob_train_random_attack_means.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["hack_prob_train_random_attack_means"])
        for row in hack_prob_train_random_attack_means.tolist():
            writer.writerow([row])

    with open(output_dir + "/data/hack_prob_train_random_attack_stds.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["hack_prob_train_random_attack_stds"])
        for row in hack_prob_train_random_attack_stds.tolist():
            writer.writerow([row])

    with open(output_dir + "/data/hack_prob_train_two_agents_means.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["hack_prob_train_two_agents_means"])
        for row in hack_prob_train_two_agents_means.tolist():
            writer.writerow([row])

    with open(output_dir + "/data/hack_prob_train_two_agents_stds.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["hack_prob_train_two_agents_stds"])
        for row in hack_prob_train_two_agents_stds.tolist():
            writer.writerow([row])

    with open(output_dir + "/data/hack_prob_eval_min_defense_means.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["hack_prob_eval_min_defense_means"])
        for row in hack_prob_eval_min_defense_means.tolist():
            writer.writerow([row])

    with open(output_dir + "/data/hack_prob_eval_min_defense_stds.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["hack_prob_eval_min_defense_stds"])
        for row in hack_prob_eval_min_defense_stds.tolist():
            writer.writerow([row])

    with open(output_dir + "/data/hack_prob_eval_random_defense_means.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["hack_prob_eval_random_defense_means"])
        for row in hack_prob_eval_random_defense_means.tolist():
            writer.writerow([row])

    with open(output_dir + "/data/hack_prob_eval_random_defense_stds.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["hack_prob_eval_random_defense_stds"])
        for row in hack_prob_eval_random_defense_stds.tolist():
            writer.writerow([row])

    with open(output_dir + "/data/hack_prob_eval_max_attack_means.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["hack_prob_eval_max_attack_means"])
        for row in hack_prob_eval_max_attack_means.tolist():
            writer.writerow([row])

    with open(output_dir + "/data/hack_prob_eval_max_attack_stds.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["hack_prob_eval_max_attack_stds"])
        for row in hack_prob_eval_max_attack_stds.tolist():
            writer.writerow([row])

    with open(output_dir + "/data/hack_prob_eval_random_attack_means.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["hack_prob_eval_random_attack_means"])
        for row in hack_prob_eval_random_attack_means.tolist():
            writer.writerow([row])

    with open(output_dir + "/data/hack_prob_eval_random_attack_stds.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["hack_prob_eval_random_attack_stds"])
        for row in hack_prob_eval_random_attack_stds.tolist():
            writer.writerow([row])

    with open(output_dir + "/data/hack_prob_eval_two_agents_means.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["hack_prob_eval_two_agents_means"])
        for row in hack_prob_eval_two_agents_means.tolist():
            writer.writerow([row])

    with open(output_dir + "/data/hack_prob_eval_two_agents_stds.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["hack_prob_eval_two_agents_stds"])
        for row in hack_prob_eval_two_agents_stds.tolist():
            writer.writerow([row])


def plot_average_results(train_dfs, eval_dfs, train_log_frequency, eval_frequency, experiment_title, output_dir):
    hack_prob_train_data = list(map(lambda df: df["hack_probability"].values, train_dfs))
    hack_prob_train_means = np.mean(tuple(hack_prob_train_data), axis=0)
    hack_prob_train_stds = np.std(tuple(hack_prob_train_data), axis=0, ddof=1)
    hack_prob_eval_data = list(map(lambda df: df["hack_probability"].values, eval_dfs))
    hack_prob_eval_means = np.mean(tuple(hack_prob_eval_data), axis=0)
    hack_prob_eval_stds = np.std(tuple(hack_prob_eval_data), axis=0, ddof=1)

    two_line_plot_w_shades(np.array(list(range(len(hack_prob_train_data[0])))) * train_log_frequency,
                           hack_prob_train_means,
                           np.array(list(range(len(hack_prob_eval_data[0])))) * eval_frequency,
                           hack_prob_eval_means,
                           stds_1=hack_prob_train_stds, stds_2=hack_prob_eval_stds,
                           title="Likelihood of Successful Hack",
                           xlabel="Episode \#", ylabel="$\mathbb{P}[Hacked]$",
                           line1_label=experiment_title + " [Train]",
                           line2_label=experiment_title + " [Eval]", legend_loc="lower right",
                           ylims=(0, 1), markevery_1=eval_frequency, markevery_2=1,
                           file_name=output_dir + "/results/plots/avg_hack_probability"
                           )

    cumulative_reward_attacker_data = list(map(lambda df: df["attacker_cumulative_reward"].values, train_dfs))
    cumulative_reward_attacker_means = np.mean(tuple(cumulative_reward_attacker_data), axis=0)
    cumulative_reward_attacker_stds = np.std(tuple(cumulative_reward_attacker_data), axis=0, ddof=1)
    cumulative_reward_defender_data = list(map(lambda df: df["defender_cumulative_reward"].values, train_dfs))
    cumulative_reward_defender_means = np.mean(tuple(cumulative_reward_defender_data), axis=0)
    cumulative_reward_defender_stds = np.std(tuple(cumulative_reward_defender_data), axis=0, ddof=1)

    two_line_plot_w_shades(np.array(list(range(len(cumulative_reward_attacker_data[0])))) * train_log_frequency,
                           cumulative_reward_attacker_means,
                           np.array(list(range(len(cumulative_reward_defender_data[0])))) * train_log_frequency,
                           cumulative_reward_defender_means,
                           stds_1=cumulative_reward_attacker_stds, stds_2=cumulative_reward_defender_stds,
                           title="Cumulative Reward (Train)",
                           xlabel="Episode \#", ylabel="Cumulative Reward",
                           line1_label=experiment_title + " [Attacker]",
                           line2_label=experiment_title + " [Defender]", legend_loc="upper left",
                           markevery_1=eval_frequency, markevery_2=eval_frequency,
                           file_name=output_dir + "/results/plots/avg_cumulative_rewards"
                           )

    avg_episode_len_train_data = list(map(lambda df: df["avg_episode_steps"].values, train_dfs))
    avg_episode_len_train_means = np.mean(tuple(avg_episode_len_train_data), axis=0)
    avg_episode_len_train_stds = np.std(tuple(avg_episode_len_train_data), axis=0, ddof=1)
    avg_episode_len_eval_data = list(map(lambda df: df["avg_episode_steps"].values, eval_dfs))
    avg_episode_len_eval_means = np.mean(tuple(avg_episode_len_eval_data), axis=0)
    avg_episode_len_eval_stds = np.std(tuple(avg_episode_len_eval_data), axis=0, ddof=1)

    two_line_plot_w_shades(np.array(list(range(len(avg_episode_len_train_data[0])))) * train_log_frequency,
                           avg_episode_len_train_means,
                           np.array(list(range(len(avg_episode_len_eval_data[0])))) * eval_frequency,
                           avg_episode_len_eval_means,
                           stds_1=avg_episode_len_train_stds, stds_2=avg_episode_len_eval_stds,
                           title="Avg Episode Lengths",
                           xlabel="Episode \#", ylabel="Avg Length (num steps)",
                           line1_label=experiment_title + " [Train]",
                           line2_label=experiment_title + " [Eval]", legend_loc="upper left",
                           markevery_1=eval_frequency, markevery_2=1,
                           file_name=output_dir + "/results/plots/avg_episode_length"
                           )


def plot_avg_summary(train_dfs, eval_dfs, train_log_frequency, eval_frequency, experiment_title, file_name):
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6, 8))

    # Plot avg hack_probability

    hack_prob_train_data = list(map(lambda df: df["hack_probability"].values, train_dfs))
    hack_prob_train_means = np.mean(tuple(hack_prob_train_data), axis=0)
    hack_prob_train_stds = np.std(tuple(hack_prob_train_data), axis=0, ddof=1)
    hack_prob_eval_data = list(map(lambda df: df["hack_probability"].values, eval_dfs))
    hack_prob_eval_means = np.mean(tuple(hack_prob_eval_data), axis=0)
    hack_prob_eval_stds = np.std(tuple(hack_prob_eval_data), axis=0, ddof=1)

    xlims = (min(min(np.array(list(range(len(hack_prob_train_data[0])))) * train_log_frequency),
                 min(np.array(list(range(len(hack_prob_eval_data[0])))) * eval_frequency)),
             max(max(np.array(list(range(len(hack_prob_train_data[0])))) * train_log_frequency),
                 max(np.array(list(range(len(hack_prob_eval_data[0])))) * eval_frequency)))
    ylims = (0, 1)

    ax[0].plot(np.array(list(range(len(hack_prob_train_data[0])))) * train_log_frequency,
               hack_prob_train_means, label=experiment_title + " [Train]",
               marker="s", ls='-', color="#599ad3",
               markevery=eval_frequency)
    ax[0].fill_between(np.array(list(range(len(hack_prob_train_data[0])))) * train_log_frequency,
                       hack_prob_train_means - hack_prob_train_stds,
                       hack_prob_train_means + hack_prob_train_stds, alpha=0.35, color="#599ad3")

    ax[0].plot(np.array(list(range(len(hack_prob_eval_data[0])))) * eval_frequency,
               hack_prob_eval_means, label=experiment_title + " [Eval]",
               marker="o", ls='-', color='#f9a65a', markevery=1)
    ax[0].fill_between(np.array(list(range(len(hack_prob_eval_data[0])))) * eval_frequency,
                       hack_prob_eval_means - hack_prob_eval_stds,
                       hack_prob_eval_means + hack_prob_eval_stds, alpha=0.35, color='#f9a65a')

    ax[0].set_xlim(xlims)
    ax[0].set_ylim(ylims)

    ax[0].set_title("Likelihood of Successful Hack")
    ax[0].set_xlabel("Episode \#")
    ax[0].set_ylabel("$\mathbb{P}[Hacked]$")
    # set the grid on
    ax[0].grid('on')

    # tweak the axis labels
    xlab = ax[0].xaxis.get_label()
    ylab = ax[0].yaxis.get_label()

    xlab.set_size(10)
    ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax[0].spines['right'].set_color((.8, .8, .8))
    ax[0].spines['top'].set_color((.8, .8, .8))

    ax[0].legend(loc="lower right")

    # Plot Cumulative Reward

    cumulative_reward_attacker_data = list(map(lambda df: df["attacker_cumulative_reward"].values, train_dfs))
    cumulative_reward_attacker_means = np.mean(tuple(cumulative_reward_attacker_data), axis=0)
    cumulative_reward_attacker_stds = np.std(tuple(cumulative_reward_attacker_data), axis=0, ddof=1)
    cumulative_reward_defender_data = list(map(lambda df: df["defender_cumulative_reward"].values, train_dfs))
    cumulative_reward_defender_means = np.mean(tuple(cumulative_reward_defender_data), axis=0)
    cumulative_reward_defender_stds = np.std(tuple(cumulative_reward_defender_data), axis=0, ddof=1)

    xlims = (min(min(np.array(list(range(len(cumulative_reward_attacker_data[0])))) * train_log_frequency),
                 min(np.array(list(range(len(cumulative_reward_defender_data[0])))) * train_log_frequency)),
             max(max(np.array(list(range(len(cumulative_reward_attacker_data[0])))) * train_log_frequency),
                 max(np.array(list(range(len(cumulative_reward_defender_data[0])))) * train_log_frequency)))
    ylims = (min(min(cumulative_reward_attacker_means - cumulative_reward_attacker_stds),
                 min(cumulative_reward_defender_means - cumulative_reward_defender_stds)),
             max(max(cumulative_reward_attacker_means + cumulative_reward_attacker_stds),
                 max(cumulative_reward_defender_means + cumulative_reward_defender_stds)))

    ax[1].plot(np.array(list(range(len(cumulative_reward_attacker_data[0])))) * train_log_frequency,
               cumulative_reward_attacker_means, label=experiment_title + " [Attacker]",
               marker="s", ls='-', color="#599ad3",
               markevery=eval_frequency)
    ax[1].fill_between(np.array(list(range(len(cumulative_reward_attacker_data[0])))) * train_log_frequency,
                       cumulative_reward_attacker_means - cumulative_reward_attacker_stds,
                       cumulative_reward_attacker_means + cumulative_reward_attacker_stds,
                       alpha=0.35, color="#599ad3")

    ax[1].plot(np.array(list(range(len(cumulative_reward_defender_data[0])))) * train_log_frequency,
               cumulative_reward_defender_means, label=experiment_title + " [Defender]",
               marker="o", ls='-', color='#f9a65a', markevery=eval_frequency)
    ax[1].fill_between(np.array(list(range(len(cumulative_reward_defender_data[0])))) * train_log_frequency,
                       cumulative_reward_defender_means - cumulative_reward_defender_stds,
                       cumulative_reward_defender_means + cumulative_reward_defender_stds, alpha=0.35,
                       color='#f9a65a')

    ax[1].set_xlim(xlims)
    ax[1].set_ylim(ylims)

    ax[1].set_title("Cumulative Reward (Train)")
    ax[1].set_xlabel("Episode \#")
    ax[1].set_ylabel("Cumulative Reward")
    # set the grid on
    ax[1].grid('on')

    # tweak the axis labels
    xlab = ax[1].xaxis.get_label()
    ylab = ax[1].yaxis.get_label()

    xlab.set_size(10)
    ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax[1].spines['right'].set_color((.8, .8, .8))
    ax[1].spines['top'].set_color((.8, .8, .8))

    ax[1].legend(loc="upper left")

    # Plot Average Episode Length
    avg_episode_len_train_data = list(map(lambda df: df["avg_episode_steps"].values, train_dfs))
    avg_episode_len_train_means = np.mean(tuple(avg_episode_len_train_data), axis=0)
    avg_episode_len_train_stds = np.std(tuple(avg_episode_len_train_data), axis=0, ddof=1)
    avg_episode_len_eval_data = list(map(lambda df: df["avg_episode_steps"].values, eval_dfs))
    avg_episode_len_eval_means = np.mean(tuple(avg_episode_len_eval_data), axis=0)
    avg_episode_len_eval_stds = np.std(tuple(avg_episode_len_eval_data), axis=0, ddof=1)

    xlims = (min(min(np.array(list(range(len(avg_episode_len_train_data[0])))) * train_log_frequency),
                 min(np.array(list(range(len(avg_episode_len_eval_data[0])))) * eval_frequency)),
             max(max(np.array(list(range(len(avg_episode_len_train_data[0])))) * train_log_frequency),
                 max(np.array(list(range(len(avg_episode_len_eval_data[0])))) * eval_frequency)))
    ylims = (min(min(avg_episode_len_train_means - avg_episode_len_train_stds),
                 min(avg_episode_len_eval_means - avg_episode_len_eval_stds)),
             max(max(avg_episode_len_train_means + avg_episode_len_train_stds),
                 max(avg_episode_len_eval_means + avg_episode_len_eval_stds)))

    ax[2].plot(np.array(list(range(len(avg_episode_len_train_data[0])))) * train_log_frequency,
               avg_episode_len_train_means, label=experiment_title + " [Train]",
               marker="s", ls='-', color="#599ad3",
               markevery=eval_frequency)
    ax[2].fill_between(np.array(list(range(len(avg_episode_len_train_data[0])))) * train_log_frequency,
                       avg_episode_len_train_means - avg_episode_len_train_stds,
                       avg_episode_len_train_means + avg_episode_len_train_stds, alpha=0.35, color="#599ad3")

    ax[2].plot(np.array(list(range(len(avg_episode_len_eval_data[0])))) * eval_frequency,
               avg_episode_len_eval_means, label=experiment_title + " [Eval]",
               marker="o", ls='-', color='#f9a65a', markevery=1)
    ax[2].fill_between(np.array(list(range(len(avg_episode_len_eval_data[0])))) * eval_frequency,
                       avg_episode_len_eval_means - avg_episode_len_eval_stds,
                       avg_episode_len_eval_means + avg_episode_len_eval_stds, alpha=0.35, color='#f9a65a')

    ax[2].set_xlim(xlims)
    ax[2].set_ylim(ylims)

    ax[2].set_title("Avg Episode Lengths")
    ax[2].set_xlabel("Episode \#")
    ax[2].set_ylabel("Avg Length (num steps)")
    # set the grid on
    ax[2].grid('on')

    # tweak the axis labels
    xlab = ax[2].xaxis.get_label()
    ylab = ax[2].yaxis.get_label()

    xlab.set_size(10)
    ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax[2].spines['right'].set_color((.8, .8, .8))
    ax[2].spines['top'].set_color((.8, .8, .8))

    ax[2].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(file_name + ".png", format="png")
    fig.savefig(file_name + ".pdf", format='pdf', dpi=500, bbox_inches='tight', transparent=True)
    plt.close(fig)

def read_and_plot_average_results(experiment_title: str, train_csv_paths : list, eval_csv_paths: list,
                                  train_log_frequency : int, eval_frequency : int, output_dir: str):
    eval_dfs = []
    train_dfs = []
    for train_path in train_csv_paths:
        df = read_data(train_path)
        train_dfs.append(df)

    for eval_path in eval_csv_paths:
        df = read_data(eval_path)
        eval_dfs.append(df)

    plot_average_results(train_dfs, eval_dfs, train_log_frequency, eval_frequency, experiment_title, output_dir)
    plot_avg_summary(train_dfs, eval_dfs, train_log_frequency, eval_frequency, experiment_title,
                     output_dir + "/results/plots/avg_summary")


def read_and_plot_results(train_csv_path : str, eval_csv_path: str, train_log_frequency : int,
                          eval_frequency : int, eval_log_frequency : int, eval_episodes: int, output_dir: str, sim=False,
                          random_seed : int = 0):
    eval_df = read_data(eval_csv_path)
    train_df = read_data(train_csv_path)
    avg_episode_loss_attacker = None
    avg_episode_loss_defender = None
    if "avg_episode_loss_attacker" in train_df:
        avg_episode_loss_attacker = train_df["avg_episode_loss_attacker"]
    if "avg_episode_loss_defender" in train_df:
        avg_episode_loss_defender = train_df["avg_episode_loss_defender"]
    try:
        plot_results(train_df["avg_attacker_episode_rewards"].values,
                     train_df["avg_defender_episode_rewards"].values,
                     train_df["avg_episode_steps"].values,
                     train_df["epsilon_values"], train_df["hack_probability"],
                     train_df["attacker_cumulative_reward"], train_df["defender_cumulative_reward"],
                     avg_episode_loss_attacker, avg_episode_loss_defender,
                     train_df["lr_list"],
                     train_log_frequency, eval_frequency, eval_log_frequency, eval_episodes,
                     output_dir, eval=False, sim=sim, random_seed=random_seed)
        plot_results(eval_df["avg_attacker_episode_rewards"].values,
                     eval_df["avg_defender_episode_rewards"].values,
                     eval_df["avg_episode_steps"].values,
                     eval_df["epsilon_values"], eval_df["hack_probability"],
                     eval_df["attacker_cumulative_reward"], eval_df["defender_cumulative_reward"], None, None,
                     eval_df["lr_list"],
                     train_log_frequency,
                     eval_frequency, eval_log_frequency, eval_episodes, output_dir, eval=True, sim=sim,
                     random_seed=random_seed)
    except Exception as e:
        print(str(e))

    dirs = [x[0].replace("./results/data/" + str(random_seed) + "/state_values/", "")
            for x in os.walk("./results/data/" + str(random_seed) + "/state_values")]
    f_dirs = list(filter(lambda x: x.isnumeric(), dirs))
    f_dirs2 = list(map(lambda x: int(x), f_dirs))
    if len(f_dirs2) > 0:
        last_episode = max(f_dirs2)

        try:
            a_state_plot_frames = np.load("./results/data/" + str(random_seed) + "/state_values/" +
                                          str(last_episode) + "/attacker_frames.npy")
            for i in range(a_state_plot_frames.shape[0]):
                save_image(a_state_plot_frames[i], output_dir + "/results/plots/" + str(random_seed) +
                           "/final_frame_attacker" + str(i))

            a_state_values = np.load("./results/data/" + str(random_seed) + "/state_values/" + str(last_episode) +
                                     "/attacker_state_values.npy")
            probability_plot(np.array(list(range(len(a_state_values)))), a_state_values,
                             title="Attacker State Values",
                             xlabel="Time step (t)", ylabel="$V(s_t)$",
                             file_name=output_dir + "/results/plots/" + str(random_seed) +
                                       "/final_state_values_attacker")

            plot_all(train_df, eval_df, eval_frequency, a_state_values,
                     output_dir + "/results/plots/" + str(random_seed) + "/summary")
        except Exception as e:
            print("Warning: " + str(e))

        try:
            a_state_plot_frames = np.load("./results/data/" + str(random_seed) + "/state_values/" + str(last_episode)
                                          + "/defender_frames.npy")
            for i in range(a_state_plot_frames.shape[0]):
                save_image(a_state_plot_frames[i], output_dir + "/results/plots/" + str(random_seed)
                           + "/final_frame_defender" + str(i))

            d_state_values = np.load("./results/data/" + str(random_seed) + "/state_values/" + str(last_episode) +
                                     "/defender_state_values.npy")
            probability_plot(np.array(list(range(len(d_state_values)))), d_state_values,
                             title="Defender State Values",
                             xlabel="Time step (t)", ylabel="$V(s_t)$",
                             file_name= output_dir + "/results/plots/" + str(random_seed) +
                                        "/final_state_values_defender")

            plot_all(train_df, eval_df, eval_frequency, d_state_values,
                     output_dir + "/results/plots/" + str(random_seed) + "/summary")
        except Exception as e:
            print("Warning: " + str(e))


        plot_two_histograms(train_df["avg_episode_steps"].values, eval_df["avg_episode_steps"].values,
                            title="Avg Episode Lengths",
                            xlabel="Avg Length (num steps)", ylabel="Normalized Frequency",
                            file_name=output_dir + "/results/plots/" + str(random_seed) +
                                      "/train_eval_avg_episode_length",
                            hist1_label="Train", hist2_label="Eval")

        plot_two_histograms(train_df["avg_attacker_episode_rewards"].values,
                            train_df["avg_defender_episode_rewards"].values,
                            title="Avg Episode Returns (Train)",
                            xlabel="Episode \#", ylabel="Avg Return (num steps)",
                            file_name=output_dir + "/results/plots/" + str(random_seed) +
                                      "/attack_defend_avg_episode_return_train",
                            hist1_label="Attacker", hist2_label="Defender", num_bins=3)

        plot_two_histograms(eval_df["avg_attacker_episode_rewards"].values,
                            eval_df["avg_defender_episode_rewards"].values,
                            title="Avg Episode Returns (Eval)",
                            xlabel="Episode \#", ylabel="Avg Return (num steps)",
                            file_name=output_dir + "/results/plots/" + str(random_seed) +
                                      "/attack_defend_avg_episode_return_eval",
                            hist1_label="Attacker", hist2_label="Defender", num_bins=3)

        two_line_plot(np.array(list(range(len(train_df["avg_episode_steps"])))) * 1,
                      train_df["avg_episode_steps"].values,
                      np.array(list(range(len(eval_df["avg_episode_steps"])))) * 1000,
                      eval_df["avg_episode_steps"].values,
                      title="Avg Episode Lengths",
                      xlabel="Episode \#", ylabel="Avg Length (num steps)",
                      file_name=output_dir + "/results/plots/" + str(random_seed) +
                                "/avg_episode_length_train_eval",
                      line1_label="Train", line2_label="Eval", legend_loc="upper right")

        two_line_plot(np.array(list(range(len(train_df["attacker_cumulative_reward"])))) * 1,
                      train_df["attacker_cumulative_reward"].values,
                      np.array(list(range(len(train_df["defender_cumulative_reward"])))),
                      train_df["defender_cumulative_reward"].values,
                      title="Cumulative Reward (Train)",
                      xlabel="Episode \#", ylabel="Cumulative Reward",
                      file_name=output_dir + "/results/plots/" + str(random_seed) +
                                "/cumulative_reward_train_attack_defend",
                      line1_label="Attacker", line2_label="Defender", legend_loc="upper left")


def save_image(data, filename):
    """
    Utility function for saving an image from a numpy array

    :param data: the image data
    :param filename: the filename to save it to
    :return: None
    """
    sizes = np.shape(data)
    fig = plt.figure(figsize=(1,1))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data)
    plt.savefig(filename, dpi = sizes[0])
    plt.close()


def plot_results(avg_attacker_episode_rewards: np.ndarray = None, avg_defender_episode_rewards: np.ndarray = None,
                 avg_episode_steps: np.ndarray = None,
                 epsilon_values: np.ndarray = None,
                 hack_probability: np.ndarray = None, attacker_cumulative_reward: np.ndarray = None,
                 defender_cumulative_reward: np.ndarray = None, avg_episode_loss_attacker: np.ndarray = None,
                 avg_episode_loss_defender: np.ndarray = None, learning_rate_values: np.ndarray = None,
                 log_frequency: int = None,
                 eval_frequency: int = None, eval_log_frequency: int = None, eval_episodes: int = None,
                 output_dir: str = None,
                 eval: bool = False, sim:bool = False, random_seed : int = 0) -> None:
    """
    Utility function for plotting results of an experiment in the idsgame environment

    :param avg_attacker_episode_rewards: list of average episode rewards recorded every <log_frequency> of attacker
    :param avg_defender_episode_rewards: list of average episode rewards recorded every <log_frequency> of defender
    :param avg_episode_steps:  list of average episode steps recorded every <log_frequency>
    :param epsilon_values: list of epsilon values recorded every <log_frequency>
    :param hack_probability: list of hack probabilities recorded every <log_frequency>
    :param attacker_cumulative_reward: list of attacker cumulative rewards recorded every <log_frequency>
    :param defender_cumulative_reward: list of defender cumulative rewards recorded every <log_frequency>
    :param avg_episode_loss_attacker: avg episode loss for attacker
    :param avg_episode_loss_defender: avg episode loss for defender
    :param learning_rate_values: learning rate values
    :param log_frequency: frequency that the metrics were recorded
    :param eval_frequency: frequency of evaluation
    :param eval_frequency: number of evaluation episodes
    :param eval_log_frequency: log-frequency of evaluation
    :param output_dir: base directory to save the plots
    :param eval: if True save plots with "eval" suffix.
    :param sim: if True save plots with "sim" suffix.
    :param random_seed: the random seed of the experiment
    :return: None
    """
    step = log_frequency
    suffix = "train"
    if eval:
        suffix = "eval"
        step = eval_frequency
    elif sim:
        suffix = "simulation"
    if avg_attacker_episode_rewards is not None:
        simple_line_plot(np.array(list(range(len(avg_attacker_episode_rewards)))) * step, avg_attacker_episode_rewards,
                         title="Avg Attacker Episodic Returns",
                         xlabel="Episode \#", ylabel="Avg Return",
                         file_name=output_dir + "/results/plots/" + str(random_seed) +
                                   "/avg_attacker_episode_returns_" + suffix)
    if avg_defender_episode_rewards is not None:
        simple_line_plot(np.array(list(range(len(avg_defender_episode_rewards)))) * step, avg_defender_episode_rewards,
                         title="Avg Defender Episodic Returns",
                         xlabel="Episode \#", ylabel="Avg Return",
                         file_name=output_dir + "/results/plots/" + str(random_seed) +
                                   "/avg_defender_episode_returns_" + suffix)
    if avg_episode_steps is not None:
        simple_line_plot(np.array(list(range(len(avg_episode_steps))))*step, avg_episode_steps,
                         title="Avg Episode Lengths",
                         xlabel="Episode \#", ylabel="Avg Length (num steps)",
                         file_name=output_dir + "/results/plots/" + str(random_seed) +
                                   "/avg_episode_lengths_" + suffix)
    if epsilon_values is not None:
        simple_line_plot(np.array(list(range(len(epsilon_values))))*step, epsilon_values,
                         title="Exploration rate (Epsilon)",
                         xlabel="Episode \#", ylabel="Epsilon", file_name=output_dir + "/results/plots/" +
                                                                          str(random_seed) + "/epsilon_" + suffix)
    if hack_probability is not None:
        probability_plot(np.array(list(range(len(hack_probability)))) * step,
                         hack_probability,
                         title="Likelihood of Successful Hack", ylims=(0, 1),
                         xlabel="Episode \#", ylabel="$\mathbb{P}[Hacked]$", file_name=output_dir +
                                                                         "/results/plots/" + str(random_seed) +
                                                                                       "/hack_probability_" + suffix)
    if attacker_cumulative_reward is not None:
        simple_line_plot(np.array(list(range(len(attacker_cumulative_reward)))) * step, attacker_cumulative_reward,
                         title="Attacker Cumulative Reward",
                         xlabel="Episode \#", ylabel="Cumulative Reward",
                         file_name=output_dir + "/results/plots/" + str(random_seed) +
                                   "/attacker_cumulative_reward_" + suffix)
    if defender_cumulative_reward is not None:
        simple_line_plot(np.array(list(range(len(defender_cumulative_reward)))) * step,
                         defender_cumulative_reward,
                         title="Defender Cumulative Reward",
                         xlabel="Episode \#", ylabel="Cumulative Reward",
                         file_name=output_dir + "/results/plots/" + str(random_seed) +
                                   "/defender_cumulative_reward_" + suffix)
    if avg_episode_loss_attacker is not None and len(avg_episode_loss_attacker) > 0:
        try:
            simple_line_plot(np.array(list(range(len(avg_episode_loss_attacker)))) * step,
                             avg_episode_loss_attacker,
                             title="Avg Episode Loss (Attacker)",
                             xlabel="Episode \#", ylabel="Loss",
                             file_name=output_dir + "/results/plots/" + str(random_seed) +
                                       "/avg_episode_loss_attacker_" + suffix)
        except Exception as e:
            print(str(e))
    if avg_episode_loss_defender is not None and len(avg_episode_loss_defender) > 0:
        try:
            simple_line_plot(np.array(list(range(len(avg_episode_loss_defender)))) * step,
                             avg_episode_loss_defender,
                             title="Avg Episode Loss (Defender)",
                             xlabel="Episode \#", ylabel="Loss",
                             file_name=output_dir + "/results/plots/" + str(random_seed) +
                                       "/avg_episode_loss_defender_" + suffix)
        except Exception as e:
            print(str(e))
    if learning_rate_values is not None:
        simple_line_plot(np.array(list(range(len(learning_rate_values)))) * step, learning_rate_values,
                         title="Learning rate (Eta)",
                         xlabel="Episode \#", ylabel="Learning Rate",
                         file_name=output_dir + "/results/plots/" + str(random_seed) + "/lr_" + suffix)


