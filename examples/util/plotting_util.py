
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

def simple_line_plot(x, y, title="Test", xlabel="test", ylabel="test", file_name="test.eps",
                     xlims=None, ylims = None, log=False):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
    if xlims is None:
        xlims = (min(x), max(x))
    if ylims is None:
        ylims = (min(y), max(y))
    smooth = interp1d(x, y)
    x_smooth = np.linspace(min(x), max(x), 100)
    ax.errorbar(x, y, yerr=None, color="red", ls='-', ecolor='black')
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
    fig.show()
    fig.savefig(file_name, format="png")
    #fig.savefig(file_name, format='eps', dpi=500, bbox_inches='tight', transparent=True)

def plot_results(episode_rewards, episode_steps):
    simple_line_plot(list(range(len(episode_rewards))), episode_rewards, title="Episodic Returns",
                     xlabel="Episode", ylabel="Return", file_name="episode_returns.png")
    simple_line_plot(list(range(len(episode_steps))), episode_steps, title="Episode Lengths",
                     xlabel="Episode", ylabel="Length (num steps)", file_name="episode_lengths.png")
