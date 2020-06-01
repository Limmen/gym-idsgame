import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def main2():
    data = [
        [1, 1, 1, 1],
        [4, 5, 6,7],
        [3, 3, 3, 3],
        [2, 2, 2, 2],
    ]
    cmap = matplotlib.colors.ListedColormap(['white', 'red', "Blue", "gray", "#3C3813",
                                             "#615A19", "#A4940A", "#FFE600"])
    plt.figure(figsize=(8, 8))
    plt.pcolor(data[::-1], cmap=cmap, edgecolors='k', linewidths=3)
    plt.axis("off")
    plt.show()
    # plt.imshow(data)
    # plt.show()

def main():
    N = 10
    # make an empty data set
    data = np.ones((N, N)) * np.nan
    # fill in some fake data
    for j in range(3)[::-1]:
        data[N//2 - j : N//2 + j +1, N//2 - j : N//2 + j +1] = j
    # make a figure + axes
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    # make color map
    my_cmap = matplotlib.colors.ListedColormap(['r', 'g', 'b'])
    # set the 'bad' values (nan) to be white and transparent
    my_cmap.set_bad(color='w', alpha=0)
    # draw the grid
    for x in range(N + 1):
        ax.axhline(x, lw=2, color='k', zorder=5)
        ax.axvline(x, lw=2, color='k', zorder=5)
    # draw the boxes
    ax.imshow(data, interpolation='none', cmap=my_cmap, extent=[0, N, 0, N], zorder=0)
    # turn off the axis labels
    ax.axis('off')
    fig.show()


if __name__ == '__main__':
    main2()