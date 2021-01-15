import matplotlib.pyplot as plt


font = {'size': '20', 'color': 'black', 'weight': 'normal'}


def makeFigSingle(title, xlabel, ylabel, xlim=[0, 0], ylim=[0, 0]):
    """
    A function that defines a figure with legible axis labels
    """

    fig = plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(10, 10, forward=True)

    ax = fig.add_subplot(111)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(font['size'])

    ax.set_ylabel(ylabel, **font)
    if ylim != [0, 0] and ylim[0] < ylim[1]:
        ax.set_ylim(ylim)

    ax.set_xlabel(xlabel, **font)
    if xlim != [0, 0] and xlim[0] < xlim[1]:
        ax.set_xlim(xlim)

    ax.set_title(title, **font)

    return fig, ax


def makeFigDouble(title, xlabel, ylabel1, ylabel2, xlim=[0, 0], ylim1=[0, 0], ylim2=[0, 0]):
    """
    A function that defines a figure and axes with two panels that share an
    x-axis and has legible axis labels

    """

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig = plt.gcf()
    fig.set_size_inches(10, 10, forward=True)
    fig.subplots_adjust(hspace=0)

    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontsize(font['size'])
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontsize(font['size'])

    ax1.set_ylabel(ylabel1, **font)
    if ylim1 != [0, 0] and ylim1[0] < ylim1[1]:
        ax1.set_ylim(ylim1)

    ax2.set_ylabel(ylabel2, **font)
    if ylim2 != [0, 0] and ylim2[0] < ylim2[1]:
        ax2.set_ylim(ylim2)

    ax2.set_xlabel(xlabel, **font)
    if xlim != [0, 0] and xlim[0] < xlim[1]:
        ax2.set_xlim(xlim)

    ax1.set_title(title, **font)

    return fig, ax1, ax2
