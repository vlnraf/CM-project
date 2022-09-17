import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_matrix(matrix_path):
    M = np.loadtxt(matrix_path)
    return M

def make_plot(normsGradient, relativeErrors, plot_path='plot/', type = 'M1'):
    font = {'family': 'serif',
        'weight': 'normal',
        'size': 20}

    plt.rc('font', **font)
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    
    fig.suptitle(
        'Matrix Type  ' + type + ' Relative error and gradient norm')

    x = list(range(len(relativeErrors)))
    axs[0].plot(x, relativeErrors)
    axs[1].plot(x, normsGradient)
    
    axs[1].set_xlabel("Iterations")
    axs[1].set_ylabel("Norm")
    axs[0].set_xlabel("Iterations")
    axs[0].set_ylabel("Relative Error")

    Path(plot_path).mkdir(exist_ok=True)
    plt.savefig(plot_path + type + '.png')
