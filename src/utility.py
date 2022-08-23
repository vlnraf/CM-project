import numpy as np
import matplotlib.pyplot as plt

def load_matrix(type):
    PATH = "../matrix/"
    M = np.loadtxt(PATH + type)

    return M

def make_plot(normsGradient, relativeErrors, type = 'M1', experiment_name='M'):
    font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 20}
    plt.rc('font', **font)
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    #relErrorPlot = fig.add_subplot(2,1,1)
    #gradientPlot = fig.add_subplot(2,1,2)
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    
    # Title
    fig.suptitle(
        'Matrix Type  ' + type + ' Relative error and gradient norm')
    
    for i in range(len(relativeErrors)):
        relativeErrors[i] = [max(err, 1e-20) for err in relativeErrors[i]]
        
        x = list(range(len(relativeErrors[i])))
        axs[0].plot(x, relativeErrors[i])
        axs[1].plot(x, normsGradient[i])
    
    axs[1].set_xlabel("Iterations")
    axs[1].set_ylabel("Norm")
    axs[0].set_xlabel("Iterations")
    axs[0].set_ylabel("Relative Error")

    plot_path = 'plot/'
    plt.savefig(plot_path + experiment_name + '.png')
