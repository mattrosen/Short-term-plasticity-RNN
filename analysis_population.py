import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np 
import glob
import os
import pickle
import sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import seaborn as sns


# Load + process single network file
def load_single_network_file(filename):

    # load data
    dat_dict = pickle.load(open(filename, 'rb'))

    # extract weight matrix, return
    return dat_dict['weights']['w_rnn']

################################################################################ 
"""                         Analysis: SVM/clustering                         """
################################################################################ 
def perform_clustering_analysis(X, y, to_plot=True):
    """ Perform clustering analysis; assume [n_nets x n_neur x n_neur]."""

    # Fit TSNE
    tsne = TSNE(n_components=2, perplexity=500).fit_transform(X)

    # Plot if desired
    if to_plot:
        colors = sns.color_palette("husl", len(np.unique(y)))
        print(colors, type(colors), tsne.shape, y.shape, y.dtype)
        fig, ax = plt.subplots(1)
        ax.scatter(tsne[:,0], tsne[:,1], c=y)
        fig.savefig("TSNE_plot.png", dpi=500)

    return

def perform_SVM_analysis(X, y):
    """ Perform SVM analysis; assume [n_nets x n_neur x n_neur]."""

    # Split train/test data
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5)

    # Train SVM
    clf = svm.SVC(kernel='rbf', class_weight='balanced', probability=True)
    clf.fit(X_tr, y_tr)

    # Predict on test data, return accuracy
    return clf.score(X_te, y_te)

################################################################################ 
if __name__ == "__main__":

    # Define some params
    N = 500  # number of networks total to load
    N_PER = 10 # number of networks of each type to load
    N_NEUR = 60

    # Load in dataset names
    task_list = ['DMS', 'DMC']#, 'DMC_dual']
    fns = {}
    for task in task_list:
        fns[task] = glob.glob(f"./savedir/*/{task}*.pkl")

    # Load N networks of each type
    nets = {}
    for task in task_list:
        choice_inds = np.random.choice(len(fns[task]), N // N_PER)
        nets[task] = np.zeros((N, N_NEUR, N_NEUR))
        for t, i in enumerate(choice_inds):
            nets[task][t * N_PER:(t + 1) * N_PER, :, :] = load_single_network_file(fns[task][i])

    # Prepare data for analysis
    labels = np.ones(N * len(task_list)).astype(np.int32)
    all_data = np.zeros((N * len(task_list), N_NEUR * N_NEUR))
    for i, task in enumerate(task_list):
        all_data[i * N:(i + 1) * N, :] = np.reshape(nets[task], (N, -1))
        labels[i * N:(i + 1) * N] *= i

    # Conduct SVM analysis
    perform_clustering_analysis(all_data, labels)
    acc = perform_SVM_analysis(all_data, labels)
    print(f"Accuracy: {acc}")
    print(f"Mean weights (DMS): {np.mean(all_data[0:N, :].flatten())}")
    print(f"Mean weights (DMC): {np.mean(all_data[N:, :].flatten())}")

