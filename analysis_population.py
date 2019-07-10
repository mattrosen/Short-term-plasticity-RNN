import matplotlib
#matplotlib.use('Agg')
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
from parameters import *
import model


# Load + process single network file
def load_single_network_file(filename, simple_version = True):

    # Load data
    dat_dict = pickle.load(open(filename, 'rb'))

    # extract weight matrix, return
    if simple_version:
        return dat_dict['weights']['w_rnn']
    else:
        return dat_dict

################################################################################ 
"""                         Analysis: SVM/clustering                         """
################################################################################ 
def perform_clustering_analysis(X, y, to_plot=False):
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

def perform_sparsity_analysis(X, orig_shape, to_plot=True):

    # Knock out some percentage of weights (nearest to 0)
    pcts = np.linspace(0.0, 0.1, 5)
    for net_id in range(X.shape[0]):
        sorted_inds = np.argsort(abs(X[net_id,:]))
        for pct in pcts:
            to_remove = int(pct * len(net))
            sparsified = X[net_id, :]
            sparsified[sorted_inds[:to_remove]] = 0
            net = sparsified.reshape(orig_shape)


    return

################################################################################ 
if __name__ == "__main__":

    # Define some params
    N = 10  # number of networks total to load
    N_PER = 10 # number of networks of each type to load
    N_NEUR = 60

    # Load in dataset names
    task_list = ['DMS', 'DMRS180']#, 'DMC_dual']
    fns = {}
    for task in task_list:
        #fns[task] = glob.glob(f"./savedir/*/{task}*.pkl")
        fns[task] = glob.glob(f"/Volumes/My Passport/Lots of RNNs 1/*/{task}*.pkl")

    #print(fns)

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

    # Sparsity analysis
    for task in task_list:
        choice_inds = np.random.choice(len(fns[task]), N // N_PER)
        nets[task] = np.zeros((N, N_NEUR, N_NEUR))
        for t, i in enumerate(choice_inds):
            all_params = load_single_network_file(fns[task][i], False)
            nets[task][t * N_PER:(t + 1) * N_PER, :, :] = all_params['weights']['w_rnn']

            # update weights in par
            #for key, val in all_params['weights'].items():
            #    print(key, val.shape)
            #    if key + "0" in par.keys():
            #        par[key + "0"] = val

            # update parameters in par
            update_parameters(all_params['parameters'])
            
            for key, val in all_params['weights'].items():
                print(key, val.shape)
                if key + "0" in par.keys():
                    par[key + "0"] = val

            w_rnn = all_params['weights']['w_rnn']

            # do the sparsification, one network at a time
            pcts = np.linspace(0.3, 0.9, 5)
            sorted_inds = np.array([np.argsort(abs(w_rnn[net_id,:,:]), None) for net_id in range(w_rnn.shape[0])])
            for pct in pcts:
                to_remove = int(pct * N_NEUR * N_NEUR)
                for net_id in range(w_rnn.shape[0]):
                    rows = sorted_inds[net_id, 0:to_remove] // N_NEUR
                    cols = sorted_inds[net_id, 0:to_remove] % N_NEUR
                    #print(w_rnn[net_id, rows, cols])
                    #print(cols)
                    w_rnn[net_id, rows, cols] = 0
                
                par['w_rnn0'] = w_rnn
                par['num_iterations'] = 20
                par['iters_between_outputs'] = 10
                par['num_network_sets_per_gpu'] = 1

                # try model
                model.main(None)



