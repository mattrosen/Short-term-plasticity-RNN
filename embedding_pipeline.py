################################################################################
# embedding_pipeline.py
# Authors: Matt Rosen + Oliver Zhu
# 
# Pipeline for isomorphism-aware embedding of weight matrices.
################################################################################

################################################################################
# Imports
########################################
import networkx as nx
import numpy as np 
import subprocess
import pickle 
import json
import glob
import sys
import os

################################################################################
# Utility functions
########################################
def load_single_network_file(filename, simple_version = True):

    # Load data
    dat_dict = pickle.load(open(filename, 'rb'))

    # extract weight matrix, return
    if simple_version:
        return dat_dict['weights']['w_rnn']
    else:
        return dat_dict

################################################################################
# Main method
########################################
if __name__ == "__main__":

    # Import graphs
    N      = 10          # number of networks total to load
    N_PER  = 10          # number of networks of each type to load
    N_NEUR = 60          # number of neurons per network
    N_DIM  = sys.argv[1] # number of dimensions for embedding

    # Load in dataset names
    task_list = ['DMS', 'DMRS180']
    fns = {}
    for task in task_list:
        #fns[task] = glob.glob(f"./savedir/*/{task}*.pkl")
        fns[task] = glob.glob(f"/Volumes/My Passport/Lots of RNNs 1/*/{task}*.pkl")

    # Ensure output folder exists
    if not os.path.exists("./dataset/"):
        os.makedirs("./dataset/")

    # Load our weight matrices + write to JSON
    nets = {}
    counter = 0
    metadata = []
    for task in task_list:
        choice_inds = np.random.choice(len(fns[task]), N // N_PER)
        for t, i in enumerate(choice_inds):
            all_params = load_single_network_file(fns[task][i], False)
            w_rnn = all_params['weights']['w_rnn']

            # Sparsify each network separately
            sparsity_fraction = 0.7
            sorted_inds = np.array([np.argsort(abs(w_rnn[net_id,:,:]), None) for net_id in range(w_rnn.shape[0])])

            to_remove = int(sparsity_fraction * N_NEUR * N_NEUR)
            for net_id in range(w_rnn.shape[0]):
                rows = sorted_inds[net_id, 0:to_remove] // N_NEUR
                cols = sorted_inds[net_id, 0:to_remove] % N_NEUR
                w_rnn[net_id, rows, cols] = 0

                # Convert each to JSON as expected for graph2vec; 
                # 2 fields (edges, features)
                G = nx.DiGraph(np.squeeze(w_rnn[net_id, :, :]))
                edges = G.edges()
                edge_list = [[int(a[0]), int(a[1])] for a in edges]
                to_write = json.dumps({"edges": edge_list})
                with open(f"dataset/{counter}.json", 'w') as f:
                    f.write(to_write)

                # Save metadata (e.g. network name/type)
                metadata.append(f"{str(counter)} {fns[task][i][36:]}")
                counter += 1

    with open("metadata.txt", 'w') as f:
        f.write("\n".join(metadata))

    # Invoke the command to run graph2vec through subprocess.call
    subprocess.call(["python3", "../graph2vec/src/graph2vec.py", 
                     "--input-path", "./dataset/", 
                     "--output-path", "embeddings.csv",
                     "--dimensions",  str(N_DIM)])
