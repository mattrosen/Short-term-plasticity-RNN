import numpy as np
from parameters import *
import model
import sys


def try_model(gpu_id):
    try:
        # Run model
        model.main(gpu_id)
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')

# Wrapped into main
if __name__ == "__main__":

    # Handle args
    try:
        gpu_id = sys.argv[1]
        print('Selecting GPU ', gpu_id)
    except:
        gpu_id = None

    trainer_id = int(gpu_id)
    n_networks = 25000

    # Update parameters
    update_parameters({ 'simulation_reps'           : 0,
                        'batch_train_size'          : 2048,
                        'learning_rate'             : 0.02,
                        'noise_rnn_sd'              : 0.5,
                        'noise_in_sd'               : 0.1,
                        'num_iterations'            : 300,
                        'spike_regularization'      : 'L1',
                        'synaptic_config'           : 'full',
                        'test_cost_multiplier'      : 2.,
                        'balance_EI'                : True,
                        'weight_cost'               : 1,
                        'spike_cost'                : 1e-3,
                        'fix_time'                  : 200,
                        'sample_time'               : 200,
                        'delay_time'                : 500,
                        'test_time'                 : 200,
                        'num_network_sets_per_gpu'  : n_networks // par['n_networks'],
                        'savedir'                   : './savedir/'})

    task_list = ['DMRS180','ABBA','ABCA','dualDMS',]
    # For Oliver:
    #task_list = ['DMS+DMRS','DMS+DMRS_early_cue', 'DMS+DMRS_full_cue', 'DMS+DMC','DMS+DMRS+DMC','location_DMS']

    # Run models
    for task in task_list:
        #save_fn = task + '.pkl'
        update_parameters({'trial_type': task})#, 'save_fn': save_fn})
        try_model(gpu_id)
