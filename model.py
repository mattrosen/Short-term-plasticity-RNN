"""
Nicolas Masse 2017
Contributions from Gregory Grant, Catherine Lee
"""

import tensorflow as tf
import numpy as np
import stimulus
import time
import analysis
import AdamOpt
from parameters import *
import pickle
import multistim
import matplotlib.pyplot as plt

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print('Using EI Network:\t', par['EI'])
print('Synaptic configuration:\t', par['synapse_config'], "\n")

"""
Model setup and execution
"""

class Model:

    def __init__(self, input_data, td, target_data, mask, td_input, gate_learning):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = tf.unstack(input_data, axis=1)
        self.target_data = tf.unstack(target_data, axis=1)
        self.mask = tf.unstack(mask, axis=0)
        if par['dynamic_topdown']:
            self.td_input = td_input
        else:
            self.td = tf.tile(tf.reshape(td,[par['n_hidden'],1]), [1, par['batch_train_size']])
            self.td_input = -1
        self.gate_learning = gate_learning

        #self.td = tf.constant(np.float32(1))

        # Load the initial hidden state activity to be used at the start of each trial
        self.hidden_init = tf.constant(par['h_init'])

        # Load the initial synaptic depression and facilitation to be used at the start of each trial
        self.synapse_x_init = tf.constant(par['syn_x_init'])
        self.synapse_u_init = tf.constant(par['syn_u_init'])

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()


    def run_model(self):

        """
        Run the reccurent network
        History of hidden state activity stored in self.hidden_state_hist
        """
        self.rnn_cell_loop(self.input_data, self.hidden_init, self.synapse_x_init, self.synapse_u_init)

        with tf.variable_scope('output'):
            W_out = tf.get_variable('W_out', initializer = par['w_out0'], trainable=True)
            b_out = tf.get_variable('b_out', initializer = par['b_out0'], trainable=True)

        """
        Network output
        Only use excitatory projections from the RNN to the output layer
        """
        if par['EI']:
            #self.y_hat = [tf.matmul(tf.nn.relu(W_out),h) + b_out for h in self.hidden_state_hist]
            self.y_hat = [tf.matmul(W_out,h) + b_out for h in self.hidden_state_hist]
        else:
            self.y_hat = [tf.matmul(W_out,h) + b_out for h in self.hidden_state_hist]


    def rnn_cell_loop(self, x_unstacked, h, syn_x, syn_u):

        """
        Initialize weights and biases
        """
        with tf.variable_scope('rnn_cell'):
            W_in = tf.get_variable('W_in', initializer = par['w_in0'], trainable=True)
            W_rnn = tf.get_variable('W_rnn', initializer = par['w_rnn0'], trainable=True)
            b_rnn = tf.get_variable('b_rnn', initializer = par['b_rnn0'], trainable=True)
            if par['dynamic_topdown']:
                W_td = tf.get_variable('W_td', initializer = np.transpose(np.stack(par['topdown'])), trainable=True)

        self.W_ei = tf.constant(par['EI_matrix'])

        self.hidden_state_hist = []
        self.syn_x_hist = []
        self.syn_u_hist = []

        if par['dynamic_topdown']:
            #self.td = tf.matmul(tf.minimum(np.float32(1), tf.nn.relu(W_td)), self.td_input)
            #self.td = tf.matmul(tf.minimum(np.float32(0.99), tf.nn.relu(W_td)), self.td_input)
            self.td = tf.matmul(tf.nn.sigmoid(W_td), self.td_input)
            #print('td', self.td)

        """
        Loop through the neural inputs to the RNN, indexed in time
        """
        for rnn_input in x_unstacked:
            h, syn_x, syn_u = self.rnn_cell(rnn_input, h, syn_x, syn_u)
            self.hidden_state_hist.append(h)
            self.syn_x_hist.append(syn_x)
            self.syn_u_hist.append(syn_u)

    def rnn_cell(self, rnn_input, h, syn_x, syn_u):

        """
        Main computation of the recurrent network
        """
        with tf.variable_scope('rnn_cell', reuse=True):
            W_in = tf.get_variable('W_in')
            W_rnn = tf.get_variable('W_rnn')
            b_rnn = tf.get_variable('b_rnn')

        if par['EI']:
            # ensure excitatory neurons only have postive outgoing weights,
            # and inhibitory neurons have negative outgoing weights
            W_rnn_effective = tf.tensordot(tf.nn.relu(W_rnn), self.W_ei, [[2],[0]])
        else:
            W_rnn_effective = W_rnn

        """
        Update the synaptic plasticity paramaters
        """
        if par['synapse_config'] == 'std_stf':
            # implement both synaptic short term facilitation and depression
            syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h
            syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
            syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
            syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
            h_post = syn_u*syn_x*h

        elif par['synapse_config'] == 'std':
            # implement synaptic short term derpression, but no facilitation
            # we assume that syn_u remains constant at 1
            syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_x*h
            syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
            syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
            h_post = syn_x*h

        elif par['synapse_config'] == 'stf':
            # implement synaptic short term facilitation, but no depression
            # we assume that syn_x remains constant at 1
            syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
            syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
            h_post = syn_u*h

        else:
            # no synaptic plasticity
            h_post = h

        """
        Update the hidden state
        Only use excitatory projections from input layer to RNN
        All input and RNN activity will be non-negative
        """

        # Dendrite summations
        W_in = tf.reduce_sum(W_in, axis=1)
        W_rnn_effective = tf.reduce_sum(W_rnn_effective, axis=1)

        # Hidden state update
        h = self.td*tf.nn.relu(h*(1-par['alpha_neuron'])
                       + par['alpha_neuron']*(tf.matmul(tf.nn.relu(W_in), tf.nn.relu(rnn_input))
                       + tf.matmul(W_rnn_effective, h_post) + b_rnn)
                       + tf.random_normal([par['n_hidden'], par['batch_train_size']], 0, par['noise_rnn'], dtype=tf.float32))

        return h, syn_x, syn_u



    def optimize(self):


        # Use all trainable variables, except those in the convolutional layers
        #variables = [var for var in tf.trainable_variables() if not var.op.name.find('conv')==0]
        variables = [var for var in tf.trainable_variables()]
        adam_optimizer = AdamOpt.AdamOpt(variables, learning_rate = par['learning_rate'])

        previous_weights_mu_minus_1 = {}
        reset_prev_vars_ops = []
        self.big_omega_var = {}
        self.big_omega_terms = {}
        self.grads = {}
        self.aux_loss = 0.0
        self.big_omega_term_op =[]

        for var in variables:
            previous_weights_mu_minus_1[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            self.big_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            self.grads[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            #self.big_omega_terms[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            self.aux_loss += par['omega_c']*tf.reduce_sum(tf.multiply(self.big_omega_var[var.op.name], \
                tf.square(previous_weights_mu_minus_1[var.op.name] - var) ))
            #self.big_omega_term_op.append(tf.assign(self.big_omega_terms[var.op.name], \
            #    tf.square(previous_weights_mu_minus_1[var.op.name] - var)))
            reset_prev_vars_ops.append( tf.assign(previous_weights_mu_minus_1[var.op.name], var ) )

        """
        perf_loss = [mask*tf.reduce_mean(tf.square(y_hat-desired_output),axis=0)
                     for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)]
        """
        perf_loss = [mask*tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = desired_output, dim=0) \
                for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)]


        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        #spike_loss = [par['spike_cost']*tf.reduce_mean(tf.square(h), axis=0) for h in self.hidden_state_hist]
        spike_loss = [par['spike_cost']*tf.reduce_mean(tf.matmul(tf.nn.relu(self.W_ei), h)) for h in self.hidden_state_hist]

        self.wiring_loss = tf.constant(0.)
        for var in [var for var in variables if ('W' in var.op.name and not 'td' in var.op.name)]:
            if 'W_in' in var.op.name:
                self.wiring_loss += tf.reduce_sum(var * tf.constant(par['w_in_pos'], dtype=tf.float32))
            elif 'W_rnn' in var.op.name:
                self.wiring_loss += tf.reduce_sum(var * tf.constant(par['w_rnn_pos'], dtype=tf.float32))
            elif 'W_rnn' in var.op.name:
                self.wiring_loss += tf.reduce_sum(var * tf.constant(par['w_out_pos'], dtype=tf.float32))

        self.wiring_loss *= par['wiring_cost']
        self.perf_loss = tf.reduce_mean(tf.stack(perf_loss, axis=0))
        self.spike_loss = tf.reduce_mean(tf.stack(spike_loss, axis=0))
        self.loss = self.perf_loss + self.spike_loss + self.wiring_loss

        """
        for var in variables:
            if not var.op.name == "rnn_cell/W_td":
                print('weight cost ', var.op.name)
                self.loss += tf.constant(0.0001)*tf.reduce_sum(tf.abs(var))
        """

        if par['dynamic_topdown']:
            self.td_loss = par['td_cost']*tf.reduce_mean(self.td)
        else:
            self.td_loss = 0.0

        # OPTION 1
        self.train_op = adam_optimizer.compute_gradients(self.loss + self.aux_loss + self.td_loss, self.gate_learning)

        # OPTION 2
        """
        opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        grads_and_vars = opt.compute_gradients(self.loss)
        capped_gvs = []
        for grad, var in grads_and_vars:
            if var.name == "rnn_cell/W_rnn:0":
                grad *= par['w_rnn_mask']
                print('Applied weight mask to w_rnn.')
            elif var.name == "output/W_out:0":
                grad *= par['w_out_mask']
                print('Applied weight mask to w_out.')
            if not str(type(grad)) == "<class 'NoneType'>":
                capped_gvs.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))

        self.train_op = opt.apply_gradients(capped_gvs)
        """

        if par['stabilization'] == 'pathint':
            # Zenke method
            self.pathint_stabilization(variables, adam_optimizer, previous_weights_mu_minus_1)

        elif par['stabilization'] == 'EWC':
            # Kirkpatrick method
            self.EWC(variables)



        self.reset_prev_vars = tf.group(*reset_prev_vars_ops)
        self.reset_adam_op = adam_optimizer.reset_params()

        """
        correct_prediction = [tf.reduce_sum(mask*tf.cast(tf.equal(tf.argmax(y_hat,0), tf.argmax(desired_output,0)), tf.float32)) \
            for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)]
        correct_count = [tf.reduce_sum(mask) for mask in self.mask]
        """

        correct_prediction = [tf.reduce_sum(mask*tf.cast(tf.less(tf.argmax(desired_output,0), par['num_motion_dirs']), tf.float32)*tf.cast(tf.equal(tf.argmax(y_hat,0), tf.argmax(desired_output,0)), tf.float32)) \
            for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)]
        correct_count = [tf.reduce_sum(mask*tf.cast(tf.less(tf.argmax(desired_output,0),par['num_motion_dirs']),tf.float32)) \
            for (desired_output, mask) in zip(self.target_data, self.mask)]

        self.accuracy = tf.reduce_sum(tf.stack(correct_prediction))/tf.reduce_sum(tf.stack(correct_count))

    def EWC(self, variables):
        # Kirkpatrick method
        epsilon = 1e-9

        #for var in self.variables:
            #self.fisher_mat[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
        fisher_ops = []
        ewc_opt = tf.train.GradientDescentOptimizer(1)
        y_unstacked = tf.unstack(self.y_hat, axis = 0)
        #for y in y_unstacked:
        for y1 in tf.unstack(y_unstacked[:par['EWC_fisher_calc_batch']]):
            grads_and_vars = ewc_opt.compute_gradients(tf.log(y1 + epsilon))
            for grad, var in grads_and_vars:
                print(var.op.name, grad)
                fisher_ops.append(tf.assign_add(self.big_omega_var[var.op.name], \
                    grad*grad/par['EWC_fisher_calc_batch']/par['EWC_fisher_num_batches']/par['n_output']))

        self.update_big_omega = tf.group(*fisher_ops)

    def pathint_stabilization(self, variables, adam_optimizer, previous_weights_mu_minus_1):
        # Zenke method

        #optimizer_task = tf.train.GradientDescentOptimizer(learning_rate =  par['learning_rate'])
        optimizer_task = tf.train.GradientDescentOptimizer(learning_rate =  1)
        small_omega_var = {}
        delta_params = {}

        reset_small_omega_ops = []
        update_small_omega_ops = []
        update_grads_op = []
        update_big_omega_ops = []
        initialize_prev_weights_ops = []

        for var in variables:

            small_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            delta_params[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            reset_small_omega_ops.append( tf.assign( small_omega_var[var.op.name], small_omega_var[var.op.name]*0.0 ) )
            reset_small_omega_ops.append( tf.assign( delta_params[var.op.name], delta_params[var.op.name]*0.0 ) )
            #k = par['last_layer_mult'] if var.op.name.find('2')>0 else 1
            #print(var.op.name, ' omega multiplier ', k)
            """
            update_big_omega_ops.append( tf.assign_add( self.big_omega_var[var.op.name], tf.div(tf.nn.relu(small_omega_var[var.op.name]), \
            	(par['omega_xi'] + tf.square(var-previous_weights_mu_minus_1[var.op.name])))))
            """
            update_big_omega_ops.append( tf.assign_add( self.big_omega_var[var.op.name], tf.div(tf.nn.relu(small_omega_var[var.op.name]), \
            	(par['omega_xi'] + tf.square(delta_params[var.op.name])))))

        """
        gated_neurons = {}
        s = tf.cast(tf.greater(self.td, 0), tf.float32)
        rnn_gate = tf.cast(tf.greater(tf.matmul(s,tf.transpose(s)),0), tf.float32)
        print('rnn_gate', rnn_gate)
        for var in variables:
            gated_neurons[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            if var.op.name == 'rnn_cell/W_rnn':
                print('gated weights ', var.op.name)
        """

        # After each task is complete, call update_big_omega and reset_small_omega
        self.update_big_omega = tf.group(*update_big_omega_ops)
        #new_big_omega_var = big_omega_var

        # Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
        self.reset_small_omega = tf.group(*reset_small_omega_ops)

        #self.task_op = adam_optimizer_task.compute_gradients(self.task_loss, gates)
        #with tf.control_dependencies([self.train_op]):
        self.delta_grads = adam_optimizer.return_delta_grads()
        self.gradients = optimizer_task.compute_gradients(self.loss)
        # This is called every batch
        for grad,var in self.gradients:
            #update_small_omega_ops.append(tf.assign_add(small_omega_var[var.op.name], -self.delta_grads[var.op.name]*grad ) )
            update_small_omega_ops.append(tf.assign_add(small_omega_var[var.op.name], -self.delta_grads[var.op.name]*self.grads[var.op.name] ) )
            update_small_omega_ops.append(tf.assign_add(delta_params[var.op.name], self.delta_grads[var.op.name]) )
            update_grads_op.append(tf.assign(self.grads[var.op.name], grad))

        self.update_small_omega = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!
        self.update_grads = tf.group(*update_grads_op)


def train_and_analyze(gpu_id, save_fn):

    tf.reset_default_graph()
    main(gpu_id, save_fn)
    #update_parameters(revert_analysis_par)


def main(gpu_id, save_fn):

    if not par['no_gpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    """
    Reset TensorFlow before running anything
    """
    tf.reset_default_graph()

    """
    Create the stimulus class to generate trial paramaters and input activity
    """
    stim = multistim.MultiStimulus()

    n_input, n_hidden, n_output = par['shape']
    N = par['batch_train_size'] # trials per iteration, calculate gradients after batch_train_size

    """
    Define all placeholder
    """
    mask = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_train_size']])
    x = tf.placeholder(tf.float32, shape=[n_input, par['num_time_steps'], par['batch_train_size']])  # input data
    y = tf.placeholder(tf.float32, shape=[n_output, par['num_time_steps'], par['batch_train_size']]) # target data
    td = tf.placeholder(tf.float32, shape=[par['n_hidden']]) # target data
    td_input = tf.placeholder(tf.float32, shape=[par['num_tasks'], par['batch_train_size']]) # top-down input signal
    gate_learning = tf.placeholder(tf.float32)

    config = tf.ConfigProto()
    #config.gpu_options.allow_growth=True

    par['task_list'] = ['XXX' for i in range(19)]

    # enter "config=tf.ConfigProto(log_device_placement=True)" inside Session to check whether CPU/GPU in use
    with tf.Session(config=config) as sess:

        if par['no_gpu']:
            model = Model(x, td, y, mask, td_input, gate_learning)
            init = tf.global_variables_initializer()
        else:
            with tf.device("/gpu:0"):
                model = Model(x, td, y, mask, td_input, gate_learning)
                init = tf.global_variables_initializer()

        sess.run(init)
        t_start = time.time()
        sess.run(model.reset_prev_vars)

        quit()

        saver = tf.train.Saver()
        # Restore variables from previous model if desired
        if par['load_previous_model']:
            saver.restore(sess, par['save_dir'] + par['ckpt_load_fn'])
            print('Model ' +  par['ckpt_load_fn'] + ' restored.')

        # keep track of the model performance across training
        model_performance = {'accuracy': np.zeros((len(par['task_list']), len(par['task_list'])), dtype=np.float32)}


        for j in range(19):

            td_input_signal = np.zeros((par['num_tasks'], par['batch_train_size']), dtype = np.float32)
            td_input_signal[j, :] = 1

            for i in range(par['num_iterations']):

                # generate batch of batch_train_size
                trial_info = {}
                task_name, trial_info = stim.generate_trial(j)
                if i < 0:
                    gl = 0.0
                else:
                    gl = 1.0
                """
                plt.imshow(trial_info['desired_output'][:,:,0], interpolation='none', aspect='auto')
                plt.colorbar()
                plt.show()
                plt.imshow(trial_info['desired_output'][:,:,1], interpolation='none', aspect='auto')
                plt.colorbar()
                plt.show()
                plt.imshow(trial_info['desired_output'][:,:,2], interpolation='none', aspect='auto')
                plt.colorbar()
                plt.show()
                plt.imshow(trial_info['neural_input'][:,:,1], interpolation='none', aspect='auto')
                plt.colorbar()
                plt.show()
                plt.imshow(trial_info['neural_input'][:,:,1], interpolation='none', aspect='auto')
                plt.colorbar()
                plt.show()
                quit()
                """
                if par['stabilization'] == 'pathint':
                    """
                    _, loss, perf_loss, spike_loss, y_hat, state_hist, syn_x_hist, syn_u_hist, aux_loss = \
                        sess.run([model.train_op, model.loss, model.perf_loss, model.spike_loss, model.y_hat, \
                        model.hidden_state_hist, model.syn_x_hist, model.syn_u_hist, model.aux_loss], feed_dict = {x: trial_info['neural_input'], \
                        td:par['topdown'][j], y: trial_info['desired_output'], mask: trial_info['train_mask'], gate_learning: gate})
                    """
                    _, _, acc, aux_loss, perf_loss, h, td_gating = sess.run([model.train_op, model.update_grads,model.accuracy, \
                        model.aux_loss, model.perf_loss, model.hidden_state_hist, model.td], feed_dict = {x: trial_info['neural_input'], \
                        td: np.float32(par['topdown'][j]), y: trial_info['desired_output'], mask: trial_info['train_mask'], td_input: td_input_signal, gate_learning: gl})

                    # This is potentially important, especially for RNNs
                    # Perf loss can be very large during first several iterations, leading to very lareg omega_c values
                    if perf_loss < 2:
                        sess.run(model.update_small_omega)


                elif par['stabilization'] == 'EWC':
                    aux_loss = -1
                    _, acc, perf_loss, h = sess.run([model.train_op, model.accuracy, model.perf_loss, model.hidden_state_hist], \
                        feed_dict = {x: trial_info['neural_input'], \
                        td: np.float32(par['topdown'][j]), y: trial_info['desired_output'], mask: trial_info['train_mask'], td_input: td_input_signal, gate_learning:gl})


                if (i-1)//par['iters_between_outputs'] == (i-1)/par['iters_between_outputs']:
                    print('Iter ', i, 'Accuracy ', acc , ' AuxLoss ', aux_loss , 'Perf Loss ', perf_loss, ' Mean sr ', np.mean(h), ' TD ', np.mean(td_gating))
                    #bo_var = [np.sum(b) for b in bo.values()]
                    #print('Big Omega ', bo_var)
                    #bo_var = [np.sum(b) for b in bot.values()]
                    #print('Big Omega Terms', bo_var)
                    #bo_var = [np.sum(b*b1) for (b,b1) in zip(bot.values(),bo.values())]
                    #print('Big Omega Sums', bo_var)


                        # Update big omegaes, and reset other values before starting new task
            if par['stabilization'] == 'pathint':
                #sess.run(model.update_big_omega,feed_dict={x:stim_in, td:td_in, mask:mk, droput_keep_pct:1.0})
                big_omegas = sess.run([model.update_big_omega, model.big_omega_var])
            elif par['stabilization'] == 'EWC':
                for n in range(par['EWC_fisher_num_batches']):
                    _, trial_info = stim.generate_trial(j)
                    big_omegas = sess.run([model.update_big_omega,model.big_omega_var], feed_dict = {x:trial_info['neural_input'], \
                    td: np.float32(par['topdown'][j]),  y: trial_info['desired_output'],mask:trial_info['train_mask'], td_input: td_input_signal, gate_learning:gl})

            sess.run(model.reset_adam_op)
            sess.run(model.reset_prev_vars)
            if par['stabilization'] == 'pathint':
                sess.run(model.reset_small_omega)

            for k in range(j+1):

                # generate batch of batch_train_size
                _, trial_info = stim.generate_trial(j)
                acc, h, syn_x, syn_u = sess.run([model.accuracy, model.hidden_state_hist, model.syn_x_hist, model.syn_u_hist], \
                    feed_dict = {x: trial_info['neural_input'], td: np.float32(par['topdown'][k]), \
                    y: trial_info['desired_output'], mask: trial_info['train_mask'], td_input: td_input_signal})
                print('ACC ',j,k,acc)
                model_performance['accuracy'][j,k] = acc

            print(model_performance['accuracy'])
            model_performance['par'] = par
            model_performance['task_list'] = par['task_list']
            #pickle.dump(model_performance, open(par['save_dir'] + save_fn, 'wb'))
            #print('Saving data ', save_fn)
            #iteration_time = time.time() - t_start
            #model_performance = append_model_performance(model_performance, accuracy, loss, perf_loss, spike_loss, (i+1)*N, iteration_time)

            """
            Save the network model and output model performance to screen

            if (i+1)%par['iters_between_outputs']==0 or i+1==par['num_iterations']:
                print_results(i, N, iteration_time, perf_loss, spike_loss, state_hist, accuracy)
            """


        """
        Save model, analyze the network model and save the results

        model_performance['par'] = par
        model_performance['task_list'] = par['task_list']
        pickle.dump(model_performance, open(par['save_dir'] + save_fn, 'wb'))
        """

        #save_path = saver.save(sess, par['save_dir'] + par['ckpt_save_fn'])
        if par['analyze_model']:
            update = {'decoding_reps': 100, 'simulation_reps' : 100}
            update_parameters(update)
            weights = eval_weights()


            analysis.analyze_model(trial_info, y_hat, state_hist, syn_x_hist, syn_u_hist, model_performance, weights, \
                simulation = True, tuning = False, decoding = False, load_previous_file = False, save_raw_data = False)

            # Generate another batch of trials with decoding_test_mode = True (sample and test stimuli
            # are independently drawn), and then perform tuning and decoding analysis
            update = {'decoding_test_mode': True}
            update_parameters(update)
            trial_info = stim.generate_trial()
            y_hat, state_hist, syn_x_hist, syn_u_hist = \
                sess.run([model.y_hat, model.hidden_state_hist, model.syn_x_hist, model.syn_u_hist], \
                {x: trial_info['neural_input'], y: trial_info['desired_output'], mask: trial_info['train_mask']})
            analysis.analyze_model(trial_info, y_hat, state_hist, syn_x_hist, syn_u_hist, model_performance, weights, \
                simulation = False, tuning = par['analyze_tuning'], decoding = True, load_previous_file = True, save_raw_data = False)

            if par['trial_type'] == 'dualDMS':
                # run an additional session with probe stimuli
                save_fn = 'probe_' + par['save_fn']
                update = {'probe_trial_pct': 1, 'save_fn': save_fn}
                update_parameters(update)
                trial_info = stim.generate_trial()
                y_hat, state_hist, syn_x_hist, syn_u_hist = \
                    sess.run([model.y_hat, model.hidden_state_hist, model.syn_x_hist, model.syn_u_hist], \
                    {x: trial_info['neural_input'], y: trial_info['desired_output'], mask: trial_info['train_mask']})
                analysis.analyze_model(trial_info, y_hat, state_hist, syn_x_hist, \
                    syn_u_hist, model_performance, weights, simulation = False, tuning = False, decoding = True, \
                    load_previous_file = False, save_raw_data = False)


def append_model_performance(model_performance, accuracy, loss, perf_loss, spike_loss, trial_num, iteration_time):

    model_performance['accuracy'].append(accuracy)
    model_performance['loss'].append(loss)
    model_performance['perf_loss'].append(perf_loss)
    model_performance['spike_loss'].append(spike_loss)
    model_performance['trial'].append(trial_num)
    model_performance['time'].append(iteration_time)

    return model_performance

def eval_weights():

    with tf.variable_scope('rnn_cell', reuse=True):
        W_in = tf.get_variable('W_in')
        W_rnn = tf.get_variable('W_rnn')
        b_rnn = tf.get_variable('b_rnn')

    with tf.variable_scope('output', reuse=True):
        W_out = tf.get_variable('W_out')
        b_out = tf.get_variable('b_out')

    weights = {
        'w_in'  : W_in.eval(),
        'w_rnn' : W_rnn.eval(),
        'w_out' : W_out.eval(),
        'b_rnn' : b_rnn.eval(),
        'b_out'  : b_out.eval()
    }

    return weights

def print_results(iter_num, trials_per_iter, iteration_time, perf_loss, spike_loss, state_hist, accuracy):

    print('Trial {:7d}'.format((iter_num+1)*trials_per_iter) + ' | Time {:0.2f} s'.format(iteration_time) +
      ' | Perf loss {:0.4f}'.format(np.mean(perf_loss)) + ' | Spike loss {:0.4f}'.format(np.mean(spike_loss)) +
      ' | Mean activity {:0.4f}'.format(np.mean(state_hist)) + ' | Accuracy {:0.4f}'.format(np.mean(accuracy)))



main('0', 'testing')
