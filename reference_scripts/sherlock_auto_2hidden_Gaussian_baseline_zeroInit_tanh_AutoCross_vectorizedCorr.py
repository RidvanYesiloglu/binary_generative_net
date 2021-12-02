import numpy as np
import tensorflow as tf
import itertools

import matplotlib.pyplot as plt

import csv
import os
from os import path

# Create environment that starts as zeros
n = 63
m = 4
folder_name = 'results_vect_' + str(n) + 'bits_' + str(m) + 'codes' #_sigma02_5en5' # FOLDER NAME
n_tot = n*m
start_indices = np.arange(0, n*m, n)
pairs = np.array(list(itertools.combinations(start_indices, 2)))
npairs = np.size(pairs, 0)
first_i = pairs[:,0]; second_i = pairs[:,1]

# Hyperparameters: can change H_SIZE, batch_size, ETA, INPUT_DIM, and potentially GAMMA
H_SIZE = 2*n_tot # number of hidden layer neurons
batch_size = 100 # update params after every batch_size number of episodes
ETA = 1e-4 # learning Rate
GAMMA = 0.99 # discount factor
sigma_sqr = 0.1
log_prob_K = -1*np.log(np.sqrt(2*np.pi*sigma_sqr))*n_tot

INPUT_DIM = n_tot # input dimensions
OUTPUT_DIM = n_tot # output dimsensions


#Network to define moving left or right
input = tf.placeholder(tf.float32, [None,INPUT_DIM] , name="input_x")
W1 = tf.get_variable("W1", shape=[INPUT_DIM, H_SIZE],
           initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1", shape=[H_SIZE], initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(input,W1)+b1)
W2 = tf.get_variable("W2", shape=[H_SIZE, H_SIZE],  
           initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2", shape=[H_SIZE], initializer=tf.contrib.layers.xavier_initializer())
layer2 = tf.nn.relu(tf.matmul(layer1,W2)+b2)
W3 = tf.get_variable("W3", shape=[H_SIZE, OUTPUT_DIM],  
           initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable("b3", shape=[OUTPUT_DIM], initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer2,W3) + b3
probability = tf.nn.tanh(score)


#From here we define the parts of the network needed for learning a good policy.
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32,[None,OUTPUT_DIM], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal") # TODO: modify advantages size to be [None, 1] or [None,]? Might be okay
baseline = tf.placeholder(tf.float32, name='baseline')


# The loss function. This sends the weights in the direction of making actions 
# that gave good advantage (reward over time) more likely, and actions that didn't less likely.
loglik_perbit = (input_y - probability)*(input_y - probability)
loglik = -1/(2*sigma_sqr) * tf.reduce_sum(loglik_perbit, axis=1) + log_prob_K # TODO: check if want to reduce sum?
loglike_times_adv = loglik * (advantages-baseline)
loss = -tf.reduce_mean(loglike_times_adv)
newGrads = tf.gradients(loss,tvars)


adam = tf.train.AdamOptimizer(learning_rate=ETA) # Adam optimizer
W1Grad = tf.placeholder(tf.float32,name="batch_grad1") # Placeholders for final gradients once update happens
b1Grad = tf.placeholder(tf.float32,name='batch_grad2')
W2Grad = tf.placeholder(tf.float32,name='batch_grad3')
b2Grad = tf.placeholder(tf.float32,name='batch_grad4')
W3Grad = tf.placeholder(tf.float32,name='batch_grad5')
b3Grad = tf.placeholder(tf.float32,name='batch_grad6')
batchGrad = [W1Grad,b1Grad,W2Grad,b2Grad,W3Grad,b3Grad]
updateGrads = adam.apply_gradients(zip(batchGrad,tvars)) # an TF Operation (tf.python.framework.ops.Operation)


def discount_rewards(r):
    # r -- 1D float array of rewards 
    # output: compute discounted reward (discounted by GAMMA)
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add
    return discounted_r

def comp_stats(action_batch, batch_size, m, n, npairs, start_indices, first_i, second_i):
    # Compute auto/cross correlation reward and balance
    state_pm1_batch = action_batch*2 - 1

    auto_comp = np.nan*np.ones([batch_size, m])
    cross_comp = np.nan*np.ones([batch_size, npairs])
    abs_bal_comp = np.nan*np.ones([batch_size, m])

    for i in range(m):
        # Compute auto correlation for ith sequence
        curr_start_i = start_indices[i]
        state_pm1 = state_pm1_batch[:, curr_start_i:(curr_start_i+n)]
        auto_corr_vectors = np.flip( np.fft.ifft(np.fft.fft(np.flip(state_pm1, axis=1)) * np.fft.fft(state_pm1)), axis=1 )
        mean_side_peak_auto_batch = np.mean( np.square(np.abs(auto_corr_vectors[:,1:])), axis=1 )
        auto_comp[:, i] = mean_side_peak_auto_batch

        # Compute balance for ith sequence
        abs_bal_seqi = np.abs( np.sum(state_pm1, axis=1) )
        abs_bal_comp[:, i] = abs_bal_seqi

    # Compute cross-correlations
    for i in range(npairs):
        i1_curr = first_i[i]
        i2_curr = second_i[i]
        first_seqs = state_pm1_batch[:, i1_curr:(i1_curr+n)]
        second_seqs = state_pm1_batch[:, i2_curr:(i2_curr+n)]

        # Compute cross-correlation for ith pair and save results
        cross_corr_vectors = np.flip( np.fft.ifft(np.fft.fft(np.flip(first_seqs, axis=1)) * np.fft.fft(second_seqs)), axis=1 )
        mean_side_peak_cross_batch = np.mean( np.square(np.abs(cross_corr_vectors)), axis=1 )
        cross_comp[:, i] = mean_side_peak_cross_batch

    # Get auto / cross correlation objectives and reward for batch
    auto_obj_batch = np.mean( auto_comp, axis=1 )
    cross_obj_batch = np.mean( cross_comp, axis=1 )
    reward_batch = -1*np.maximum(auto_obj_batch, cross_obj_batch)

    # Get average absolute balance for each family in batch
    bal_batch = np.mean( abs_bal_comp, axis=1 )

    return reward_batch, auto_obj_batch, cross_obj_batch, bal_batch


def relu(x):
    return np.maximum(x, 0)


# Setup
xs,hs,drs,ys = [],[],[],[]   #Arrays to store parameters till an update happens
running_reward = None
running_rwd_auto = None
running_rwd_cross = None
running_bal = None
episode_number = 0
total_episodes = 1e6
init = tf.initialize_all_variables()
ave_rwd_per_ep = []
ave_auto_per_ep = []
ave_cross_per_ep = []
ave_bal_per_ep = []
running_rwd_tot = []
running_rwd_auto_tot = []
running_rwd_cross_tot = []
running_bal_tot = []
baseline_val = 0
num_test_samps = 1000


# Save file information
folder_name_plot = folder_name + '/plots'
folder_name_test = folder_name + '/test_data'
folder_name_train = folder_name + '/train_data' 
if path.isdir(folder_name):
    raise Exception('\n   Results folder already exists! \n   folder name: ' + folder_name)



os.mkdir(folder_name)
os.mkdir(folder_name_plot)
os.mkdir(folder_name_test)
os.mkdir(folder_name_train)
meta_file = open(folder_name+'/meta_data.txt', 'w')
meta_file_lines = ['sequence length: ' + str(n) + '\n',
                    'family size: ' + str(m) + '\n',
                    'num hidden: ' + str(H_SIZE) + '\n', 
                    'batch size: ' + str(batch_size) + '\n',
                    'learning rate: ' + str(ETA) + '\n',
                    'gamma: ' + str(GAMMA) + '\n',
                    'sigma_sqr: ' + str(sigma_sqr) + '\n',
                    'total episodes: ' + str(total_episodes) + '\n', 
                    'num test samples: ' + str(num_test_samps) + '\n']
meta_file.writelines(meta_file_lines)
meta_file.close()

print_str = 'Created results folder and meta file. \n'
print(print_str)
train_log_file_name_str = folder_name_train+'/train_log.txt'
train_log_file = open(train_log_file_name_str, 'w')
train_log_file.writelines(print_str)
train_log_file.close()


# Training
with tf.Session() as sess:
    rendering = False
    sess.run(init) # TODO: change to global_variables_initializer
    input_initial = np.array([False]*n*m*batch_size, dtype=int) # Initial state of the environment
    
    # Array to store gradients for each min-batch step
    gradBuffer = sess.run(tvars)
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    
    W1_init_val, b1_init_val, W2_init_val, b2_init_val, W3_init_val, b3_init_val = sess.run([W1, b1, W2, b2, W3, b3])
    while episode_number < total_episodes:
            
        # Format the state for placeholder
        x = np.reshape(input_initial,[batch_size,INPUT_DIM])
        
        # Run policy network 
        mu_values = sess.run(probability,feed_dict={input: x})
        a_samp_batch = np.random.normal(loc=mu_values[0], scale=sigma_sqr, size=[batch_size, OUTPUT_DIM])
        action_batch = (a_samp_batch >= 0.0).astype(int)
        
        xs.append(x) # Store x (input)
        y_batch = a_samp_batch # Output sampled values
        ys.append(y_batch)

        # Compute statistics
        reward_batch, auto_obj_batch, cross_obj_batch, bal_batch = comp_stats(action_batch, batch_size, m, n, npairs, start_indices, first_i, second_i)

        # Get average results over batch (for checking/viewing progress)
        rwd_ave_batch = np.mean(reward_batch)
        auto_ave_batch = np.mean(auto_obj_batch)
        cross_ave_batch = np.mean(cross_obj_batch)
        bal_ave_batch = np.mean(bal_batch)
        
        # TODO: update gradient computation
        drs.append(reward_batch) # store reward after action is taken

        # Only 1 action per episode
        episode_number += batch_size
        
        # Stack the memory arrays to feed in session
        batch_x = np.vstack(xs)
        batch_y = np.vstack(ys)
        batch_r = np.vstack(drs)
        baseline_val = rwd_ave_batch # NOTE: Can change to 0 to test without baseline
        
        xs,hs,drs,ys = [],[],[],[] #Reset Arrays

        
        # Get and save the gradient
        ll_pb, ll, llta, loss_val = sess.run([loglik_perbit, loglik, loglike_times_adv, loss],feed_dict={input: batch_x.astype(float), input_y: batch_y.astype(float), advantages: batch_r, baseline: baseline_val})
        tGrad = sess.run(newGrads,feed_dict={input: batch_x.astype(float), input_y: batch_y.astype(float), advantages: batch_r, baseline: baseline_val})
        for ix,grad in enumerate(tGrad):
            gradBuffer[ix] += grad
        # Update Params after Min-Batch number of episodes
        if episode_number % batch_size == 0: 
            sess.run(updateGrads,feed_dict={W1Grad: gradBuffer[0], b1Grad: gradBuffer[1], 
                                            W2Grad: gradBuffer[2], b2Grad: gradBuffer[3],
                                            W3Grad: gradBuffer[4], b3Grad: gradBuffer[5]})
            for ix,grad in enumerate(gradBuffer):
                gradBuffer[ix] = grad * 0
            
            # Print details of the present model
            running_reward = rwd_ave_batch if running_reward is None else running_reward * 0.99 + rwd_ave_batch * 0.01
            running_rwd_auto = auto_ave_batch if running_rwd_auto is None else running_rwd_auto * 0.99 + auto_ave_batch * 0.01
            running_rwd_cross = cross_ave_batch if running_rwd_cross is None else running_rwd_cross * 0.99 + cross_ave_batch * 0.01
            running_bal = bal_ave_batch if running_bal is None else running_bal * 0.99 + bal_ave_batch * 0.01
            
            val_ave_rwd_per_ep = rwd_ave_batch
            val_ave_auto_per_ep = auto_ave_batch
            val_ave_cross_per_ep = cross_ave_batch
            val_ave_bal_per_ep = bal_ave_batch
            val_running_rwd_tot = running_reward
            val_running_rwd_auto = running_rwd_auto
            val_running_rwd_cross = running_rwd_cross
            val_running_bal = running_bal
            
            # Append average rewards           
            ave_rwd_per_ep.append(val_ave_rwd_per_ep)
            ave_auto_per_ep.append(val_ave_auto_per_ep)
            ave_cross_per_ep.append(val_ave_cross_per_ep)
            ave_bal_per_ep.append(val_ave_bal_per_ep)

            # Append running rewards
            running_rwd_tot.append(val_running_rwd_tot)
            running_rwd_auto_tot.append(val_running_rwd_auto)
            running_rwd_cross_tot.append(val_running_rwd_cross)
            running_bal_tot.append(val_running_bal)          

            print_str = 'Average reward (n=' + str(n) + ', m=' + str(m) + ') for episode ' \
            + str(episode_number) + ' is ' + str(round(val_ave_rwd_per_ep,2)) + \
                '.  Total average reward ' + str(round(val_running_rwd_tot,4)) + '. \n'
            print(print_str)
            train_log_file = open(train_log_file_name_str, 'a')
            train_log_file.writelines(print_str)
            train_log_file.close()

    W1_val, b1_val, W2_val, b2_val, W3_val, b3_val = sess.run([W1, b1, W2, b2, W3, b3])

print_str = str(episode_number) + ' episodes completed. \n'
train_log_file = open(train_log_file_name_str, 'a')
train_log_file.writelines(print_str)
train_log_file.close()




# Test performance
ave_perf = 0
ave_perf_auto = 0
ave_perf_cross = 0
ave_balance = 0
nom_ave_perf = 0
nom_ave_auto = 0
nom_ave_cross = 0
nom_ave_balance = 0
ave_auto_list = []
ave_cross_list = []
ave_balance_list = []
nom_ave_auto_list = []
nom_ave_cross_list = []
nom_ave_balance_list = []
for i in range(num_test_samps):
    input_initial = np.array([False]*n*m, dtype=int) # Initial state of the environment
    input_initial = np.reshape(input_initial,[1,INPUT_DIM])
    
    # Run policy network 
    mu_values = np.tanh(relu(relu(input_initial@W1_val + b1_val)@W2_val + b2_val)@W3_val + b3_val  )
    a_samp_batch = np.random.normal(loc=mu_values[0], scale=sigma_sqr, size=[1, OUTPUT_DIM])
    action_batch = (a_samp_batch >= 0.0).astype(int)

    # Compute reward, auto/cross components, and balance
    reward_batch, auto_obj_batch, cross_obj_batch, bal_batch = comp_stats(action_batch, 1, m, n, npairs, start_indices, first_i, second_i)

    reward_batch = reward_batch[0]
    auto_obj_batch = auto_obj_batch[0]
    cross_obj_batch = cross_obj_batch[0]
    bal_batch = bal_batch[0]

    ave_perf += reward_batch
    ave_perf_auto += auto_obj_batch
    ave_perf_cross += cross_obj_batch
    ave_balance += bal_batch
    ave_auto_list.append(auto_obj_batch)
    ave_cross_list.append(cross_obj_batch)
    ave_balance_list.append(bal_batch)
    


nominal_code_family = None
for i in range(num_test_samps):
    input_initial = np.array([False]*n*m, dtype=int) # Initial state of the environment
    input_initial = np.reshape(input_initial,[1,INPUT_DIM])
    
    # Run policy network 
    mu_values = np.tanh(relu(relu(input_initial@W1_val + b1_val)@W2_val + b2_val)@W3_val + b3_val  )
    action_batch = (mu_values >= 0.0).astype(int)
    nominal_code_family = action_batch[0, :]

    # Compute reward, auto/cross components, and balance
    reward_batch, auto_obj_batch, cross_obj_batch, bal_batch = comp_stats(action_batch, 1, m, n, npairs, start_indices, first_i, second_i)

    reward_batch = reward_batch[0]
    auto_obj_batch = auto_obj_batch[0]
    cross_obj_batch = cross_obj_batch[0]
    bal_batch = bal_batch[0]

    nom_ave_perf += reward_batch
    nom_ave_auto += auto_obj_batch
    nom_ave_cross += cross_obj_batch
    nom_ave_balance += bal_batch
    nom_ave_auto_list.append(auto_obj_batch)
    nom_ave_cross_list.append(cross_obj_batch)
    nom_ave_balance_list.append(bal_batch)
#     print(input_initial)

# Get and save code value:
# input_initial = env.reset() # Initial state of the environment
# mu_values = np.tanh(relu(relu(input_initial@W1_val + b1_val)@W2_val + b2_val)@W3_val + b3_val  )
# nominal_code_family = (mu_values >= 0).astype(int)

ave_perf /= num_test_samps
ave_perf_auto /= num_test_samps
ave_perf_cross /= num_test_samps
ave_balance /= num_test_samps
nom_ave_perf /= num_test_samps
nom_ave_auto /= num_test_samps
nom_ave_cross /= num_test_samps
nom_ave_balance /= num_test_samps
print()
print('Average performance:', ave_perf)
print('Average auto:', ave_perf_auto)
print('Average cross:', ave_perf_cross)
print('Average balance:', ave_balance)
print()
print('Nominal average performance:', nom_ave_perf)
print('Nominal average auto:', nom_ave_auto)
print('Nominal average cross:', nom_ave_cross)
print('Nominal average balance:', nom_ave_balance)
print()

print('Auto:', auto_obj_batch)
print('Cross:', cross_obj_batch)
print('Balance:', bal_batch)
print()


###########################################################################################
#################################                         #################################
#################################        Save Data        #################################
#################################                         #################################
###########################################################################################

# Average reward per episode
ave_rwd_per_ep_file = open(folder_name_train+'/ave_rwd_per_ep.csv', 'w')
ave_rwd_per_ep_wr = csv.writer(ave_rwd_per_ep_file, quoting=csv.QUOTE_ALL)
ave_rwd_per_ep_wr.writerow(ave_rwd_per_ep)
ave_rwd_per_ep_file.close()

# Average auto-correlation per episode
ave_auto_per_ep_file = open(folder_name_train+'/ave_auto_per_ep.csv', 'w')
ave_auto_per_ep_wr = csv.writer(ave_auto_per_ep_file, quoting=csv.QUOTE_ALL)
ave_auto_per_ep_wr.writerow(ave_auto_per_ep)
ave_auto_per_ep_file.close()

# Average cross-correlation per episode
ave_cross_per_ep_file = open(folder_name_train+'/ave_cross_per_ep.csv', 'w')
ave_cross_per_ep_wr = csv.writer(ave_cross_per_ep_file, quoting=csv.QUOTE_ALL)
ave_cross_per_ep_wr.writerow(ave_cross_per_ep)
ave_cross_per_ep_file.close()

# Average balance per episode
ave_bal_per_ep_file = open(folder_name_train+'/ave_bal_per_ep.csv', 'w')
ave_bal_per_ep_wr = csv.writer(ave_bal_per_ep_file, quoting=csv.QUOTE_ALL)
ave_bal_per_ep_wr.writerow(ave_bal_per_ep)
ave_bal_per_ep_file.close()

# Running reward
running_rwd_tot_file = open(folder_name_train+'/running_rwd_tot.csv', 'w')
running_rwd_tot_wr = csv.writer(running_rwd_tot_file, quoting=csv.QUOTE_ALL)
running_rwd_tot_wr.writerow(running_rwd_tot)
running_rwd_tot_file.close()

# Running auto-correlation
running_rwd_auto_tot_file = open(folder_name_train+'/running_rwd_auto_tot.csv', 'w')
running_rwd_auto_tot_wr = csv.writer(running_rwd_auto_tot_file, quoting=csv.QUOTE_ALL)
running_rwd_auto_tot_wr.writerow(running_rwd_auto_tot)
running_rwd_auto_tot_file.close()

# Running cross-correlation
running_rwd_cross_tot_file = open(folder_name_train+'/running_rwd_cross_tot.csv', 'w')
running_rwd_cross_tot_wr = csv.writer(running_rwd_cross_tot_file, quoting=csv.QUOTE_ALL)
running_rwd_cross_tot_wr.writerow(running_rwd_cross_tot)
running_rwd_cross_tot_file.close()

# Running balance
running_bal_tot_file = open(folder_name_train+'/running_bal_tot.csv', 'w')
running_bal_tot_wr = csv.writer(running_bal_tot_file, quoting=csv.QUOTE_ALL)
running_bal_tot_wr.writerow(running_bal_tot)
running_bal_tot_file.close()

# Test results -- average auto-correlation
ave_perf_auto_file = open(folder_name_test+'/test_ave_auto.csv', 'w')
ave_perf_auto_wr = csv.writer(ave_perf_auto_file, quoting=csv.QUOTE_ALL)
ave_perf_auto_wr.writerow(ave_auto_list)
ave_perf_auto_file.close()

# Test results -- average cross-correlation
ave_perf_cross_file = open(folder_name_test+'/test_ave_cross.csv', 'w')
ave_perf_cross_wr = csv.writer(ave_perf_cross_file, quoting=csv.QUOTE_ALL)
ave_perf_cross_wr.writerow(ave_cross_list)
ave_perf_cross_file.close()

# Test results -- average balance
ave_balance_file = open(folder_name_test+'/test_ave_balance.csv', 'w')
ave_balance_wr = csv.writer(ave_balance_file, quoting=csv.QUOTE_ALL)
ave_balance_wr.writerow(ave_balance_list)
ave_balance_file.close()

# Test results -- nominal auto-correlation
nom_ave_auto_file = open(folder_name_test+'/test_nom_ave_auto.csv', 'w')
nom_ave_auto_wr = csv.writer(nom_ave_auto_file, quoting=csv.QUOTE_ALL)
nom_ave_auto_wr.writerow(nom_ave_auto_list)
nom_ave_auto_file.close()

# Test results -- nominal cross-correlation
nom_ave_cross_file = open(folder_name_test+'/test_nom_ave_cross.csv', 'w')
nom_ave_cross_wr = csv.writer(nom_ave_cross_file, quoting=csv.QUOTE_ALL)
nom_ave_cross_wr.writerow(nom_ave_cross_list)
nom_ave_cross_file.close()

# Test results -- nominal balance
nom_ave_balance_file = open(folder_name_test+'/test_nom_ave_balance.csv', 'w')
nom_ave_balance_wr = csv.writer(nom_ave_balance_file, quoting=csv.QUOTE_ALL)
nom_ave_balance_wr.writerow(nom_ave_balance_list)
nom_ave_balance_file.close()

# Key results -- auto/cross/balance
results_file = open(folder_name_test+'/results.txt', 'w')
results_file_lines = ['average auto: ' + str(ave_perf_auto) + '\n',
                    'average cross: ' + str(ave_perf_cross) + '\n',
                    'average balance: ' + str(ave_balance) + '\n', 
                    'nominal auto: ' + str(nom_ave_auto) + '\n',
                    'nominal cross: ' + str(nom_ave_cross) + '\n',
                    'nominal balance: ' + str(nom_ave_balance) + '\n']
results_file.writelines(results_file_lines)
results_file.close()

# Save code
nom_code_file = open(folder_name+'/nom_code.csv', 'w')
nom_code_wr = csv.writer(nom_code_file, quoting=csv.QUOTE_ALL)
nom_code_wr.writerow(nominal_code_family)
nom_code_file.close()



##########################################################################################
#################################                        #################################
#################################        Plotting        #################################
#################################                        #################################
##########################################################################################

plt.figure(figsize=[7,4])
plt.plot(ave_rwd_per_ep)
plt.xlabel('batch number')
plt.ylabel('average reward')
plt.title('Average reward per batch (' + str(batch_size) + ' episodes) -- ' + str(n) + ' bits, ' + str(m) + ' seqs')
plt.grid()
plt.savefig(folder_name_plot+'/ave_rwd.png')

plt.figure(figsize=[7,4])
plt.plot(ave_auto_per_ep)
plt.xlabel('batch number')
plt.ylabel('average mean sqr auto-corr')
plt.title('Average auto-correlation cost per batch (' + str(batch_size) + ' episodes) -- ' + str(n) + ' bits, ' + str(m) + ' seqs')
plt.grid()
plt.savefig(folder_name_plot+'/ave_auto.png')

plt.figure(figsize=[7,4])
plt.plot(ave_cross_per_ep)
plt.xlabel('batch number')
plt.ylabel('average mean sqr cross-corr')
plt.title('Average cross-correlation cost per batch (' + str(batch_size) + ' episodes) -- ' + str(n) + ' bits, ' + str(m) + ' seqs')
plt.grid()
plt.savefig(folder_name_plot+'/ave_cross.png')

plt.figure(figsize=[7,4])
plt.plot(ave_bal_per_ep)
plt.xlabel('batch number')
plt.ylabel('average balance')
plt.title('Average balance per batch (' + str(batch_size) + ' episodes) -- ' + str(n) + ' bits, ' + str(m) + ' seqs')
plt.grid()
plt.savefig(folder_name_plot+'/ave_bal.png')

# plt.figure(figsize=[7,4])
# plt.hist(ave_rwd_per_ep, bins=50)
# plt.xlabel('average reward')
# plt.ylabel('frequency')
# plt.title('Histogram of average rewards per batch -- '  + str(n) + ' bits, ' + str(m) + ' seqs')
# plt.grid()

plt.figure(figsize=[7,4])
plt.plot(running_rwd_tot)
plt.xlabel('batch number')
plt.ylabel('running average reward')
plt.title('Running exponential average reward -- ' + str(n) + ' bits, ' + str(m) + ' seqs')
plt.grid()
plt.savefig(folder_name_plot+'/running_rwd.png')


plt.figure(figsize=[7,4])
plt.plot(running_rwd_auto_tot)
plt.xlabel('batch number')
plt.ylabel('running average mean sqr auto-corr')
plt.title('Running exponential average auto-corr cost -- ' + str(n) + ' bits, ' + str(m) + ' seqs')
plt.grid()
plt.savefig(folder_name_plot+'/running_rwd_auto.png')


plt.figure(figsize=[7,4])
plt.plot(running_rwd_cross_tot)
plt.xlabel('batch number')
plt.ylabel('running average mean sqr cross-corr')
plt.title('Running exponential average cross-corr cost -- ' + str(n) + ' bits, ' + str(m) + ' seqs')
plt.grid()
plt.savefig(folder_name_plot+'/running_rwd_cross.png')

plt.figure(figsize=[7,4])
plt.plot(running_bal_tot)
plt.xlabel('batch number')
plt.ylabel('running average balance')
plt.title('Running exponential average balance -- ' + str(n) + ' bits, ' + str(m) + ' seqs')
plt.grid()
plt.savefig(folder_name_plot+'/running_bal.png')