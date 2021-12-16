import runset_train.parameters as parameters
import torch
import torch.optim as optim
import torch.distributions.bernoulli as Bernoulli
import numpy as np
import math
import os
import errno
#from tqdm import tqdm #(for time viewing)
import time
import functions.objectives as objectives #objectives.calc_obj(zk_pm1, obj_name, cons_dict = {})
import models.sigmBernNes.write_actions_sigmBernNes as wr_acts 
import models.sigmBernNes.model_sigmBernNes as mod
from pathlib import Path
def update_runset_summary(args, runset_folder):
    reads = ""
    for i in range(args.totalInds):
        if Path(os.path.join(runset_folder, 'ind_runs', 'run{}.txt'.format(i+1))).is_file():
            run_i_file = open(os.path.join(runset_folder, 'ind_runs', 'run{}.txt'.format(i+1)), "r+")
            reads += run_i_file.read() + "\n"
            run_i_file.close()
    summary_file = open(os.path.join(runset_folder, 'runset_{}.txt'.format(args.runsetName)), "w")
    summary_file.write(reads)
    summary_file.close()

def main():
    params_dict = parameters.decode_arguments_dictionary('params_dictionary')
    working_dir = '/scratch/groups/gracegao/'+params_dict.proj_name
    args = parameters.get_arguments(params_dict)
    repr_str = parameters.create_repr_str(args, [info.name for info in params_dict.param_infos])
    
    dtype = torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.set_printoptions(precision=7)
    print_freq = 250000 # print thetas once in "print_frequency" epochs 
    write_freq = 10000 # print thetas once in "print_frequency" epochs
    
    save_folder = os.path.join(working_dir, 'detailed_results', args.obj, "{}_{}".format(args.K, args.N), args.net, repr_str)
    if not os.path.exists(save_folder):
        try:
            os.makedirs(save_folder)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
                
    runset_folder = os.path.join(working_dir, 'runset_summaries', args.runsetName)
    if not os.path.exists(os.path.join(runset_folder, 'ind_runs')):
        try:
            os.makedirs(os.path.join(runset_folder, 'ind_runs'))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    if Path(os.path.join(runset_folder, 'ind_runs', 'run{}.txt'.format(args.indRunNo))).is_file():
        ind_run_sit_log = open(os.path.join(runset_folder, 'ind_runs', 'run{}.txt'.format(args.indRunNo)), "r+")
        prev_sit_log = ind_run_sit_log.read()
        ind_run_sit_log.close()
    else:
        prev_sit_log = ""
    ind_run_sit_log = open(os.path.join(runset_folder, 'ind_runs', 'run{}.txt'.format(args.indRunNo)), "w+")
    ind_run_sit_log.write(prev_sit_log+'Ind Run: {}, Conf: {}, Situ: Just started'.format(args.indRunNo, repr_str))
    ind_run_sit_log.close()
    update_runset_summary(args, runset_folder)
    # inps_dict_pre_tt: {'save_folder': save_folder, 'args':args, 'repr_str':repr_str, 'device':device}
    # outs_dict_pre_t: {'mean_f_zk':mean_f_zk, 'log_mean_f_zk':log_mean_f_zk, 'final_f_zk_vals':final_f_zk_vals, 'final_thetas':final_thetas, \
                     #'codes_inds_1':codes_inds_1, 'codes_inds_2':codes_inds_2}
    preallruns_dict = wr_acts.preallruns_actions({'save_folder': save_folder, 'args':args, 'repr_str':repr_str, 'device':device})
    
    for run_number in range(args.noRuns):
        if (args.ini == np.asarray(['sp1','sp2'])).sum() > 0:
            args.N = args.N//2
            nover2reprstr = parameters.create_repr_str(args, [info.name for info in params_dict.param_infos])
            nover2folder = os.path.join(working_dir, 'detailed_results', args.obj, "{}_{}".format(args.K, args.N), args.net, nover2reprstr)
            args.N = args.N * 2
            model = mod.SigmBernNes({'save_folder': save_folder, 'args':args, 'repr_str':repr_str, 'nover2folder':nover2folder}, device, dtype)
        else:
            model = mod.SigmBernNes({'save_folder': save_folder, 'args':args, 'repr_str':repr_str}, device, dtype)
                
        optimizer = optim.Adam(model.parameters(), args.lr) #optim.SGD(model.parameters(), lr=learning_rate)
        
        
        sum_log_base = prev_sit_log+'\nInd Run: {}, Conf: {}, Situ: '.format(args.indRunNo, repr_str)
        for prev_run_ind in range(0,run_number):
            sum_log_base += 'Tr {} Res: {:.8f}, '.format(prev_run_ind, preallruns_dict['final_f_zk_vals'][prev_run_ind])
        last_log = 'Tr. {} starting'.format(run_number)
        ind_run_sit_log = open(os.path.join(runset_folder, 'ind_runs', 'run{}.txt'.format(args.indRunNo)), "w+")
        ind_run_sit_log.write(sum_log_base + last_log)
        ind_run_sit_log.close()
        update_runset_summary(args, runset_folder)
        
        wr_acts.prerun_i_actions({'save_folder': save_folder, 'args':args, 'repr_str':repr_str, 'run_number':run_number, 'model': model, 'device':device}, preallruns_dict)
        f_zks_r = []
        losses_r = [] 
        start_time = time.time()
        for t in range(args.eps):
            # Forward pass: compute thetas.
            thetas = model()
            # Create a distribution with the thetas 
            dist_theta = Bernoulli.Bernoulli(thetas)
            # Sample from the distribution:
            zk = dist_theta.sample(torch.Size([args.smps])).detach()
            # Convert samples from {0,1} to {1,-1}
            zk_pm1 = (1-2*zk).type(torch.int8)
            # Compute objective function (f_zk) on the converted samples:
            f_zk = objectives.calc_obj(zk_pm1, args.obj, \
                                        cons_dict = {'codes_inds_1':preallruns_dict['codes_inds_1'], 'codes_inds_2':preallruns_dict['codes_inds_2']})
    
            # Compute expression zk*theta+(1-zk)*(1-theta):
            pi_zk = zk*(thetas.unsqueeze(0))+(1-zk)*(1-(thetas.unsqueeze(0)))
            # Calculate loss by averaging over samples:
            main_loss_term = (f_zk * (torch.log(pi_zk).sum((1,2)))).mean()
            if (args.wdC == 0):
                loss =  main_loss_term
                wd_loss_term = None
            else:
                wd_loss_term = args.wdC*(model.w**2).mean() / math.pow(10,args.lg10WdDiv)
                loss = main_loss_term + wd_loss_term
            
            # Backpropagate:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Save mean f_zk, the loss, and the theta
            f_zks_r.append(f_zk.mean().item())
            losses_r.append(loss.item())
    
            theta_ts = np.asarray(thetas.tolist())
            # Print
            if (t+1) % print_freq == 0:
                wr_acts.print_freq_actions({'args':args, 't':t, 'theta_ts':theta_ts, 'f_zk':f_zk})
            if (t+1) % write_freq == 0:
                last_log = 'Tr {} Ep {} Res: {:.8f}'.format(run_number, t+1, f_zks_r[-1])
                ind_run_sit_log = open(os.path.join(runset_folder, 'ind_runs', 'run{}.txt'.format(args.indRunNo)), "w+")
                ind_run_sit_log.write(sum_log_base + last_log)
                ind_run_sit_log.close()
                update_runset_summary(args, runset_folder)
                
                write_freq_dict = wr_acts.write_freq_actions({'args':args, 't':t, 'theta_ts':theta_ts, 'f_zk':f_zk, 'pi_zk':pi_zk, 'start_time':start_time, \
                    'model': model, 'save_folder': save_folder, 'run_number':run_number, 'main_loss_term':main_loss_term, 'wd_loss_term':wd_loss_term,\
                        'f_zks_r':f_zks_r, 'losses_r':losses_r})
                start_time = write_freq_dict['start_time']
            if model.is_converged()[0]:
                r_logs = open(os.path.join(save_folder, 'logs_{}.txt'.format(run_number)), "a")
                r_logs.write('Ended early at epoch {} due to convergence.\n'.format(t+1))
                break
        wr_acts.postrun_i_actions({'args':args, 'thetas':thetas, 'run_number':run_number, 'theta_ts':theta_ts, 'save_folder':save_folder, 'f_zks_r':f_zks_r, 'losses_r':losses_r, 'device':device},\
                                  preallruns_dict)
        last_log = 'Tr {} Res: {:.8f}'.format(run_number, preallruns_dict['final_f_zk_vals'][run_number])
        ind_run_sit_log = open(os.path.join(runset_folder, 'ind_runs', 'run{}.txt'.format(args.indRunNo)), "w+")
        ind_run_sit_log.write(sum_log_base + last_log)
        ind_run_sit_log.close()
        update_runset_summary(args, runset_folder)
    wr_acts.postallruns_actions({'args':args, 'save_folder':save_folder}, preallruns_dict)
    mean_log = ' Mean Res: {:.8f}'.format(preallruns_dict['final_f_zk_vals'].mean())
    ind_run_sit_log = open(os.path.join(runset_folder, 'ind_runs', 'run{}.txt'.format(args.indRunNo)), "w+")
    ind_run_sit_log.write(sum_log_base + last_log + mean_log)
    ind_run_sit_log.close()
    update_runset_summary(args, runset_folder)
if __name__ == "__main__":
    main()
