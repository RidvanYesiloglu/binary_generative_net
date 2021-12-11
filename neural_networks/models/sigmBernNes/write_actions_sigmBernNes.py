import os
import torch
import numpy as np
import functions.verify_correlations as corr_fns
import functions.objectives as objectives
import time
from models.sigmBernNes import plot_sigmBernNes as plt_model
import functions.verify_correlations as corr_fns
import functions.create_initialization_from_smaller_opt as cr_sp_ini
# inps_dict: {'save_folder': save_folder, 'args':args, 'repr_str':repr_str, 'device':device}
# outs_dict: {'mean_f_zk':mean_f_zk, 'log_mean_f_zk':log_mean_f_zk, 'final_f_zk_vals':final_f_zk_vals, 'final_thetas':final_thetas}
def preallruns_actions(inps_dict):
    args = inps_dict['args']
    mean_f_zk = 0
    log_mean_f_zk = 0
    
    main_logs = open(os.path.join(inps_dict['save_folder'], 'main_logs.txt'), "a")
    main_logs.write('Runset Name: {}, Individual Run No: {}\n'.format(args.runsetName, args.indRunNo))
    main_logs.write('Configuration: {}\n'.format(inps_dict['repr_str']))
    if torch.cuda.is_available():
        main_logs.write('GPU Total Memory [GB]: {}\n'.format(torch.cuda.get_device_properties(0).total_memory/1e9))
    else:
        main_logs.write('Using CPU.\n')
    main_logs.close()
        
    final_f_zk_vals = np.zeros((args.noRuns))
    final_thetas = np.zeros((args.noRuns, args.K, args.N))
    np.save(os.path.join(inps_dict['save_folder'],'final_f_zk_vals'),final_f_zk_vals)
    
    no_parts=1
    codes_inds_1 = corr_fns.create_codes_inds_1(args.K, args.N, args.smps, no_parts, inps_dict['device'])
    codes_inds_2 = corr_fns.create_codes_inds_2(args.K, args.N, args.smps, no_parts, inps_dict['device'])
    
    preallruns_dict = {'mean_f_zk':mean_f_zk, 'log_mean_f_zk':log_mean_f_zk, 'final_f_zk_vals':final_f_zk_vals, 'final_thetas':final_thetas, \
                 'codes_inds_1':codes_inds_1, 'codes_inds_2':codes_inds_2}
    return preallruns_dict


# inps_dict: {'save_folder': save_folder, 'args':args, 'repr_str':repr_str, 'run_number':run_number, 'model': model}
# outs_dict: 
def prerun_i_actions(inps_dict, preallruns_dict):
    args =inps_dict['args']
    init_thetas = inps_dict['model']()
    device = inps_dict['device']
    
    no_parts=1
    codes_inds_1_1 = corr_fns.create_codes_inds_1(args.K, args.N, 1, no_parts, device)
    codes_inds_2_1 = corr_fns.create_codes_inds_2(args.K, args.N, 1, no_parts, device)
    init_f_zk = objectives.calc_obj(1-2*(init_thetas>0.5), args.obj, \
                                    cons_dict = {'codes_inds_1':codes_inds_1_1, 'codes_inds_2':codes_inds_2_1}).item()
    init_thetas_str = "Run no: {}\n".format(inps_dict['run_number'])+"Initial thetas:\n" + str(np.asarray(init_thetas.tolist())) + '\n'
    init_f_zk_str = 'Initial f_zk with thresholded thetas: {:.4f} (log:{:.4f}) \n'.format(init_f_zk, np.log(init_f_zk))
    
    r_logs = open(os.path.join(inps_dict['save_folder'], 'logs_{}.txt'.format(inps_dict['run_number'])), "a")
    r_logs.write('Runset Name: {}, Individual Run No: {}\n'.format(args.runsetName, args.indRunNo))
    r_logs.write('Configuration: {}\n'.format(inps_dict['repr_str']))
    r_logs.write(init_thetas_str)
    r_logs.write(init_f_zk_str)
    r_logs.close()
    
    main_logs = open(os.path.join(inps_dict['save_folder'], 'main_logs.txt'), "a")
    main_logs.write(init_thetas_str)
    main_logs.write(init_f_zk_str)
    main_logs.close()
    

def print_freq_actions(inps_dict):
    args =inps_dict['args']
    print('Epoch:', inps_dict['t']+1, ', Thetas:')
    print(inps_dict['theta_ts'])
    print('Mean {}:'.format(args.obj), inps_dict['f_zk'].mean().item(), ', log:', np.log(inps_dict['f_zk'].mean().item()))
    
def write_freq_actions(inps_dict):
    args =inps_dict['args']
    end_time = time.time()
    r_logs = open(os.path.join(inps_dict['save_folder'], 'logs_{}.txt'.format(inps_dict['run_number'])), "a")
    r_logs.write('Epoch: '+ str(inps_dict['t']+1)+ ', Time: ' + str(end_time-inps_dict['start_time']) + ', Thetas:\n')
    r_logs.write(str(inps_dict['theta_ts']) + '\nMean {}: '.format(args.obj) + str(inps_dict['f_zk'].mean().item()) + '\n')
    r_logs.write('Log: ' + str(np.log(inps_dict['f_zk'].mean().item())) + '\n')
    start_time = time.time()
    r_logs.write('Mean Fzk: {}, Mean Logsum: {}, Main loss term {}, Wd loss term: {}\n'.format(\
        inps_dict['f_zk'].mean(), torch.log(inps_dict['pi_zk']).sum((1,2)).mean(), inps_dict['main_loss_term'], inps_dict['wd_loss_term']))
    r_logs.write(str(inps_dict['model'].is_converged()[1]) + ' not converged yet.\n')
    r_logs.close()
    plt_model.plot_change_of_objective(inps_dict['f_zks_r'], args.obj, args.K, args.N, inps_dict['run_number'], True, inps_dict['save_folder'])
    plt_model.plot_change_of_loss(inps_dict['losses_r'], args.obj, args.K, args.N, inps_dict['run_number'], True, inps_dict['save_folder'])
    return {'start_time':start_time}


def postrun_i_actions(inps_dict, preallruns_dict):
    args =inps_dict['args']
    thetas = inps_dict['thetas']
    final_f_zk_vals = preallruns_dict['final_f_zk_vals']
    run_number = inps_dict['run_number']
    final_thetas = preallruns_dict['final_thetas']
    theta_ts = inps_dict['theta_ts']
    save_folder = inps_dict['save_folder']
    f_zks_r = inps_dict['f_zks_r']
    losses_r = inps_dict['losses_r']
    device = inps_dict['device']

    # Compute final code set by hardthresholding thetas (they should be almost binary by now)
    final_zk_pm1 = (1-2*(thetas>0.5))
    no_parts=1
    codes_inds_1_1 = corr_fns.create_codes_inds_1(args.K, args.N, 1, no_parts, device)
    codes_inds_2_1 = corr_fns.create_codes_inds_2(args.K, args.N, 1, no_parts, device)
    final_f_zk = objectives.calc_obj(final_zk_pm1, args.obj, \
                                    cons_dict = {'codes_inds_1':codes_inds_1_1, 'codes_inds_2':codes_inds_2_1}).item()
    print('**************')
    print('Final Thetas: ', thetas.cpu().detach().numpy())
    print('FINAL {}: {:.5f} (log: {:.5f})'.format(args.obj, final_f_zk, np.log(final_f_zk)))
    preallruns_dict['mean_f_zk'] += final_f_zk/args.noRuns
    preallruns_dict['log_mean_f_zk'] += np.log(final_f_zk)/args.noRuns
    print('******************************************')
    
    preallruns_dict['final_f_zk_vals'][run_number] = final_f_zk
    np.save(os.path.join(save_folder,'final_f_zk_vals'), preallruns_dict['final_f_zk_vals'])
    preallruns_dict['final_thetas'][run_number] = theta_ts
    np.save(os.path.join(save_folder,'final_thetas'),preallruns_dict['final_thetas'])
    
    outs_log ='Final Thetas:\n' + str(thetas.cpu().detach().numpy()) + '\nFinal Thresholded Thetas:\n' + str(final_zk_pm1.cpu().detach().numpy()) \
    + '\nFINAL {}: {:.5f} (log: {:.5f})\n*****************************************\n'.format(args.obj,final_f_zk,np.log(final_f_zk))

    r_logs = open(os.path.join(save_folder, 'logs_{}.txt'.format(run_number)), "a")
    r_logs.write(outs_log)
    r_logs.close()
    
    main_logs = open(os.path.join(save_folder, 'main_logs.txt'), "a")
    main_logs.write(outs_log)
    main_logs.close()
    
    plt_model.plot_change_of_objective(f_zks_r, args.obj, args.K, args.N, run_number, True, save_folder)
    plt_model.plot_change_of_loss(losses_r, args.obj, args.K, args.N, run_number, True, save_folder)
    
    no_parts=1
    codes_inds_1_1 = corr_fns.create_codes_inds_1(args.K, args.N, 1, no_parts, device)
    codes_inds_2_1 = corr_fns.create_codes_inds_2(args.K, args.N, 1, no_parts, device)
    plt_model.plot_au(corr_fns.autocorr_even_ft(final_zk_pm1).detach().cpu().numpy(), args.K, args.N, save_folder, run_number)
    plt_model.plot_cc(corr_fns.crosscorr_even_ft(final_zk_pm1,codes_inds_1_1,codes_inds_2_1).detach().cpu().numpy(), args.K, args.N, save_folder, run_number)

def postallruns_actions(inps_dict, preallruns_dict):
    args =inps_dict['args']
    save_folder = inps_dict['save_folder']
    mean_f_zk = preallruns_dict['mean_f_zk']
    log_mean_f_zk = preallruns_dict['log_mean_f_zk']
    final_thetas = preallruns_dict['final_thetas']
    final_f_zk_vals = preallruns_dict['final_f_zk_vals']
    print("All f_zk_vals", final_f_zk_vals)
    main_log = 'Mean f_zk over all runs: {:.6f} (log: {:.7f})\n'.format(mean_f_zk, np.log(mean_f_zk))
    main_log += 'Log Mean f_zk over all runs: {:.6f} (exp: {:.7f})\n'.format(log_mean_f_zk, np.exp(log_mean_f_zk))
    print(main_log)
    print('********************************TRAINING ENDED**********************************************')
    main_logs = open(os.path.join(save_folder, 'main_logs.txt'), "a")
    main_logs.write(main_log)
    main_logs.close()
    
    if (args.ini == np.asarray(['sp1','sp2'])).sum() > 0:
        smaller_opt_pm1 = 1.0-2.0*(final_thetas[final_f_zk_vals.argmin()]>0.5)
        print('This small codeset: ' + str(smaller_opt_pm1))
        init_param = cr_sp_ini.main(smaller_opt_pm1, args.ini)
        print('Init param for larger: ' + str(init_param))
        print('Shape: ' + str(init_param.shape))
        np.save(os.path.join(save_folder,'init_param_for_2N_{}.npy'.format(args.ini)), init_param)
