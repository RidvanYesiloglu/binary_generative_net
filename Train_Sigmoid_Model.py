import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions.bernoulli as Bernoulli
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import errno
import itertools
import sys
from matplotlib.ticker import FormatStrFormatter
#from tqdm import tqdm

# model definition
class Sigmoid_Model(nn.Module):
    # Init mode = {0: worst initialization, 1: random initialization}
    def __init__(self, params, sg_slope, device, dtype):
        super(Sigmoid_Model, self).__init__()
        self.sg_slope = sg_slope
        self.device = device
        self.dtype = dtype
        if params.init_mode == 'cns_ini':
            y = torch.from_numpy(1-2*np.ones((params.K,params.N))*np.random.randint(2, size=(params.K,1))).to(device)
            x = torch.randn((params.K, params.N), device=device, dtype=dtype)
            self.w = nn.Parameter((y==1)*((x<0)*(-x)+(x>=0)*x) + (y==-1)*((x<0)*x+(x>=0)*(-x)), requires_grad=True)
        elif params.init_mode == 'rnd_ini':
            self.w = nn.Parameter(torch.randn((params.K, params.N), device=device, dtype=dtype), requires_grad=True)
        elif (params.init_mode == np.asarray(['gold_ini','weil_ini','gw_c_ini'])).sum() > 0:
            self.w = nn.Parameter(torch.from_numpy(np.load('{}_param_for_{}_{}.npy'.format(params.init_mode,params.K,int(params.N/2)))).to(device),  requires_grad=True)
        elif (params.init_mode == np.asarray(['sp1_ini','sp2_ini'])).sum() > 0:
            self.w = nn.Parameter(torch.from_numpy(np.load('{}_wde{}_param_for_{}_{}.npy'.format(params.init_mode,params.wd_exists,params.K,int(params.N/2)))).to(device),  requires_grad=True)
        else:
            assert 1==2
        self.sigmoid_layer = nn.Sigmoid()
    # forward propagate input
    def forward(self):
        self.thetas = self.sigmoid_layer(self.sg_slope*self.w)
        return self.thetas
        
class Parameters():
    def __init__(self):
        super(Parameters,self).__init__()
    def set_K(self,K):
        self.K=K
    def set_N(self,N):
        self.N=N
    def set_obj_name(self,obj_name):
        self.obj_name=obj_name
    def set_init_mode(self,init_mode):
        self.init_mode=init_mode
    def set_run_name(self,run_name):
        self.run_name = run_name
    def set_wd_exists(self,wd_exists):
        self.wd_exists = wd_exists
        
# codeset: (batch_size,K,N), output: (batch_size,K,N)
# (zero-delays are removed if called with .narrow(2,1,N-1))
def autocorr(codeset, roll_inds):
    if len(codeset.shape) == 2:
        codeset = codeset.unsqueeze(0)
    assert (len(codeset.shape) == 3)
    [batch_size, K, N] = codeset.shape
    return (torch.gather(codeset.unsqueeze(-1).repeat(1,1,1,N),2,roll_inds.repeat(batch_size,1,1,1))*codeset.unsqueeze(-1)).sum(2)/float(N)

# codeset: (batch_size,K,N), output: (batch_size,K*(K-1)/2,N)
def crosscorr(codeset,codes_inds,roll_inds):
    if len(codeset.shape) == 2:
        codeset = codeset.unsqueeze(0)
    assert (len(codeset.shape) == 3)
    [batch_size, K, N] = codeset.shape
    return (torch.gather(torch.gather(codeset,1,codes_inds.narrow(2,1,1).repeat(batch_size,1,N)).unsqueeze(-1).repeat(1,1,1,N),2,roll_inds.repeat(batch_size,1,1,1))*torch.gather(codeset,1,codes_inds.narrow(2,0,1).repeat(batch_size,1,N)).unsqueeze(-1)).sum(2)/float(N)

def plot_change_of_objective(f_zks, obj_name, K, N, runno, to_save=False, save_folder=None):
    x1 = np.arange(1,len(f_zks)+1)
    y1 = np.asarray(f_zks)
    
    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle('Change of the Objective Value w.r.t. Epoch No (Objective {}, (K,N)=({},{}))'.format(obj_name,K,N))
    
    ax1.plot(x1, y1)
    ax1.set_ylabel('Objective Value')
    ax1.set_xlabel('Epoch Number')
    plt.show()
    if to_save:
        plt.savefig(os.path.join(save_folder, 'obj_vs_ep_{}.png'.format(runno)),bbox_inches='tight')
    return plt

def plot_change_of_loss(losses, obj_name, K, N, runno, to_save=False, save_folder=None):
    x1 = np.arange(1,len(losses)+1)
    y1 = np.asarray(losses)
    
    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle('Change of the Loss Value w.r.t. Epoch No (Objective {}, (K,N)=({},{}))'.format(obj_name,K,N))
    
    ax1.plot(x1, y1)
    ax1.set_ylabel('Loss Value')
    ax1.set_xlabel('Epoch Number')
    plt.show()
    if to_save:
        plt.savefig(os.path.join(save_folder, 'loss_vs_ep_{}.png'.format(runno)),bbox_inches='tight')
    return plt

def calc_f_zk(zk_pm1, obj_name, N, roll_inds_au,codes_inds_cr,roll_inds_cr):
    # List: MeMeAC, MaMeAC, MEWSD, MEWSD_d, MF, MF_d, ELW
    if obj_name == 'MeMeAC': # MeMeAC
        f_zk = (autocorr(zk_pm1,roll_inds_au).narrow(2,1,N-1).pow(2).mean((1,2)) + crosscorr(zk_pm1,codes_inds_cr,roll_inds_cr).pow(2).mean((1,2)))/2.0
    elif obj_name == 'MaMeAC': # MaMeAC
        f_zk = torch.max(autocorr(zk_pm1,roll_inds_au).narrow(2,1,N-1).pow(2).mean((1,2)), crosscorr(zk_pm1,codes_inds_cr,roll_inds_cr).pow(2).mean((1,2)))
    elif obj_name == 'MEWSD': # MEWSD
        f_zk = 1
    elif obj_name == 'MEWSD_d': # MEWSD_d
        f_zk = 1
    elif obj_name == 'MF': # MF
        f_zk = 1
    elif obj_name == 'MF_d': # MF_d
        f_zk = 1
    elif obj_name == 'ELW': # ELW 
        f_zk = 1
    elif obj_name == 'l_MeMeAC': # l_MeMeAC
        f_zk = torch.log((autocorr(zk_pm1,roll_inds_au).narrow(2,1,N-1).pow(2).mean((1,2))  + crosscorr(zk_pm1,codes_inds_cr,roll_inds_cr).pow(2).mean((1,2)))/2.0)
    else:
        assert 1==2
    return f_zk
    
def plot_au(fin_autocorr_np, K, N, save_folder, run_number):
    au = np.square(fin_autocorr_np)
    fig, ax2 = plt.subplots(figsize=(16, 8), dpi=80)
    au_mean_at_e_d = au.mean((0,1))[1:]
    au_min_at_e_d = au.min((0,1))[1:]
    au_max_at_e_d = au.max((0,1))[1:]
    xs = np.arange(1, N, 1)
    plt.plot(xs,au_max_at_e_d,'r.-')
    plt.plot(xs,au_mean_at_e_d,'m.-')
    plt.plot(xs,au_min_at_e_d,'b.-')
    plt.xlabel('Delay', fontsize=18)
    plt.xticks(fontsize=16)
    plt.ylabel('Squared Autocorrelation',fontsize=18)
    plt.yticks(fontsize=16)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.title('Max., Mean, and Min. Squared Autocorrelation Values Across Codes at Each Delay',fontsize=22)
    plt.legend(["Max Value", "Mean Value","Min Value"],fontsize=16,bbox_to_anchor=(1.2, 0.6))
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'run_{}_ac_vs_delay.png'.format(run_number)),bbox_inches='tight')  
    
    fig2, ax3 = plt.subplots(figsize=(16, 8), dpi=80)
    print('AUU: ', au)
    au_mean_f_e_c = au[:,:,1:].mean((0,2))
    au_min_f_e_c = au[:,:,1:].min((0,2))
    au_max_f_e_c = au[:,:,1:].max((0,2))
    print(au_mean_f_e_c)
    print(au_min_f_e_c)
    print(au_max_f_e_c)
    xs = np.arange(1, K+1, 1)
    plt.plot(xs,au_max_f_e_c,'r.-')
    plt.plot(xs,au_mean_f_e_c,'m.-')
    plt.plot(xs,au_min_f_e_c,'b.-')
    plt.xlabel('Code No', fontsize=18)
    plt.xticks(fontsize=16)
    plt.ylabel('Squared Autocorrelation',fontsize=18)
    plt.yticks(fontsize=16)
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.title('Max., Mean, and Min. Squared Autocorrelation Values Across Delays for Each Code',fontsize=22)
    plt.legend(["Max Value", "Mean Value","Min Value"],fontsize=16,bbox_to_anchor=(1.2, 0.6))
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'run_{}_ac_vs_code.png'.format(run_number)),bbox_inches='tight')
    
def plot_cc(fin_crosscorr_np, K, N, save_folder, run_number):
    fig, ax = plt.subplots(figsize=(16, 8), dpi=80)
    cc = np.square(fin_crosscorr_np)
    cc_mean = cc.mean((0,1))
    cc_min = cc.min((0,1))
    cc_max = cc.max((0,1))
    xs = np.arange(0, N, 1)
    plt.plot(xs,cc_max,'r.-')
    plt.plot(xs,cc_mean,'m.-')
    plt.plot(xs,cc_min,'b.-')
    plt.xlabel('Delay', fontsize=18)
    plt.xticks(fontsize=16)
    plt.ylabel('Squared Cross-correlation',fontsize=18)
    plt.yticks(fontsize=16)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.title('Max., Mean, and Min. Squared Cross-correlation Values Across Codes at Each Delay',fontsize=22)
    plt.legend(["Max Value", "Mean Value","Min Value"],fontsize=16,bbox_to_anchor=(1.2, 0.6))
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'run_{}_cc_vs_delay.png'.format(run_number)),bbox_inches='tight')  

    fig2, ax2 = plt.subplots(figsize=(16, 8), dpi=80)
    cc_mean_f_e_c = cc.mean((0,2))
    cc_min_f_e_c = cc.min((0,2))
    cc_max_f_e_c = cc.max((0,2))
    xs = np.arange(1, K*(K-1)//2 + 1, 1)
    plt.plot(xs,cc_max_f_e_c,'r.-')
    plt.plot(xs,cc_mean_f_e_c,'m.-')
    plt.plot(xs,cc_min_f_e_c,'b.-')
    plt.xlabel('No of 2 Codes', fontsize=18)
    plt.xticks(fontsize=16)
    plt.ylabel('Squared Cross-correlation',fontsize=18)
    plt.yticks(fontsize=16)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.title('Max., Mean, and Min. Squared Cross-correlation Values Across Delays for Each 2 Codes',fontsize=22)
    plt.legend(["Max Value", "Mean Value","Min Value"],fontsize=16,bbox_to_anchor=(1.2, 0.6))
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'run_{}_cc_vs_code.png'.format(run_number)),bbox_inches='tight')
    
# MAIN CODE TO EXECUTE
print('Code execution started.')
for no,param in enumerate(sys.argv):
    print(no, param)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
params = Parameters()
params.set_K(int(sys.argv[1]))#int(input('Write the number of codes (satellites): '))
params.set_N(int(sys.argv[2]))#int(input('Write the period of codes: '))
params.set_obj_name(str(sys.argv[3]))#int(input('Write the objective function no: '))
params.set_init_mode(str(sys.argv[4]))#int(input('Which init mode to use?')'
params.set_run_name(str(sys.argv[5]))#str(input('Write run name: '))
params.set_wd_exists((int(sys.argv[6])==1))
samps = int(sys.argv[7]) # no of samplings at each epoch for monte carlo gradient estimation
# Hyperparameters
sg_slp = 1.0
dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 0.005
epochs = int(sys.argv[8])
no_of_runs = int(sys.argv[9])
print_freq = 25000 # print thetas once in "print_frequency" epochs
write_freq = 25000 # print thetas once in "print_frequency" epochs

params_str = 'init{}_sgs{:.1f}_lr{:.2f}_eps{}_smps{}_run{}'.format(params.init_mode,sg_slp, learning_rate,epochs,samps,params.run_name)
save_folder = os.path.join('results', 'obj'+str(params.obj_name), 'sigm_bern_nes_model', str(params.K)+'_'+str(params.N),params_str)
if not os.path.exists(save_folder):
    try:
        os.makedirs(save_folder)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
mean_f_zk = 0
log_mean_f_zk = 0
main_logs = open(os.path.join(save_folder, 'main_logs.txt'), "a")
main_logs.write('GPU available: ' + str(torch.cuda.is_available()) + '\nConfiguration: sigm_bern_nes_model/' + params_str + '\n')
main_logs.write('T r a i n i n g     t h e     n e t w o r k...\n')
main_logs.close()
main_logs = open(os.path.join(save_folder, 'main_logs.txt'), "a")
print('\nT r a i n i n g     t h e     n e t w o r k...\n')
roll_inds_au=torch.remainder(torch.arange(params.N).unsqueeze(0).repeat(params.N,1)+torch.arange(params.N).unsqueeze(1), params.N).unsqueeze(0).repeat(params.K,1,1).unsqueeze(0).to(device)
codes_inds_cr= torch.combinations(torch.arange(params.K),2,with_replacement=False).unsqueeze(0).to(device)
roll_inds_cr=torch.remainder(torch.arange(params.N).unsqueeze(0).repeat(params.N,1)+torch.arange(params.N).unsqueeze(1), params.N).unsqueeze(0).repeat(int(params.K*(params.K-1)/2),1,1).unsqueeze(0).to(device)
final_f_zk_vals = np.zeros((no_of_runs))
final_thetas = np.zeros((no_of_runs,params.K,params.N))
np.save(os.path.join(save_folder,'final_f_zk_vals'),final_f_zk_vals)
for run_number in range(no_of_runs):
    model = Sigmoid_Model(params, sg_slp, device, dtype)
    init_thetas = model().to(device)
    np.set_printoptions(precision=4)
    optimizer = optim.Adam(model.parameters(), learning_rate) #optim.SGD(model.parameters(), lr=learning_rate)
    f_zks_r = []
    losses_r = []
    r_logs = open(os.path.join(save_folder, 'logs_{}.txt'.format(run_number)), "a")
    r_logs.write('Configuration: sigm_bern_nes_model/' + params_str + '\n')
    #outs_log.append("Run no: {}\n".format(run_number)+"Initial thetas:\n" + str(np.asarray(init_thetas.tolist())) + '\n')
    outs_log="Run no: {}\n".format(run_number)+"Initial thetas:\n" + str(np.asarray(init_thetas.tolist())) + '\n'
    
    init_f_zk = calc_f_zk(1-2*(init_thetas>0.5), params.obj_name, params.N, roll_inds_au,codes_inds_cr,roll_inds_cr).item() #initial f_zk with hard thresholding
    outs_log += 'Initial f_zk with thresholded thetas: {:.4f} (log:{:.4f}) \n'.format(init_f_zk, np.log(init_f_zk))
    r_logs.write(outs_log)
    main_logs.write(outs_log)
    main_logs.close()
    main_logs = open(os.path.join(save_folder, 'main_logs.txt'), "a")
    #main_log = "Run no: {}\n".format(run_number)+"Initial thetas:\n" + str(np.asarray(init_thetas.tolist())) + '\n'
    #main_log += 'Initial f_zk with thresholded thetas: {:.4f} (log:{:.4f}) \n'.format(init_f_zk, np.log(init_f_zk))
    print(outs_log)
    # print("Initial thetas:\n" + str(np.asarray(init_thetas.tolist())))
    # print('Initial f_zk with thresholded thetas: ', calc_f_zk(1-2*(init_thetas>0.5)).item(), '\n')
    for t in range(epochs):
        # Forward pass: compute thetas.
        thetas = model()
        # Create a distribution with the thetas
        dist_theta = Bernoulli.Bernoulli(thetas)
        # Sample from the distribution:
        zk = dist_theta.sample(torch.Size([samps])).detach()
        # Convert samples from {0,1} to {1,-1}
        zk_pm1 = 1-2*zk
        # Compute objective function (f_zk) on the converted samples:
        f_zk = calc_f_zk(zk_pm1, params.obj_name, params.N, roll_inds_au,codes_inds_cr,roll_inds_cr)
        # Compute expression zk*theta+(1-zk)*(1-theta):
        pi_zk = zk*(thetas.unsqueeze(0))+(1-zk)*(1-(thetas.unsqueeze(0)))
        # Calculate loss by averaging over samples:
        loss = (f_zk * (torch.log(pi_zk).sum((1,2)))).mean()
        
        # Backpropagate:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Save mean f_zk, the loss, and the theta
        f_zks_r.append(f_zk.mean().item())
        losses_r.append(loss.item())
        #theta_ts[run_number, t] = np.asarray(thetas.tolist())
        theta_ts = np.asarray(thetas.tolist())
        # Print
        if (t+1) % print_freq == 0:
            print('Epoch:', t+1, ', Thetas:')
            print(theta_ts)
            print('Mean {}:'.format(params.obj_name), f_zk.mean().item(), ', log:', np.log(f_zk.mean().item()))
        if t % write_freq == 0:
            outs_log = 'Epoch: '+ str(t)+ ', Thetas:\n' + str(theta_ts) + '\nMean {}: '.format(params.obj_name) + str(f_zk.mean().item()) + '\n Log: ' + str(np.log(f_zk.mean().item())) + '\n'
            r_logs.write(outs_log)
            r_logs.write(str((1-np.logical_or(theta_ts<=0.02,theta_ts>=0.98)).sum()) + ' not converged yet.\n')
            '''if (1-np.logical_or(theta_ts<=0.02,theta_ts>=0.98)).sum() <= 3:
                inds = (1-np.logical_or(theta_ts<=0.02,theta_ts>=0.98))
                for twon in range(int(np.power(2,inds.sum()))):
                    theta_thr = 1-2*(theta_ts.copy()>0.5)
                    codeset_str = str(bin(twon))[2:]
                    if len(codeset_str) < inds.sum():
                        str_to_add = '0'*(inds.sum()-len(codeset_str))
                        codeset_str = str_to_add + codeset_str
                    print(codeset_str)
                    unc = np.ones((inds.sum()))
                    for ind1 in range(inds.sum()):
                        unc[ind1] = int(1-2*int(codeset_str[ind1]))
                    theta_thr[inds>0] = unc
                    print(theta_thr)
                    f_zk = calc_f_zk(torch.from_numpy(theta_thr).to(device), params.obj_name,roll_inds_au,codes_inds_cr,roll_inds_cr)
                    print(f_zk)
                    r_logs.write('f_zk_'+str(twon+1)+': ' + str(f_zk) + ', ')
            r_logs.write('\n')'''
            r_logs.close()
            r_logs = open(os.path.join(save_folder, 'logs_{}.txt'.format(run_number)), "a")
        if np.logical_or(theta_ts<=0.02,theta_ts>=0.98).all():
            outs_log = 'Ended at epoch {}\n'.format(t+1)
            r_logs.write(outs_log)
            break
        '''elif (1-np.logical_or(theta_ts<=0.02,theta_ts>=0.98)).sum() <= 3:
            inds = (1-np.logical_or(theta_ts<=0.02,theta_ts>=0.98))
            for twon in range(int(np.power(2,inds.sum()))):
                theta_thr = 1-2*(theta_ts.copy()>0.5)
                codeset_str = str(bin(twon))[2:]
                if len(codeset_str) < inds.sum():
                    str_to_add = '0'*(inds.sum()-len(codeset_str))
                    codeset_str = str_to_add + codeset_str
                print(codeset_str)
                unc = np.ones((inds.sum()))
                for ind1 in range(inds.sum()):
                    unc[ind1] = int(1-2*int(codeset_str[ind1]))
                theta_thr[inds>0] = unc
                print(theta_thr)
                f_zk = calc_f_zk(torch.from_numpy(theta_thr).to(device), params.obj_name,roll_inds_au,codes_inds_cr,roll_inds_cr)
                print(f_zk)
                r_logs.write('f_zk_'+str(twon+1)+': ' + str(f_zk) + ', ')
            r_logs.write('\n')
            r_logs.close()
            r_logs = open(os.path.join(save_folder, 'logs_{}.txt'.format(run_number)), "a")'''
    # Compute final code set by hardthresholding thetas (they should be almost binary by now)
    final_zk_pm1 = (1-2*(thetas>0.5))
    final_f_zk = calc_f_zk(final_zk_pm1, params.obj_name, params.N, roll_inds_au,codes_inds_cr,roll_inds_cr).item()
    print('**************')
    print('Final Thetas: ', thetas.cpu().detach().numpy())
    print('FINAL {}: {:.5f} (log: {:.5f})'.format(params.obj_name, final_f_zk,np.log(final_f_zk)))
    mean_f_zk += final_f_zk/no_of_runs
    log_mean_f_zk += np.log(final_f_zk)/no_of_runs
    print('******************************************')
    final_f_zk_vals[run_number] = final_f_zk
    final_thetas[run_number] = theta_ts
    outs_log ='Final Thetas:\n' + str(thetas.cpu().detach().numpy()) + '\nFinal Thresholded Thetas:\n' + str(final_zk_pm1.cpu().detach().numpy()) \
    + '\nFINAL {}: {:.5f} (log: {:.5f})\n*****************************************\n'.format(params.obj_name,final_f_zk,np.log(final_f_zk))
    r_logs.write(outs_log)
    main_logs.write(outs_log)
    main_logs.close()
    main_logs = open(os.path.join(save_folder, 'main_logs.txt'), "a")
    r_logs.close()
    #np.save(os.path.join(save_folder,'theta_ts'),theta_ts)
    plot_change_of_objective(f_zks_r, params.obj_name, params.K, params.N, run_number, True, save_folder)
    plot_change_of_loss(losses_r, params.obj_name, params.K, params.N, run_number, True, save_folder)
    plot_au(autocorr(final_zk_pm1,roll_inds_au).detach().cpu().numpy(), params.K, params.N, save_folder, run_number)
    plot_cc(crosscorr(final_zk_pm1,codes_inds_cr,roll_inds_cr).detach().cpu().numpy(), params.K, params.N, save_folder, run_number)
np.save(os.path.join(save_folder,'final_f_zk_vals'),final_f_zk_vals)
np.save(os.path.join(save_folder,'final_thetas'),final_thetas)
main_log = 'Mean f_zk over all runs: {:.6f} (log: {:.7f})\n'.format(mean_f_zk, np.log(mean_f_zk))
main_log += 'Log Mean f_zk over all runs: {:.6f} (exp: {:.7f})\n'.format(log_mean_f_zk, np.exp(log_mean_f_zk))
print(main_log)
print('********************************TRAINING ENDED**********************************************')
main_logs.write(main_log)
main_logs.close()
# if (input('Wanna save the log files (yes=1)? ') == '1'):
#     f_logs = open(os.path.join(save_folder, 'main_logs.txt'), "w")
#     f_logs.write(main_log)
#     f_logs.close()
#     for run_number in range(no_of_runs):
#         f_logs = open(os.path.join(save_folder, 'logs_{}.txt'.format(run_number)), "w")
#         f_logs.write(outs_log[run_number])
#         f_logs.close()
#
# if (input('Wanna save theta_ts\' (yes=1)? ') == '1'):
#     np.save(os.path.join(save_folder,'theta_ts'),theta_ts)
    
# to_save_plts = (input('Wanna save the plots (yes=1)? ') == '1')

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# plt.close('all')
# for run_number in range(no_of_runs):
#     plot_change_of_objective(f_zks[run_number], params.obj_name, K, N, run_number, to_save_plts, save_folder)
#     plot_change_of_loss(losses[run_number], params.obj_name, K, N, run_number, to_save_plts, save_folder)



'''
import torch
import math
K=3
N=3
obj_no=1
rang = torch.arange(int(math.pow(2,K*N))).unsqueeze(-1).unsqueeze(-1).repeat(1,K,N)
divs = torch.pow(2, torch.arange(K*N).reshape(K,N)).unsqueeze(0)
codesets = torch.pow(-1,rang//divs)
f_zk = calc_f_zk(codesets, obj_no)
'''
