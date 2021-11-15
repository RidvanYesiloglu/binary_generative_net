# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:48:06 2021

@author: ridva
"""
import numpy as np
import os
# Creates initialization from (1,N) to (1,2*N)
def create_initialization_from_smaller_opt(smaller_opt, mode):
    K,N = smaller_opt.shape
    fft_smaller_opt = np.fft.fft(smaller_opt) # dimension (K,N)    
    print('FT Small: ' + str(fft_smaller_opt))
    # (np.abs(ft_smaller_opt)**2).sum() is N**2
    mult_fac = np.sqrt(2)
    const_odd = np.sqrt(2*N)
    fft_larger = np.ones((K,2*N))*const_odd
    for i in range(N):
        fft_larger[:,2*i] = fft_smaller_opt[:,i]*mult_fac
    # (np.abs(fft_ex)**2).sum() should be (2*N)**2
    print('FT Large: ' + str(fft_larger))
    inv_fft_larger = np.fft.ifft(fft_larger) # dimension (1,2*N)
    print('Initial inv: ' + str(inv_fft_larger))
    if mode==1:
        inv_fft_larger[inv_fft_larger<-1] = -1
        inv_fft_larger[inv_fft_larger>1] = 1
        inv_fft_larger *= 0.6
        inv_fft_larger = inv_fft_larger/2.0 + 0.5
        print('Initial sigmoid: ' + str(inv_fft_larger))
        init_param = np.log(inv_fft_larger/(1-inv_fft_larger))
    elif mode == 2:
        inv_fft_larger[inv_fft_larger<=0] = 0
        inv_fft_larger[inv_fft_larger>0] = 1
        y = torch.from_numpy(find_rnd_worst_codeset(K, N, obj_no))
        x = torch.randn((K, N), device=device, dtype=dtype)
        self.w = nn.Parameter((y==1)*((x<0)*(-x)+(x>=0)*x) + (y==-1)*((x<0)*x+(x>=0)*(-x)), requires_grad=True)
    return init_param
print('PRINT..')
K = int(input('Write the number of codes (satellites): '))
N = int(input('Write the period of codes: '))
obj_no = int(input('Write the objective function no: '))
# Hyperparameters
sg_slp = 1.0
init_mode = int(input('Which init mode to use? (0: worst initialization, 1: random initalization) ')) # 0: worst initialization, 1: random initalization
learning_rate = 0.005 #0.3 idi, hizlandirmak icin 0.35 yaptim
epochs = 500000
samps = 500 # no of samplings at each epoch for monte carlo gradient estimation
run_name = str(input('Write run name: '))
params_str = 'init{}_sgs{:.1f}_lr{:.2f}_eps{}_smps{}_run{}'.format(init_mode,sg_slp, learning_rate,epochs,samps,run_name)
save_folder = os.path.join('data', 'obj'+str(obj_no), 'sigm_bern_nes_model', str(K)+'_'+str(N),params_str)
theta_ts = 1-2*(np.load(os.path.join(save_folder, 'final_codesets.npy'))>= 0.5)
fzks = np.load('final_f_zk_vals.npy')
print(fzks)
print(theta_ts)
print('Shape: ' + str(theta_ts.shape))
print('Min log: ' + str(np.log(fzks.min())) + ', argmin: ' + str(fzks.argmin()))

print('Small codeset: ' + str(theta_ts[fzks.argmin()]))
init_param = create_initialization_from_smaller_opt(theta_ts[fzks.argmin()], 1)
print(init_param)
np.save('init_param_{}_{}.npy'.format(K,N), init_param)