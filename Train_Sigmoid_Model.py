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

# model definition
class Sigmoid_Model(nn.Module):
    # Init mode = {0: worst initialization, 1: random initialization}
    def __init__(self, K, N, obj_no, sg_slope, init_mode, device, dtype):
        super(Sigmoid_Model, self).__init__()
        self.sg_slope = sg_slope
        self.device = device
        self.dtype = dtype
        if init_mode == 0:
            y = torch.from_numpy(find_rnd_worst_codeset(K, N, obj_no))
            x = torch.randn((K, N), device=device, dtype=dtype)
            self.w = nn.Parameter((y==1)*((x<0)*(-x)+(x>=0)*x) + (y==-1)*((x<0)*x+(x>=0)*(-x)), requires_grad=True)
        elif init_mode == 1:
            self.w = nn.Parameter(torch.randn((K, N), device=device, dtype=dtype), requires_grad=True)
        self.sigmoid_layer = nn.Sigmoid()

    # forward propagate input
    def forward(self):
        self.thetas = self.sigmoid_layer(self.sg_slope*self.w)
        return self.thetas

# codeset: (batch_size,K,N), output: (batch_size,K,N)
# (zero-delays are removed if called with .narrow(2,1,N-1))
def autocorr(codeset):
    if len(codeset.shape) == 2:
        codeset = codeset.unsqueeze(0)
    assert (len(codeset.shape) == 3)
    [batch_size, K, N] = codeset.shape
    roll_inds=torch.remainder(torch.arange(N).unsqueeze(0).repeat(N,1)+torch.arange(N).unsqueeze(1), N).unsqueeze(0).repeat(K,1,1).unsqueeze(0)
    return (torch.gather(codeset.unsqueeze(-1).repeat(1,1,1,N),2,roll_inds)*codeset.unsqueeze(-1)).sum(2)/float(N)

# codeset: (batch_size,K,N), output: (batch_size,K*(K-1)/2,N)
def crosscorr(codeset):
    if len(codeset.shape) == 2:
        codeset = codeset.unsqueeze(0)
    assert (len(codeset.shape) == 3)
    [batch_size, K, N] = codeset.shape
    codes_inds = torch.combinations(torch.arange(K),2,with_replacement=False).unsqueeze(0)
    roll_inds=torch.remainder(torch.arange(N).unsqueeze(0).repeat(N,1)+torch.arange(N).unsqueeze(1), N).unsqueeze(0).repeat(int(K*(K-1)/2),1,1).unsqueeze(0)
    return (torch.gather(torch.gather(codeset,1,codes_inds.narrow(2,1,1).repeat(1,1,N)).unsqueeze(-1).repeat(1,1,1,N),2,roll_inds)*torch.gather(codeset,1,codes_inds.narrow(2,0,1).repeat(1,1,N)).unsqueeze(-1)).sum(2)/float(N)

def plot_change_of_objective(f_zks, obj_no, K, N, to_save=False, save_folder=None):
    x1 = np.arange(1,len(f_zks)+1)
    y1 = np.asarray(f_zks)
    
    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle('Change of the Objective Value w.r.t. Epoch No (Objective {}, (K,N)=({},{}))'.format(obj_no,K,N))
    
    ax1.plot(x1, y1)
    ax1.set_ylabel('Objective Value')
    ax1.set_xlabel('Epoch Number')
    plt.show()
    if to_save:
        plt.savefig(os.path.join(save_folder, 'obj_vs_ep.png'),bbox_inches='tight')
    return plt

def plot_change_of_loss(losses, obj_no, K, N, to_save=False, save_folder=None):
    x1 = np.arange(1,len(losses)+1)
    y1 = np.asarray(losses)
    
    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle('Change of the Loss Value w.r.t. Epoch No (Objective {}, (K,N)=({},{}))'.format(obj_no,K,N))
    
    ax1.plot(x1, y1)
    ax1.set_ylabel('Loss Value')
    ax1.set_xlabel('Epoch Number')
    plt.show()
    if to_save:
        plt.savefig(os.path.join(save_folder, 'loss_vs_ep.png'),bbox_inches='tight')
    return plt

def calc_f_zk(zk_pm1, obj_no):
    if obj_no == 1:
        f_zk = (autocorr(zk_pm1).narrow(2,1,N-1).pow(2).mean((1,2))  + crosscorr(zk_pm1).pow(2).mean((1,2)))/2.0
    elif obj_no == 2:
        f_zk = (autocorr(zk_pm1).narrow(2,1,N-1).pow(2).sum((1,2))/float(K*N)  + crosscorr(zk_pm1).pow(2).sum((1,2))/float(K*(K-1)*N/2.0))/2.0
    elif obj_no == 3:
        f_zk = (autocorr(zk_pm1).narrow(2,1,N-1).pow(2).sum((1,2))/float(N)  + crosscorr(zk_pm1).pow(2).sum((1,2))/float(N))/2.0
    elif obj_no == 4:
        f_zk = np.log((autocorr(zk_pm1).narrow(2,1,N-1).pow(2).mean((1,2))  + crosscorr(zk_pm1).pow(2).mean((1,2)))/2.0)
    elif obj_no == 5:
        f_zk = np.log((autocorr(zk_pm1).narrow(2,1,N-1).pow(2).sum((1,2))/float(K*N)  + crosscorr(zk_pm1).pow(2).sum((1,2))/float(K*(K-1)*N/2.0))/2.0)
    elif obj_no == 6:
        f_zk = np.log((autocorr(zk_pm1).narrow(2,1,N-1).pow(2).sum((1,2))/float(N)  + crosscorr(zk_pm1).pow(2).sum((1,2))/float(N))/2.0)
    else:
        assert 1==2
    return f_zk

# Returns a randomly chosen worst code set for worst initialization
def find_rnd_worst_codeset(K,N, obj_no):
    rang = (np.expand_dims(np.arange(int(math.pow(2,K*N))),(-1,1)).repeat(K,1).repeat(N,2))
    divs = (np.expand_dims(np.power(2,np.reshape(np.arange(K*N),(K,N))),0))
    codesets = np.power(-1,rang//divs)

    au_fs_vct = calc_autocorr_np(codesets)[:,:,1:]
    cr_fs_vct = calc_crosscorr_np(codesets)
    
    if (obj_no == 1) or (obj_no == 4):
        corr_sq_sums = ((au_fs_vct**2).sum(-1).sum(-1)*(K-1)*N+(cr_fs_vct**2).sum(-1).sum(-1)*(N-1)*2.0)
        div_factor = K*(K-1)*N*(N-1)*2.0*N*N
    elif (obj_no == 2) or (obj_no == 5):
        corr_sq_sums = ((au_fs_vct**2).sum(-1).sum(-1)*(K-1)+(cr_fs_vct**2).sum(-1).sum(-1)*2.0)
        div_factor = K*(K-1)*N*2.0*N*N
    elif (obj_no == 3) or (obj_no == 6):
        corr_sq_sums = ((au_fs_vct**2).sum(-1).sum(-1)+(cr_fs_vct**2).sum(-1).sum(-1))
        div_factor = N*2.0*N*N
    else:
        assert 1==2
    inds_of_worst_codesets = np.where(corr_sq_sums == corr_sq_sums.max())[0]
    return codesets[inds_of_worst_codesets[np.random.randint(0,len(inds_of_worst_codesets))]]
# codesets: (1,K,N), output: (K,N)
def calc_autocorr_np(codesets, normalize=False):
    if len(codesets.shape) == 2:
        codesets = np.expand_dims(codesets,0)
    assert (len(codesets.shape) == 3)
    ba_sz,K,N = codesets.shape
    roll_inds=np.expand_dims(np.expand_dims((np.expand_dims(np.arange(N),0).repeat(N,0)+np.expand_dims(np.arange(N),1)) % N,0).repeat(K,0),0)
    result = (np.take_along_axis(np.expand_dims(codesets,-1).repeat(N,-1),roll_inds,2)*np.expand_dims(codesets,-1)).sum(2)
    if normalize:
        result = result/float(N)
    return result

# codeset: (1,K,N), output: (K*(K-1)/2,N) 
def calc_crosscorr_np(codesets, normalize=False):
    if len(codesets.shape) == 2:
        codesets = np.expand_dims(codesets,0)
    assert (len(codesets.shape) == 3)
    ba_sz,K,N = codesets.shape
    codes_inds = np.expand_dims(np.asarray(list(itertools.combinations(range(K), r=2))),0)
    roll_inds=np.expand_dims(np.expand_dims((np.expand_dims(np.arange(N),0).repeat(N,0)+np.expand_dims(np.arange(N),1)) % N,0).repeat(int(K*(K-1)/2),0),0)
    result = (np.take_along_axis(np.expand_dims(np.take_along_axis(codesets,codes_inds[:,:,1:2].repeat(N,2),1),-1).repeat(N,-1),roll_inds,2)*np.expand_dims(np.take_along_axis(codesets,codes_inds[:,:,0:1].repeat(N,2),1),-1)).sum(2)
    if normalize:
        result = result/float(N)
    return result    
# Main Code to Execute
K = int(input('Write the number of codes (satellites): '))
N = int(input('Write the period of codes: '))
obj_no = int(input('Write the objective function no: '))

# Hyperparameters
sg_slp = 1.0
init_mode = int(input('Which init mode to use? ')) # 0: worst initialization, 1: random initalization
dtype = torch.float
device = torch.device("cpu")
model = Sigmoid_Model(K, N, obj_no, sg_slp, init_mode, device, dtype)
init_thetas = model()
            
learning_rate = 0.35 #0.3 idi, hizlandirmak icin 0.35 yaptim
epochs = 500000
samps = 100 # no of samplings at each epoch for monte carlo gradient estimation
print_freq = 25000 # print thetas once in "print_frequency" epochs
write_freq = 25000 # print thetas once in "print_frequency" epochs

params_str = 'init{}_sgs{:.1f}_lr{:.2f}_eps{}_smps{}'.format(init_mode,sg_slp, learning_rate,epochs,samps)
save_folder = os.path.join('data', 'obj'+str(obj_no), 'sigm_bern_nes_model', str(K)+'_'+str(N),params_str)
if not os.path.exists(save_folder):
    try:
        os.makedirs(save_folder)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

np.set_printoptions(precision=4)

optimizer = optim.SGD(model.parameters(), lr=learning_rate)
f_zks = []
losses = []
theta_ts = np.zeros((epochs,K,N))
outs_log = "Initial thetas:\n" + str(np.asarray(init_thetas.tolist())) + '\n'
init_f_zk = calc_f_zk(1-2*(init_thetas>0.5), obj_no).item() #initial f_zk with hard thresholding
outs_log += 'Initial f_zk with thresholded thetas: {:.4f} (log:{:.4f}) \n'.format(init_f_zk, np.log(init_f_zk))
print(outs_log)
# print("Initial thetas:\n" + str(np.asarray(init_thetas.tolist())))
# print('Initial f_zk with thresholded thetas: ', calc_f_zk(1-2*(init_thetas>0.5)).item(), '\n')
print('\nT r a i n i n g     t h e     n e t w o r k...\n')
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
    f_zk = calc_f_zk(zk_pm1, obj_no)
    # Compute expression zk*theta+(1-zk)*(1-theta):
    pi_zk = zk*(thetas.unsqueeze(0))+(1-zk)*(1-(thetas.unsqueeze(0)))
    # Calculate loss by averaging over samples:
    loss = (f_zk * (torch.log(pi_zk).sum((1,2)))).mean()
    
    # Backpropagate:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Save mean f_zk, the loss, and the theta
    f_zks.append(f_zk.mean().item())
    losses.append(loss.item())
    theta_ts[t] = np.asarray(thetas.tolist())
    
    # Print
    if t % print_freq == 0:
        print('Epoch: ', t, ', Thetas: ')
        print(theta_ts[t])
        print('Mean f_zk: ', f_zk.mean().item())
    if t % write_freq == 0:
        outs_log += 'Epoch: '+ str(t)+ ', Thetas:\n' + str(theta_ts[t]) + '\nMean f_zk: ' + str(f_zk.mean().item())
    
# Compute final code set by hardthresholding thetas (they should be almost binary by now)
final_zk_pm1 = (1-2*(thetas>0.5))
final_f_zk = calc_f_zk(final_zk_pm1, obj_no).item()
print('**************')
print('Final Thetas: ', thetas)
print('FINAL f_zk: {:.5f} (log: {:.5f})'.format(final_f_zk,np.log(final_f_zk)))

outs_log += 'Final Thetas: ' + str(thetas) + '\nFINAL f_zk: {:.5f} (log: {:.5f})'.format(final_f_zk,np.log(final_f_zk))

if (input('Wanna save the log file (yes=1)? ') == '1'):
    f_logs = open(os.path.join(save_folder, 'logs.txt'), "w")
    f_logs.write(outs_log)
    f_logs.close()
    
if (input('Wanna save theta_ts (yes=1)? ') == '1'):
    np.save(os.path.join(save_folder,'theta_ts'),theta_ts)
    
to_save_plts = (input('Wanna save the plots (yes=1)? ') == '1')
os.environ['KMP_DUPLICATE_LIB_OK']='True'
plt.close('all')
plot_change_of_objective(f_zks, obj_no, K, N, to_save_plts, save_folder)
plot_change_of_loss(losses, obj_no, K, N, to_save_plts, save_folder)
    