import torch
import numpy as np
import time
def autocorr_even_roll(codesets, roll_inds, no_parts,arg4=None,arg5=None,arg6=None):
    if len(codesets.shape) == 2:
        codesets = codesets.unsqueeze(0)
    assert (len(codesets.shape) == 3)
    [batch_size, K, N] = codesets.shape
    codesets = (codesets==1)
    auto = torch.logical_xor(codesets.narrow(0,0,batch_size//no_parts).unsqueeze(-2).repeat(1,1,N,2)[:,:,roll_inds].view(batch_size//no_parts,K,N,N),codesets.narrow(0,0,batch_size//no_parts).unsqueeze(-2))
    auto = (N-2*auto.sum(3))/float(N)
    for i in range(1,no_parts):
        #next_auto = (torch.gather(codesets.narrow(0,i*batch_size//no_parts,batch_size//no_parts).unsqueeze(-2).repeat(1,1,N,2),2,roll_inds.repeat(batch_size//no_parts,1,1,1))*codesets.narrow(0,i*batch_size//no_parts,batch_size//no_parts).unsqueeze(-2)).sum(3)/float(N)
        next_auto = torch.logical_xor(codesets.narrow(0,i*batch_size//no_parts,batch_size//no_parts).unsqueeze(-2).repeat(1,1,N,2)[:,:,roll_inds].view(batch_size//no_parts,K,N,N),codesets.narrow(0,i*batch_size//no_parts,batch_size//no_parts).unsqueeze(-2))
        next_auto = (N-2*next_auto.sum(3))/float(N)
        auto = torch.cat((auto, next_auto),0) 
    rem_length = batch_size-(no_parts)*batch_size//no_parts
    if rem_length > 0:
        print(rem_length)
        next_auto = torch.logical_xor(codesets.narrow(0,(no_parts-1)*batch_size//no_parts,rem_length).unsqueeze(-2).repeat(1,1,N,2)[:,:,roll_inds].view(rem_length,K,N,N),codesets.narrow(0,(no_parts-1)*batch_size//no_parts,batch_size).unsqueeze(-2))
        next_auto = (N-2*next_auto.sum(3))/float(N)
        auto = torch.cat((auto, next_auto),0) 
    return auto
def autocorr_even_ft(codesets,arg2=None,arg3=None,arg4=None,arg5=None,arg6=None):
    if len(codesets.shape) == 2:
        codesets = codesets.unsqueeze(0)
    assert (len(codesets.shape) == 3)
    [batch_size, K, N] = codesets.shape
    fft_codesets = torch.fft.fft(codesets)
    return torch.fft.ifft(torch.conj(fft_codesets) * fft_codesets).real/float(N)
def autocorr_even_simple(codesets,arg2=None,arg3=None,arg4=None,arg5=None,arg6=None):
    codesets = np.asarray(codesets)
    [batch_size, K, N] = codesets.shape
    autocorr_mat = np.zeros((batch_size, K, N))
    for delay in range(N):
        curr_autocorr = np.zeros((batch_size, K))
        for ind in range(N):
            curr_autocorr += codesets[:,:,ind] * codesets[:,:,(ind-delay) % N]
        autocorr_mat[:,:,delay] = curr_autocorr / float(N)
    return autocorr_mat

def autocorr_odd_roll(codesets, roll_inds, sigmas, no_parts,arg5=None,arg6=None):
    if len(codesets.shape) == 2:
        codesets = codesets.unsqueeze(0)
    assert (len(codesets.shape) == 3)
    [batch_size, K, N] = codesets.shape
    codesets = (codesets==1)
    auto = torch.logical_xor(torch.logical_xor(codesets.narrow(0,0,batch_size//no_parts).unsqueeze(-2).repeat(1,1,N,2)[:,:,roll_inds].view(batch_size//no_parts,K,N,N),codesets.narrow(0,0,batch_size//no_parts).unsqueeze(-2)), sigmas)
    auto = (-N+2*auto.sum(3))/float(N)
    for i in range(1,no_parts):
        #next_auto = (torch.gather(codesets.narrow(0,i*batch_size//no_parts,batch_size//no_parts).unsqueeze(-2).repeat(1,1,N,2),2,roll_inds.repeat(batch_size//no_parts,1,1,1))*codesets.narrow(0,i*batch_size//no_parts,batch_size//no_parts).unsqueeze(-2)).sum(3)/float(N)
        next_auto = torch.logical_xor(torch.logical_xor(codesets.narrow(0,i*batch_size//no_parts,batch_size//no_parts).unsqueeze(-2).repeat(1,1,N,2)[:,:,roll_inds].view(batch_size//no_parts,K,N,N),codesets.narrow(0,i*batch_size//no_parts,batch_size//no_parts).unsqueeze(-2)),sigmas)
        next_auto = (-N+2*next_auto.sum(3))/float(N)
        auto = torch.cat((auto, next_auto),0) 
    rem_length = batch_size-(no_parts)*batch_size//no_parts
    if rem_length > 0:
        print(rem_length)
        next_auto = torch.logical_xor(torch.logical_xor(codesets.narrow(0,(no_parts-1)*batch_size//no_parts,rem_length).unsqueeze(-2).repeat(1,1,N,2)[:,:,roll_inds].view(rem_length,K,N,N),codesets.narrow(0,(no_parts-1)*batch_size//no_parts,batch_size).unsqueeze(-2)), sigmas)
        next_auto = (-N+2*next_auto.sum(3))/float(N)
        auto = torch.cat((auto, next_auto),0) 
    return auto

def autocorr_odd_simple(codesets,arg2=None,arg3=None,arg4=None,arg5=None,arg6=None):
    [batch_size, K, N] = codesets.shape
    autocorr_mat = np.zeros((batch_size, K, N))
    for delay in range(N):
        curr_autocorr = np.zeros((batch_size, K))
        for ind in range(N):
            if ind < delay:
                sig = -1.0
            else:
                sig = 1.0
            curr_autocorr += codesets[:,:,ind] * codesets[:,:,(ind-delay) % N] * sig
        autocorr_mat[:,:,delay] = curr_autocorr / float(N)
    return autocorr_mat

def crosscorr_even_roll(codesets, codes_inds_1, codes_inds_2, roll_inds, no_parts,arg6=None):
    if len(codesets.shape) == 2:
        codesets = codesets.unsqueeze(0)
    assert (len(codesets.shape) == 3)
    [batch_size, K, N] = codesets.shape
    codesets = (codesets==1)
    curr_ba = codesets.narrow(0,0,batch_size//no_parts)
    print('Device:',device)
    t = torch.cuda.get_device_properties(0).total_memory
    print('Total memory    :', t)
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print('Reserved memory :', r)
    print('Available memory:', a)
    print('Free memory     :', f)
    crosscorr = torch.logical_xor(torch.gather(curr_ba, 1, codes_inds_1).unsqueeze(-2).repeat(1,1,N,2)[:,:,roll_inds].view(batch_size//no_parts,K*(K-1)//2,N,N), torch.gather(curr_ba,1,codes_inds_2).unsqueeze(-2))
    print('Total memory    :', t)
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print('Reserved memory :', r)
    print('Available memory:', a)
    print('Free memory     :', f)
    crosscorr = (N-2*crosscorr.sum(3,dtype=torch.int16))/float(N)
    for i in range(1,no_parts):
        #next_auto = (torch.gather(codesets.narrow(0,i*batch_size//no_parts,batch_size//no_parts).unsqueeze(-2).repeat(1,1,N,2),2,roll_inds.repeat(batch_size//no_parts,1,1,1))*codesets.narrow(0,i*batch_size//no_parts,batch_size//no_parts).unsqueeze(-2)).sum(3)/float(N)
        curr_ba = codesets.narrow(0,i*batch_size//no_parts,batch_size//no_parts)
        next_crosscorr = torch.logical_xor(torch.gather(curr_ba, 1, codes_inds_1).unsqueeze(-2).repeat(1,1,N,2)[:,:,roll_inds].view(batch_size//no_parts,K*(K-1)//2,N,N), torch.gather(curr_ba,1,codes_inds_2).unsqueeze(-2))
        next_crosscorr = (N-2*next_crosscorr.sum(3, dtype=torch.int16))/float(N)
        crosscorr = torch.cat((crosscorr, next_crosscorr),0) 
    rem_length = batch_size-(no_parts)*batch_size//no_parts
    if rem_length > 0:
        print(rem_length)
        curr_ba = codesets.narrow(0,(no_parts-1)*batch_size//no_parts,rem_length)
        next_crosscorr = torch.logical_xor(torch.gather(curr_ba, 1, codes_inds_1).unsqueeze(-2).repeat(1,1,N,2)[:,:,roll_inds].view(batch_size//no_parts,K*(K-1)//2,N,N), torch.gather(curr_ba,1,codes_inds_2).unsqueeze(-2))
        next_crosscorr = (N-2*next_crosscorr.sum(3, dtype=torch.int16))/float(N)
        crosscorr = torch.cat((crosscorr, next_crosscorr),0) 
    return crosscorr

def crosscorr_even_ft(codesets,codes_inds_1,codes_inds_2,arg4=None,arg5=None,arg6=None):
    if len(codesets.shape) == 2:
        codesets = codesets.unsqueeze(0)
    assert (len(codesets.shape) == 3)
    [batch_size, K, N] = codesets.shape
    first_gather = torch.gather(codesets, 1, codes_inds_1)
    second_gather= torch.gather(codesets,1,codes_inds_2)
    crosscorr = torch.fft.ifft(torch.conj(torch.fft.fft(first_gather)) * torch.fft.fft(second_gather)).real/float(N)
    return crosscorr
def crosscorr_even_simple(codesets,arg2=None,arg3=None,arg4=None,arg5=None,arg6=None):
    [batch_size, K, N] = codesets.shape
    codesets = np.asarray(codesets)
    crosscorr_mat = np.zeros((batch_size, K*(K-1)//2, N))
    pair_iter = 0
    for sat1_no in range(K):
        for sat2_no in range(sat1_no+1, K):
            #print(pair_iter)
            for delay in range(N):
                curr_crosscorr = 0
                for ind in range(N):
                    curr_crosscorr += codesets[:,sat1_no, ind] * codesets[:,sat2_no, (ind-delay) % N]
                crosscorr_mat[:, pair_iter, delay] = curr_crosscorr / float(N)
            pair_iter += 1
    return crosscorr_mat

def crosscorr_odd_roll(codesets, codes_inds_1, codes_inds_2, roll_inds, sigmas, no_parts):
    if len(codesets.shape) == 2:
        codesets = codesets.unsqueeze(0)
    assert (len(codesets.shape) == 3)
    [batch_size, K, N] = codesets.shape
    codesets = (codesets==1)
    curr_ba = codesets.narrow(0,0,batch_size//no_parts)
    crosscorr = torch.logical_xor(torch.logical_xor(torch.gather(curr_ba, 1, codes_inds_1).unsqueeze(-2).repeat(1,1,N,2)[:,:,roll_inds].view(batch_size//no_parts,K*(K-1)//2,N,N), torch.gather(curr_ba,1,codes_inds_2).unsqueeze(-2)), sigmas)
    crosscorr = (-N+2*crosscorr.sum(3,dtype=torch.int16))/float(N)
    for i in range(1,no_parts):
        #next_auto = (torch.gather(codesets.narrow(0,i*batch_size//no_parts,batch_size//no_parts).unsqueeze(-2).repeat(1,1,N,2),2,roll_inds.repeat(batch_size//no_parts,1,1,1))*codesets.narrow(0,i*batch_size//no_parts,batch_size//no_parts).unsqueeze(-2)).sum(3)/float(N)
        curr_ba = codesets.narrow(0,i*batch_size//no_parts,batch_size//no_parts)
        next_crosscorr = torch.logical_xor(torch.logical_xor(torch.gather(curr_ba, 1, codes_inds_1).unsqueeze(-2).repeat(1,1,N,2)[:,:,roll_inds].view(batch_size//no_parts,K*(K-1)//2,N,N), torch.gather(curr_ba,1,codes_inds_2).unsqueeze(-2)), sigmas)
        next_crosscorr = (-N+2*next_crosscorr.sum(3, dtype=torch.int16))/float(N)
        crosscorr = torch.cat((crosscorr, next_crosscorr),0) 
    rem_length = batch_size-(no_parts)*batch_size//no_parts
    if rem_length > 0:
        print(rem_length)
        curr_ba = codesets.narrow(0,(no_parts-1)*batch_size//no_parts,rem_length)
        next_crosscorr = torch.logical_xor(torch.logical_xor(torch.gather(curr_ba, 1, codes_inds_1).unsqueeze(-2).repeat(1,1,N,2)[:,:,roll_inds].view(batch_size//no_parts,K*(K-1)//2,N,N), torch.gather(curr_ba,1,codes_inds_2).unsqueeze(-2)), sigmas)
        next_crosscorr = (-N+2*next_crosscorr.sum(3, dtype=torch.int16))/float(N)
        crosscorr = torch.cat((crosscorr, next_crosscorr),0) 
    return crosscorr

def crosscorr_odd_simple(codesets,arg2=None,arg3=None,arg4=None,arg5=None,arg6=None):
    [batch_size, K, N] = codesets.shape
    crosscorr_mat = np.zeros((batch_size, K*(K-1)//2, N))
    pair_iter = 0
    for sat1_no in range(K):
        for sat2_no in range(sat1_no+1, K):
            #print(pair_iter)
            for delay in range(N):
                curr_crosscorr = 0
                for ind in range(N):
                    if ind < delay:
                        sig = -1.0  
                    else:
                        sig = 1.0
                    curr_crosscorr += codesets[:,sat1_no, ind] * codesets[:,sat2_no, (ind-delay) % N] * sig
                crosscorr_mat[:, pair_iter, delay] = curr_crosscorr / float(N)
            pair_iter += 1
    return crosscorr_mat
    [batch_size, K, N] = codesets.shape
    crosscorr_mat = np.zeros((batch_size, K*(K-1)//2, N))
    for codeset_no, codeset in enumerate(codesets):
        pair_iter = 0
        for sat1_no in range(K):
            for sat2_no in range(sat1_no+1, K):
                for delay in range(N):
                    curr_crosscorr = 0
                    for ind in range(N):
                        if ind < delay:
                            sig = -1.0  
                        else:
                            sig = 1.0
                        curr_crosscorr += codeset[sat1_no, ind] * codeset[sat2_no, (ind-delay) % N] * sig
                    crosscorr_mat[codeset_no, pair_iter, delay] = curr_crosscorr / float(N)
                pair_iter += 1
    return crosscorr_mat

def perform_and_measure_time(func, arg1=None, arg2=None, arg3=None, arg4=None, arg5=None, arg6=None):
    start = time.time()
    res = globals()[func](arg1, arg2, arg3, arg4, arg5, arg6)
    end = time.time()
    return (res, end-start)

def create_codes_inds_1(K, N, batch_size, no_parts, device):
    return torch.combinations(torch.arange(K),2,with_replacement=False).unsqueeze(0).narrow(2,1,1).repeat(batch_size//no_parts,1,N).to(device)

def create_codes_inds_2(K, N, batch_size, no_parts, device):
    return torch.combinations(torch.arange(K),2,with_replacement=False).unsqueeze(0).narrow(2,0,1).repeat(batch_size//no_parts,1,N).to(device)

def create_roll_inds(N, device):
    roll_inds = torch.zeros((N,2*N), dtype=torch.bool).to(device)
    roll_inds[torch.arange(N).view(N,1), torch.remainder(torch.arange(N).unsqueeze(0).repeat(N,1)-torch.arange(N).unsqueeze(1)+N, 2*N)] = True
    return roll_inds

def create_sigmas(N, device):
    return torch.triu(torch.ones((N,N),dtype=torch.bool)).unsqueeze(0).to(device)

def main():
    K=31
    N=1024
    batch_size = 30
    no_parts=1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:',device)
    t = torch.cuda.get_device_properties(0).total_memory
    print('Total memory:    ', t)
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print('Reserved memory: ', r)
    print('Available memory:', a)
    print('Free memory     :', f)
    codesets = (np.random.random((batch_size, K, N)) > 0.5)*2.0 -1.0
    codesets_tr = torch.from_numpy(codesets).to(device)
    
    codes_inds_1 = create_codes_inds_1(K, N, batch_size, no_parts, device)
    codes_inds_2 = create_codes_inds_2(K, N, batch_size, no_parts, device)
    roll_inds = create_roll_inds(N, device)
    sigmas = create_sigmas(N, device)
    
    # COMPARISONS
    # Comparison 1: even autocorrelation (roll-based, and ft-based)
    print('Comparison 1: even autocorrelation (roll-based, ft-based, simple)')
    (even_auto_roll,time_even_auto_roll) = perform_and_measure_time('autocorr_even_roll', codesets_tr, roll_inds, no_parts)
    print('Roll-based even autocorrelation time: {} s'.format(time_even_auto_roll))
    (even_auto_ft,time_even_auto_ft) = perform_and_measure_time('autocorr_even_ft', codesets_tr)
    print('Ft-based even autocorrelation time: {} s'.format(time_even_auto_ft))
    (even_auto_simple,time_even_auto_simple) = perform_and_measure_time('autocorr_even_simple', codesets)
    print('Simple even autocorrelation time: {} s'.format(time_even_auto_simple))
    print('Sum of Sq. Diff. between Roll-based and Simple Even Autocorrelation: {:.9f}'.format(((even_auto_roll.cpu().numpy()-even_auto_simple)**2).sum()))
    print('Sum of Sq. Diff. between Ft-based and Simple Even Autocorrelation: {:.9f}\n'.format(((even_auto_ft.cpu().numpy()-even_auto_simple)**2).sum()))
    
    
    # Comparison 2: odd autocorrelation (roll-based, and ft-based)
    print('Comparison 2: odd autocorrelation (roll-based, simple)')
    (odd_auto_roll,time_odd_auto_roll) = perform_and_measure_time('autocorr_odd_roll', codesets_tr, roll_inds, sigmas, no_parts)
    print('Roll-based odd autocorrelation time: {} s'.format(time_odd_auto_roll))
    (odd_auto_simple,time_odd_auto_simple) = perform_and_measure_time('autocorr_odd_simple', codesets)
    print('Simple odd autocorrelation time: {} s'.format(time_odd_auto_simple))
    print('Sum of Sq. Diff. between Roll-based and Simple Odd Autocorrelation: {:.9f}\n'.format(((odd_auto_roll.cpu().numpy()-odd_auto_simple)**2).sum()))
    
    
    # Comparison 3: even cross-correlation (roll-based, and ft-based)
    print('Comparison 3: even cross-correlation (roll-based, ft-based, simple)')
    (even_cross_roll,time_even_cross_roll) = perform_and_measure_time('crosscorr_even_roll', codesets_tr, codes_inds_1, codes_inds_2, roll_inds, no_parts)
    print('Roll-based even cross-correlation time: {} s'.format(time_even_cross_roll))
    (even_cross_ft,time_even_cross_ft) = perform_and_measure_time('crosscorr_even_ft', codesets_tr, codes_inds_1.repeat(no_parts,1,1), codes_inds_2.repeat(no_parts,1,1))
    print('Ft-based even cross-correlation time: {} s'.format(time_even_cross_ft))
    (even_cross_simple,time_even_cross_simple) = perform_and_measure_time('crosscorr_even_simple', codesets)
    print('Simple even cross-correlation time: {} s'.format(time_even_cross_simple))
    print('Sum of Sq. Diff. between Roll-based and Simple Even Cross-correlation: {:.9f}'.format(((even_cross_roll.cpu().numpy()-even_cross_simple)**2).sum()))
    print('Sum of Sq. Diff. between Ft-based and Simple Even Cross-correlation: {:.9f}\n'.format(((even_cross_ft.cpu().numpy()-even_cross_simple)**2).sum()))
    
    
    # Comparison 4: odd cross-correlation (roll-based, and ft-based)
    print('Comparison 4: odd cross-correlation (roll-based, simple)')
    (odd_cross_roll,time_odd_cross_roll) = perform_and_measure_time('crosscorr_odd_roll', codesets_tr, codes_inds_1, codes_inds_2, roll_inds, sigmas, no_parts)
    print('Roll-based odd cross-correlation time: {} s'.format(time_odd_cross_roll))
    (odd_cross_simple,time_odd_cross_simple) = perform_and_measure_time('crosscorr_odd_simple', codesets)
    print('Simple odd cross-correlation time: {} s'.format(time_odd_cross_simple))
    print('Sum of Sq. Diff. between Roll-based and Simple Odd Cross-correlation: {:.9f}'.format(((odd_cross_roll.cpu().numpy()-odd_cross_simple)**2).sum()))
    
if __name__ == "__main__":
    main()






