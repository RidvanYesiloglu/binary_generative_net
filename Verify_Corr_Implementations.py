# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 14:08:21 2021

@author: ridva
"""
import torch
import numpy as np

def autocorr_even_fast(codesets, roll_inds):
    if len(codesets.shape) == 2:
        codesets = codesets.unsqueeze(0)
    assert (len(codesets.shape) == 3)
    [batch_size, K, N] = codesets.shape
    return (torch.gather(codesets.unsqueeze(-1).repeat(1,1,1,N),2,roll_inds.repeat(batch_size,1,1,1))*codesets.unsqueeze(-2)).sum(3)/float(N)

# input: codesets (batch_size, K, N)
# output: ac (batch_size, K, N)
def autocorr_even_simple(codesets):
    [batch_size, K, N] = codesets.shape
    print('Codesets shape ', codesets.shape)
    autocorr_mat = np.zeros((batch_size, K, N))
    for codeset_no, codeset in enumerate(codesets):
        for sat_no, sat_i_code in enumerate(codeset):
            for delay in range(N):
                curr_autocorr = 0
                for ind in range(N):
                    curr_autocorr += sat_i_code[ind] * sat_i_code[(ind-delay) % N]
                autocorr_mat[codeset_no, sat_no, delay] = curr_autocorr / float(N)
    return autocorr_mat
            
def autocorr_odd_fast(codesets, roll_inds):
    if len(codesets.shape) == 2:
        codesets = codesets.unsqueeze(0)
    assert (len(codesets.shape) == 3)
    [batch_size, K, N] = codesets.shape
    sigmas = torch.triu(torch.ones((N,N)))*2-1 # (1,1,n, delays)
    return (torch.gather(codesets.unsqueeze(-1).repeat(1,1,1,N),2,roll_inds.repeat(batch_size,1,1,1))*codesets.unsqueeze(-2) * sigmas).sum(3)/float(N)

def autocorr_odd_simple(codesets):
    [batch_size, K, N] = codesets.shape
    print('Codesets shape ', codesets.shape)
    autocorr_mat = np.zeros((batch_size, K, N))
    for codeset_no, codeset in enumerate(codesets):
        for sat_no, sat_i_code in enumerate(codeset):
            for delay in range(N):
                curr_autocorr = 0
                for ind in range(N):
                    if ind < delay:
                        sig = -1.0
                    else:
                        sig = 1.0
                    curr_autocorr += sat_i_code[ind] * sat_i_code[(ind-delay) % N] * sig
                autocorr_mat[codeset_no, sat_no, delay] = curr_autocorr / float(N)
    return autocorr_mat

def crosscorr_even_fast(codesets,codes_inds,roll_inds):
    if len(codesets.shape) == 2:
        codesets = codesets.unsqueeze(0)
    assert (len(codesets.shape) == 3)
    [batch_size, K, N] = codesets.shape
    return (torch.gather(torch.gather(codesets,1,codes_inds.narrow(2,1,1).repeat(batch_size,1,N)).unsqueeze(-1).repeat(1,1,1,N), 2, roll_inds.repeat(batch_size,1,1,1)) * torch.gather(codesets,1,codes_inds.narrow(2,0,1).repeat(batch_size,1,N)).unsqueeze(-2)).sum(3)/float(N)
     
# inp: (b,K,N)
# out: (b, K*(K-1)/2, N)
def crosscorr_even_simple(codesets):
    [batch_size, K, N] = codesets.shape
    crosscorr_mat = np.zeros((batch_size, K*(K-1)//2, N))
    for codeset_no, codeset in enumerate(codesets):
        pair_iter = 0
        for sat1_no in range(K):
            for sat2_no in range(sat1_no+1, K):
                for delay in range(N):
                    curr_crosscorr = 0
                    for ind in range(N):
                        curr_crosscorr += codeset[sat1_no, ind] * codeset[sat2_no, (ind-delay) % N]
                    crosscorr_mat[codeset_no, pair_iter, delay] = curr_crosscorr / float(N)
                pair_iter += 1
    return crosscorr_mat

def crosscorr_odd_fast(codesets, codes_inds, roll_inds):
    if len(codesets.shape) == 2:
        codesets = codesets.unsqueeze(0)
    assert (len(codesets.shape) == 3)
    [batch_size, K, N] = codesets.shape
    sigmas = torch.triu(torch.ones((N,N)))*2-1 # (1,1,n, delays)
    return (torch.gather(torch.gather(codesets,1,codes_inds.narrow(2,1,1).repeat(batch_size,1,N)).unsqueeze(-1).repeat(1,1,1,N), 2, roll_inds.repeat(batch_size,1,1,1)) * torch.gather(codesets,1,codes_inds.narrow(2,0,1).repeat(batch_size,1,N)).unsqueeze(-2) * sigmas.unsqueeze(0)).sum(3)/float(N)

def crosscorr_odd_simple(codesets):
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


K=20
N=30
batch_size = 50
codesets = (np.random.random((batch_size, K, N)) > 0.5)*2.0 -1.0
#print('Codesets: ')
#print(codesets)
# Roll_inds for autocorrelation:
roll_inds_for_au=torch.remainder(torch.arange(N).unsqueeze(0).repeat(N,1)-torch.arange(N).unsqueeze(1), N).unsqueeze(0).repeat(K,1,1).unsqueeze(0)
# Roll_inds and code inds for crosscorrelation:
codes_inds_for_cr = torch.combinations(torch.arange(K),2,with_replacement=False).unsqueeze(0)
roll_inds_for_cr=torch.remainder(torch.arange(N).unsqueeze(0).repeat(N,1)-torch.arange(N).unsqueeze(1), N).unsqueeze(0).repeat(int(K*(K-1)/2),1,1).unsqueeze(0)

# COMPARISONS
# Comparison 1: even autocorrelation
fast_even_auto = autocorr_even_fast(torch.from_numpy(codesets), roll_inds_for_au).numpy()
simple_even_auto = autocorr_even_simple(codesets)
print('Even Auto-correlation Fast/Simple Squared Difference Sum: {:.9f}'.format(((fast_even_auto-simple_even_auto)**2).sum()))
# Comparison 21: odd autocorrelation
fast_odd_auto = autocorr_odd_fast(torch.from_numpy(codesets), roll_inds_for_au).numpy()
simple_odd_auto = autocorr_odd_simple(codesets)
print('Odd Auto-correlation Fast/Simple Squared Difference Sum: {:.9f}'.format(((fast_odd_auto-simple_odd_auto)**2).sum()))
# Comparison 3: even crosscorrelation
fast_even_cross = crosscorr_even_fast(torch.from_numpy(codesets), codes_inds_for_cr, roll_inds_for_cr).numpy()
simple_even_cross = crosscorr_even_simple(codesets)
print('Even Cross-correlation Fast/Simple Squared Difference Sum: {:.9f}'.format(((fast_even_cross-simple_even_cross)**2).sum()))
# Comparison 4: odd crosscorrelation
fast_odd_cross = crosscorr_odd_fast(torch.from_numpy(codesets), codes_inds_for_cr, roll_inds_for_cr).numpy()
simple_odd_cross = crosscorr_odd_simple(codesets)
print('Odd Cross-correlation Fast/Simple Squared Difference Sum: {:.9f}'.format(((fast_odd_cross-simple_odd_cross)**2).sum()))


