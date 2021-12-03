import functions.verify_correlations as corr_fns
import torch

def MeMeAC(zk_pm1, cons_dict):
    codes_inds_1 = cons_dict['codes_inds_1']
    codes_inds_2 = cons_dict['codes_inds_2']
    if len(zk_pm1.shape) == 2: zk_pm1 = zk_pm1.unsqueeze(0)
    [batch_size, K, N] = zk_pm1.shape
    aumean = corr_fns.autocorr_even_ft(zk_pm1).narrow(2,1,N-1).pow(2).mean((1,2))
    crmean = corr_fns.crosscorr_even_ft(zk_pm1, codes_inds_1, codes_inds_2).pow(2).mean((1,2))
    f_zk = (aumean + crmean)/2.0
    return f_zk

def MaMeAC(zk_pm1, cons_dict):
    codes_inds_1 = cons_dict['codes_inds_1']
    codes_inds_2 = cons_dict['codes_inds_2']
    if len(zk_pm1.shape) == 2: zk_pm1 = zk_pm1.unsqueeze(0)
    [batch_size, K, N] = zk_pm1.shape
    aumean = corr_fns.autocorr_even_ft(zk_pm1).narrow(2,1,N-1).pow(2).mean((1,2))
    crmean = corr_fns.crosscorr_even_ft(zk_pm1, codes_inds_1, codes_inds_2).pow(2).mean((1,2))
    f_zk = torch.max(aumean, crmean)
    return f_zk

def l_MeMeAC(zk_pm1, cons_dict):
    codes_inds_1 = cons_dict['codes_inds_1']
    codes_inds_2 = cons_dict['codes_inds_2']
    [batch_size, K, N] = zk_pm1.shape
    aumean = corr_fns.autocorr_even_ft(zk_pm1).narrow(2,1,N-1).pow(2).mean((1,2))
    crmean = corr_fns.crosscorr_even_ft(zk_pm1, codes_inds_1, codes_inds_2).pow(2).mean((1,2))
    f_zk = torch.log((aumean + crmean)/2.0)
    return f_zk

def MEWSD(zk_pm1, cons_dict):
    raise ValueError('MEWSD function is not yet implemented.')
    
def MEWSD_d(zk_pm1, cons_dict):
    raise ValueError('MEWSD_d function is not yet implemented.')
    
def MF(zk_pm1, cons_dict):
    raise ValueError('MF function is not yet implemented.')
    
def MF_d(zk_pm1, cons_dict):
    raise ValueError('MF_d function is not yet implemented.')
    
def ELW(zk_pm1, cons_dict):
    raise ValueError('ELW function is not yet implemented.')

# calculates objective from binary ({-1,1}) codeset (zk_pm1)
def calc_obj(zk_pm1, obj_name, cons_dict = {}):
    # List: MeMeAC, MaMeAC, MEWSD, MEWSD_d, MF, MF_d, ELW
    return globals()[obj_name](zk_pm1, cons_dict)
