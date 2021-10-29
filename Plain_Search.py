import math
import numpy as np
import itertools
import time
import os
import errno
import matplotlib.pyplot as plt

# find the optimal code sets for the objective in corr_sq_sums definition
def plain_search(K,N, obj_no, ret_not_only_min): 
    rang = (np.expand_dims(np.arange(int(math.pow(2,K*N))),(-1,1)).repeat(K,1).repeat(N,2))
    divs = (np.expand_dims(np.power(2,np.reshape(np.arange(K*N),(K,N))),0))
    codesets = np.power(-1,rang//divs)

    au_fs_vct = calc_autocorr(codesets)[:,:,1:]
    cr_fs_vct = calc_crosscorr(codesets)
    
    if obj_no == 1:
        corr_sq_sums = ((au_fs_vct**2).sum(-1).sum(-1)*(K-1)*N+(cr_fs_vct**2).sum(-1).sum(-1)*(N-1)*2.0)
        div_factor = K*(K-1)*N*(N-1)*2.0*N*N
    elif obj_no == 2:
        corr_sq_sums = ((au_fs_vct**2).sum(-1).sum(-1)*(K-1)+(cr_fs_vct**2).sum(-1).sum(-1)*2.0)
        div_factor = K*(K-1)*N*2.0*N*N
    elif obj_no == 3:
        corr_sq_sums = ((au_fs_vct**2).sum(-1).sum(-1)+(cr_fs_vct**2).sum(-1).sum(-1))
        div_factor = N*2.0*N*N
    else:
        assert 1==2
    inds_of_min_codesets = np.where(corr_sq_sums == corr_sq_sums.min())
    if ret_not_only_min:
        return [codesets[inds_of_min_codesets], corr_sq_sums, div_factor]
    else:
        return codesets[inds_of_min_codesets]

# plot the histogram of the objective over all space
def save_hist(corr_sq_sums, div_factor, folder, K, N, obj_no):
    hist, bin_edges = np.histogram(np.log(corr_sq_sums), bins='auto')
    plt.figure(figsize=(14, 4))
    plt.bar(bin_edges[:-1]-np.log(div_factor), hist/len(corr_sq_sums), width=(bin_edges[1]-bin_edges[0]),align='edge')
    plt.xlabel('Objective Value')
    plt.ylabel('Percentage')
    plt.suptitle('Histogram of Log of Objective ' + str(obj_no) + ' for (K,N)=('+str(K)+','+str(N)+')')
    # find smallest ten
    smallests, smallest_counts = find_smallests(corr_sq_sums, 10)
    smallests = np.log(smallests)-np.log(div_factor)
    smallest_counts = smallest_counts*100/len(corr_sq_sums)
    subtitle = 'The Smallest {}: '.format(len(smallests)) 
    for no,smallest in enumerate(smallests):
        subtitle += '{}:{:.3f}({:.3f}%), '.format(no+1,smallest,smallest_counts[no])
        if (no+1) == (len(smallests)//2):
            subtitle += '\n'
    subtitle = subtitle[:-2] + ' (TOTAL: {:.3f}%)'.format(smallest_counts.sum()) 
    #plt.title(subtitle)
    plt.subplots_adjust(top=0.84)
    plt.text(0.5, 0.86, subtitle, fontsize=10, transform=plt.gcf().transFigure,ha="center")
    plt.show()
    plt.savefig(os.path.join(folder, str(K)+'_'+str(N)+'_obj_hist.png'),bbox_inches='tight')
    
def find_smallests(corr_sq_sums, how_many):
    smallests = [corr_sq_sums.min()]
    smallest_counts = [(corr_sq_sums == corr_sq_sums.min()).sum()]
    for i in range(how_many-1):
        if np.any(corr_sq_sums > smallests[-1]):
            smallests.append(corr_sq_sums[corr_sq_sums > smallests[-1]].min())
            smallest_counts.append((corr_sq_sums == smallests[-1]).sum())
    return np.asarray(smallests), np.asarray(smallest_counts)
# calculate objective on given min_codesets
def calc_corr_sq_sums(codesets, obj_no, return_corrs=False):
    au_l = calc_autocorr(codesets, normalize=True)[:,:,1:]
    cr_l = calc_crosscorr(codesets, normalize=True)
    K,N = codesets.shape[-2],codesets.shape[-1]
    if obj_no == 1:
        au_l = (au_l**2).mean(-1).mean(-1)
        cr_l = (cr_l**2).mean(-1).mean(-1)
    elif obj_no == 2:
        au_l = (au_l**2).sum(-1).sum(-1)/float(K*N)
        cr_l = (cr_l**2).sum(-1).sum(-1)/float(K*(K-1)*N/2.0)
    elif obj_no == 3:
        au_l = (au_l**2).sum(-1).sum(-1)/float(N)
        cr_l = (cr_l**2).sum(-1).sum(-1)/float(N)
    else:
        assert 1==2
    corr_sq_sums = (au_l+cr_l)/2.0
    if not return_corrs:
        return corr_sq_sums.min()
    else:
        return (corr_sq_sums.min(), au_l, cr_l)

# codesets: (1,K,N), output: (K,N)
def calc_autocorr(codesets, normalize=False):
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
def calc_crosscorr(codesets, normalize=False):
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

def write_to_file(codesets, folder, obj_no, no_of_digs=2):
    if not os.path.exists(os.path.dirname(folder)):
        try:
            os.makedirs(folder)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    np.set_printoptions(precision=no_of_digs,sign='+') # with two decimals
    (min_objective, au_l, cr_l) = calc_corr_sq_sums(codesets, obj_no, return_corrs=True)
    codesets_log = 'For K = ' + str(K) + ', N = ' + str(N) + ', there are ' + str(len(codesets)) + ' optimal sets.\n'
    codesets_log += 'The min objective is ' + str(min_objective) + ', and its log is '+str(np.log(min_objective))+'.\n'
    
    ac_log = 'For K = ' + str(K) + ', N = ' + str(N) + ', there are ' + str(len(codesets)) + ' optimal sets.\n'
    ac_log += 'The AC square sum is at minimum ' + str(au_l.min())+'(log=' + str(np.log(au_l.min())) + ') and at maximum ' + str(au_l.max()) + '(log=' + str(np.log(au_l.max())) + ').\n'
    
    cc_log = 'For K = ' + str(K) + ', N = ' + str(N) + ', there are ' + str(len(codesets)) + ' optimal sets.\n'
    cc_log += 'The CC square sum is at minimum ' + str(cr_l.min())+'(log=' + str(np.log(cr_l.min())) + ') and at maximum ' + str(cr_l.max()) + '(log=' + str(np.log(cr_l.max())) + ').\n'
    
    ft_log = codesets_log
    for no, min_codeset in enumerate(codesets):
        codesets_log += 'No: ' + str(no+1) + ', Code set: {'
        for k_i in range(K):
            codesets_log += str(min_codeset[k_i]) + ', '
        codesets_log = codesets_log[:-2] + '}\n'
            
        autocorr = calc_autocorr(min_codeset, normalize=True)[0]
        ac_log += 'No: ' + str(no+1) + ', AC: {'
        for k_i in range(K):
            ac_log += str(autocorr[k_i]) + ', '
        ac_log = ac_log[:-2] + '}\n'
        
        crosscorr = calc_crosscorr(min_codeset, normalize=True)[0]
        cc_log += 'No: ' + str(no+1) + ', AC: {'
        for k_i in range(int(K*(K-1)/2.0)):
            cc_log += str(crosscorr[k_i]) + ', '
        cc_log = cc_log[:-2] + '}\n'
        
        code_ft = np.fft.fft(min_codeset,norm="ortho")
        ft_log += 'No: ' + str(no+1) + ', CdFT: {'
        for k_i in range(K):
            ft_log += str(code_ft[k_i]) + ', '
        ft_log = ft_log[:-2] + '}\n'
        
        
    f_codesets = open(os.path.join(folder, str(K)+'_'+str(N)+'_codesets.txt'), "w")
    f_codesets.write(codesets_log)
    f_codesets.close()
    
    f_ac = open(os.path.join(folder, str(K)+'_'+str(N)+'_autocorrelations.txt'), "w")
    f_ac.write(ac_log)
    f_ac.close()
    
    f_cc = open(os.path.join(folder, str(K)+'_'+str(N)+'_crosscorrelations.txt'), "w")
    f_cc.write(cc_log)
    f_cc.close()
    
    f_ft = open(os.path.join(folder, str(K)+'_'+str(N)+'_ftransforms.txt'), "w")
    f_ft.write(ft_log)
    f_ft.close()
    
def print_codesets(codesets, inc_ac=True, inc_cc=True, inc_ft=True, no_of_digs=2):
    np.set_printoptions(precision=no_of_digs,sign='+') # with two decimals
    for no, min_codeset in enumerate(codesets):
        str_to_prn = '***\nCode set ' + str(no+1) + ': {'
        for k_i in range(K):
            str_to_prn += str(min_codeset[k_i]) + ', '
        str_to_prn = str_to_prn[:-2] + '}'
        if inc_ac:
            str_to_prn += ',\nAC: {'
            autocorr = calc_autocorr(min_codeset)[0]
            for k_i in range(K):
                str_to_prn += str(autocorr[k_i]) + ', '
            str_to_prn = str_to_prn[:-2] + '}'
        if inc_cc:
            str_to_prn += '\nCC: {'
            crosscorr = calc_crosscorr(min_codeset)[0]
            for k_i in range(int(K*(K-1)/2)):
                str_to_prn += str(crosscorr[k_i]) + ', '
            str_to_prn = str_to_prn[:-2] + '}'
        if inc_ft:
            str_to_prn += ',\nCdFT: {'    
            code_ft = np.fft.fft(min_codeset,norm="ortho")
            for k_i in range(K):
                str_to_prn += str(code_ft[k_i]) + ', '
            str_to_prn = str_to_prn[:-2] + '}'
        #str_to_prn +='\n'
        print(str_to_prn)

# Check whether a codeset is included in optimal codesets.
def includes_codeset(codeset, min_codesets):
    if len(codeset.shape) == 2:
        codeset = np.expand_dims(codeset, 0)
    return np.any(np.all(codeset==min_codesets,(1,2)))

# Main Code to Execute
plt.close('all')
K = int(input('Write the number of codes (satellites): '))
N = int(input('Write the period of codes: '))
obj_no = int(input('Write the objective function no: '))
print('\nP e r f o r m i n g      p l a i n      s e a r c h...\n')

[min_codesets, corr_sq_sums,div_factor] = plain_search(K, N, obj_no, ret_not_only_min=True)
print('For K = ' + str(K) + ', N = ' + str(N) + ', there are ' + str(len(min_codesets)) + ' optimal sets')
min_obj_val = calc_corr_sq_sums(min_codesets, obj_no)
print('The min objective is ' + str(min_obj_val) + ', and min log objective is ' + str(np.log(min_obj_val)))

save_folder = os.path.join('data', 'obj'+str(obj_no), 'opt_sets_pln_srch')
if (input('Wanna write them to files (1=yes)? ') == '1'):
    write_to_file(min_codesets,folder=save_folder, obj_no = obj_no)

if (input('Wanna save histogram plot (1= yes)?') == '1'):
    save_hist(corr_sq_sums, div_factor, save_folder, K, N, obj_no)

if (input('Wanna save them (1=yes)? ') == '1'):
    np.save(os.path.join(save_folder, str(K)+'_'+str(N)+'_mincodesets.npy'),min_codesets)

if (input('Wanna print them (1=yes)? ') == '1'):
    print_codesets(min_codesets, inc_ac=True, inc_cc=True, inc_ft=True)
    

