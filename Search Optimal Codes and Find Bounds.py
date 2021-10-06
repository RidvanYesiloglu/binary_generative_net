import math
import numpy as np

# calculates welch bound for K: number of codes, N: length of codes, s: any positive integer
def welch_bound_rm(K,N,s):
    return math.pow((K*N*math.factorial(s)*math.factorial(N-1)/math.factorial(N+s-1)-1)/(K*N-1),1/(2*s))

# calculates sidelnikov bound for K: number of codes, N: length of codes, s: any nonnegative integer
def sidelnikov_bound(K,N,s):
    return math.sqrt((1/N**2)*((s+1)*(2*N-s)/2.0 - math.pow(2,s)*math.pow(N,2*s+1)*math.factorial(2*N-s)/(K*math.factorial(s)*math.factorial(2*N))))

# find the optimal code set(s) for the objective in corr_sq_sums definition
def find_optimal_sets_brute(K,N):
    codeset = np.zeros((K,N))
    min_corr_sq_sums = float("inf")
    for corr_no in range(int(math.pow(2,K*N))):
        codeset_str = str(bin(corr_no))[2:]
        if len(codeset_str) < K*N:
            str_to_add = '0'*(K*N-len(codeset_str))
            codeset_str = str_to_add + codeset_str
        ind = 0
        for i1 in range(K):
            for i2 in range(N):
                codeset[i1,i2] = int(1-2*int(codeset_str[ind]))
                ind += 1
        # OBJECTIVE:
        corr_sq_sums = np.square(calc_autocorr(codeset)).sum()+np.square(calc_crosscorr(codeset)).sum()
        #print(codeset)
        #print(corr_sq_sums)
        if corr_sq_sums < min_corr_sq_sums:
            min_corr_sq_sums = corr_sq_sums
            min_codesets = [codeset.copy()]
        elif corr_sq_sums == min_corr_sq_sums:
            min_codesets.append(codeset.copy())
    return min_codesets

# calculates autocorrelation including zero-delay, output shape: (K,N)
def calc_autocorr(codeset):
    K,N = codeset.shape
    autocorr_mtr = np.zeros((K,N))
    for code_no in range(K):
        for delay in range(N):
            autocorr_mtr[code_no, delay] = np.correlate(codeset[code_no,:], np.roll(codeset[code_no,:], delay))/N
    return autocorr_mtr

# calculates crosscorrelation, output shape: (K*(K-1)/2,N)
def calc_crosscorr(codeset):
    K,N = codeset.shape
    crosscorr_mtr = np.zeros((int(K*(K-1)/2),N))
    ind = 0
    for code1_no in range(K):
        for code2_no in range(code1_no):
            for delay in range(N):
                crosscorr_mtr[ind, delay] = np.correlate(codeset[code1_no,:], np.roll(codeset[code2_no,:], delay))/N
    return crosscorr_mtr

# Main Code to Execute?
K,N = 2,5
min_codesets = find_optimal_sets_brute(K,N)
# Print results
np.set_printoptions(precision=2) # with two decimals
print('For K = ' + str(K) + ', N = ' + str(N) + ', there are ' + str(len(min_codesets)) + ' optimal sets: ')
for no, min_codeset in enumerate(min_codesets):
    str_to_prn = 'Code set ' + str(no+1) + ': {'
    for k_i in range(K):
        str_to_prn += str(min_codeset[k_i]) + ', '
    str_to_prn = str_to_prn[:-2] + '}, AC: {'
    autocorr = calc_autocorr(min_codeset)
    for k_i in range(K):
        str_to_prn += str(autocorr[k_i]) + ', '
    str_to_prn = str_to_prn[:-2] + '}, CC: {'
    crosscorr = calc_crosscorr(min_codeset)
    for k_i in range(int(K*(K-1)/2)):
        str_to_prn += str(crosscorr[k_i]) + ', '
    str_to_prn = str_to_prn[:-2] + '}'
    print(str_to_prn)
