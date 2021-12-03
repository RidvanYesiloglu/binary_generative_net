import numpy as np
import torch

def main(smaller_opt, mode):
    K,N = smaller_opt.shape
    fft_smaller_opt = np.fft.fft(smaller_opt) # dimension (K,N)    
    print('FT Small: ' + str(fft_smaller_opt))
    # (np.abs(ft_smaller_opt)**2).sum() is N**2
    mult_fac = np.sqrt(2)
    const_odd = np.sqrt(2*N)
    fft_larger = np.ones((K,2*N), dtype=np.complex128)*const_odd
    for i in range(N):
        fft_larger[:,2*i] = fft_smaller_opt[:,i]*mult_fac
    # (np.abs(fft_ex)**2).sum() should be (2*N)**2
    print('FT Large: ' + str(fft_larger))
    inv_fft_larger = np.fft.ifft(fft_larger).real # dimension (1,2*N)
    print('Initial inv: ' + str(inv_fft_larger))
    if mode=='sp1':
        inv_fft_larger[inv_fft_larger<-1] = -1
        inv_fft_larger[inv_fft_larger>1] = 1
        inv_fft_larger *= 0.6
        inv_fft_larger = inv_fft_larger/2.0 + 0.5
        print('Initial sigmoid: ' + str(inv_fft_larger))
        init_param = np.log(inv_fft_larger/(1-inv_fft_larger))
    elif mode == 'sp2':
        inv_fft_larger[inv_fft_larger<=0] = 0
        inv_fft_larger[inv_fft_larger>0] = 1
        y = 2 * inv_fft_larger - 1
        x = torch.randn((K, 2*N)).cpu().numpy()
        init_param = (y==1)*((x<0)*(-x)+(x>=0)*x) + (y==-1)*((x<0)*x+(x>=0)*(-x))
    return init_param