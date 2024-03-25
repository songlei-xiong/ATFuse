"""
Author: LihuiChen
E-mail: lihuichen@126.com
Note: The metrics for reduced-rolution is the same with the matlat codes opened by [Vivone20]. 
      Metrics for full-resolution have a little different results from the codes opened by [Vivone20].

Refercence: PansharpeningToolver1.3 and Pansharpening Toolbox for Distribution

Pansharpening metrics: The same implementation of CC, SAM, ERGAS, Q2n as the one in Matlab codes publised by:
    [Vivone15]  G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, 
                "A Critical Comparison Among Pansharpening Algorithms", IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 5, pp. 2565�2586, May 2015.
    [Vivone20]  G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, 
                "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and 
                emerging pansharpening methods",IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.
"""
from scipy.ndimage import sobel
import numpy as np
from scipy import signal, ndimage, misc
import cv2
from numpy.linalg import norm
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio

##########################################################
# Full Reference metrics for Reduced Resolution Assesment
##########################################################

def SAM(ms,ps,degs = True):
    result = np.double(ps)
    target = np.double(ms)
    if result.shape != target.shape:
        raise ValueError('Result and target arrays must have the same shape!')

    bands = target.shape[2]
    rnorm = np.sqrt((result ** 2).sum(axis=2))
    tnorm = np.sqrt((target ** 2).sum(axis=2))
    dotprod = (result * target).sum(axis=2)
    cosines = (dotprod / (rnorm * tnorm))
    sam2d = np.arccos(cosines)
    sam2d[np.invert(np.isfinite(sam2d))] = 0.  # arccos(1.) -> NaN
    if degs:
        sam2d = np.rad2deg(sam2d)
    return sam2d[np.isfinite(sam2d)].mean()


def CC(img1, img2):
    """SCC for 2D (H, W)or 3D (H, W, C) image; uint or float[0, 1]"""
    if not  img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    if img1_.ndim == 2:
        return np.corrcoef(img1_.reshape(1, -1), img2_.rehshape(1, -1))[0, 1]
    elif img1_.ndim == 3:
        ccs = [np.corrcoef(img1_[..., i].reshape(1, -1), img2_[..., i].reshape(1, -1))[0, 1]
               for i in range(img1_.shape[2])]
        return np.mean(ccs)
    else:
        raise ValueError('Wrong input image dimensions.')

def sCC(ms, ps):
    ps_sobel = sobel(ps, mode='constant')
    ms_sobel = sobel(ms, mode='constant')
    return  (np.sum(ps_sobel*ms_sobel)/np.sqrt(np.sum(ps_sobel*ps_sobel))/np.sqrt(np.sum(ms_sobel*ms_sobel)))


def _qindex(img1, img2, block_size=8):
    """Q-index for 2D (one-band) image, shape (H, W); uint or float [0, 1]"""
    assert block_size > 1, 'block_size shold be greater than 1!'
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    window = np.ones((block_size, block_size)) / (block_size**2)
    # window_size = block_size**2
    # filter, valid
    pad_topleft = int(np.floor(block_size/2))
    pad_bottomright = block_size - 1 - pad_topleft
    mu1 = cv2.filter2D(img1_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu2 = cv2.filter2D(img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1_**2, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_sq
    sigma2_sq = cv2.filter2D(img2_**2, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu2_sq
    sigma12 = cv2.filter2D(img1_ * img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_mu2

    # all = 1, include the case of simga == mu == 0
    qindex_map = np.ones(sigma12.shape)
    # sigma == 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq) == 0) * ((mu1_sq + mu2_sq) != 0)
    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
    # sigma !=0 and mu == 0 
    idx = ((sigma1_sq + sigma2_sq) != 0) * ((mu1_sq + mu2_sq) == 0)
    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
    # sigma != 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq)!=0) * ((mu1_sq + mu2_sq) != 0)
    qindex_map[idx] =((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
        (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])
    return np.mean(qindex_map)


def Q_AVE(img1, img2, block_size=8):
    """Q-index for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1]"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return _qindex(img1, img2, block_size)
    elif img1.ndim == 3:
        qindexs = [_qindex(img1[..., i], img2[..., i], block_size) for i in range(img1.shape[2])]
        return np.array(qindexs).mean()
    else:
        raise ValueError('Wrong input image dimensions.')

def ERGAS(img_fake, img_real, scale=4):
    """ERGAS for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 4."""
    if not img_fake.shape == img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    img_fake_ = img_fake.astype(np.float64)
    img_real_ = img_real.astype(np.float64)
    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()
        mse = np.mean((img_fake_ - img_real_)**2)
        return 100 / scale * np.sqrt(mse / (mean_real**2 + np.finfo(np.float64).eps))
    elif img_fake_.ndim == 3:
        means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
        means_real = means_real**2
        means_real[np.where(means_real==0)] = np.finfo(np.float64).eps
        mses = ((img_fake_ - img_real_)**2).reshape(-1, img_fake_.shape[2]).mean(axis=0)
        return 100 / scale * np.sqrt((mses/means_real).mean())
    else:
        raise ValueError('Wrong input image dimensions.')
    
def Q2n(I_GT, I_F, Q_blocks_size=32, Q_shift=32):
    N1,N2,N3 = I_GT.shape
    ori_N3 = N3
    size2 = Q_blocks_size
    stepx = int(np.ceil(float(N1)/Q_shift))
    stepy = int(np.ceil(float(N2)/Q_shift))
    # stepy = N2//Q_shift
    if stepy<=0: stepx, stepy = 1, 1
    est1 = (stepx-1)*Q_shift+Q_blocks_size-N1
    est2 = (stepy-1)*Q_shift+Q_blocks_size-N2
    
    if sum([est1!=0, est2!=0])>0:
        refref = np.zeros((N1+est1, N2+est2, N3))
        fusfus = np.zeros((N1+est1, N2+est2, N3))
        refref[:N1, :N2,:] = I_GT
        refref[:N1, N2:,:] = I_GT[:,N2-1:N2-est2-1:-1,:]
        refref[N1:,:,:] = refref[N1-1:N1-est1-1:-1,:,:]

        fusfus[:N1, :N2,:] = I_F
        fusfus[:N1,N2:,:] = I_F[:,N2-1:N2-est2-1:-1,:]
        fusfus[N1:,:,:] = fusfus[N1-1:N1-est1-1:-1,:,:]
        I_GT, I_F = refref, fusfus
    I_GT, I_F = I_GT.astype(np.uint16), I_F.astype(np.uint16)
    N1,N2,N3 = I_GT.shape
    if (np.ceil(np.log2(np.array(N3)))-np.log2(np.array(N3)))!=0:
        Ndif = np.power(2,np.ceil(np.log2(np.array(N3)))) - N3
        dif = np.zeros((N1,N2,int(Ndif)))
        I_GT = np.concatenate((I_GT, dif), axis=2)
        I_F = np.concatenate((I_F, dif), axis=2)
    N3 = I_GT.shape[2]
    
    valori = np.zeros((stepx, stepy, N3))
    for j in range(stepx):
        for i in range(stepy):
            tmp_gt = I_GT[j*Q_shift:j*Q_shift+Q_blocks_size,i*Q_shift:i*Q_shift+size2,:]
            tmp_f = I_F[j*Q_shift:j*Q_shift+Q_blocks_size,i*Q_shift:i*Q_shift+size2,:]
            o = onions_quality(tmp_gt, tmp_f, Q_blocks_size)
            valori[j,i,:] = o
    valori = valori[:,:,:ori_N3]
    Q2n_index_map = np.sqrt((valori*valori).sum(axis=2))
    Q2n_index = Q2n_index_map.mean()
    return Q2n_index

def onions_quality(dat1, dat2, size1):
    dat1, dat2 = dat1.astype(np.double), dat2.astype(np.double)
    dat2[:,:,1:] = -dat2[:,:,1:]
    N3 = dat1.shape[2]
    size2 = size1
    # Block normalization
    for i in range(N3):
        tmp = dat1[:,:,i]
        s, t = tmp.mean(), tmp.std()
        if t==0: t=np.finfo(np.float64).eps
        dat1[:,:,i] = (tmp-s)/t + 1
        if s==0:
            if i==0:
                dat2[:,:,i] = dat2[:,:,i]-s+1
            else:
                dat2[:,:,i]=-(-dat2[:,:,i]-s+1)
        else:
            if i==0:
                dat2[:,:,i] = (dat2[:,:,i]-s)/t + 1
            else:
                dat2[:,:,i]=-((-dat2[:,:,i]-s)/t+1)
    
    m1 = dat1.reshape(-1, N3).mean(axis=0, keepdims=True)
    mod_q1m =((m1*m1).sum())
    m2 = dat2.reshape(-1, N3).mean(axis=0, keepdims=True)
    mod_q2m = ((m2*m2).sum())
    
    mod_q1= np.sqrt((dat1*dat1).sum(axis=2))
    mod_q2= np.sqrt((dat2*dat2).sum(axis=2))
    
   
    
    mod_q1m = np.sqrt(mod_q1m)
    mod_q2m = np.sqrt(mod_q2m)
    termine2 = (mod_q1m*mod_q2m)
    termine4 = ((mod_q1m**2)+(mod_q2m**2))
    
    int1=(size1*size2)/((size1*size2)-1)*(mod_q1*mod_q1).mean()
    int2=(size1*size2)/((size1*size2)-1)*(mod_q2*mod_q2).mean()

    termine3=int1+int2-(size1*size2)/((size1*size2)-1)*((mod_q1m**2)+(mod_q2m**2))
    
    mean_bias = 2*termine2/termine4
    if termine3==0:
        # q = np.zeros(1, 1, N3)
        # q[:,:,N3] = mean_bias
        q = mean_bias
    else:
        cbm = 2/termine3
        qu = onion_mult2D(dat1, dat2)
        qm = onion_mult(m1, m2)
        qv = (size1*size2)/((size1*size2)-1)*(qu.reshape(-1, N3).mean(axis=0, keepdims=True))
        q = qv-(size1*size2)/((size1*size2)-1)*qm
        q = q*mean_bias*cbm
    return q

def onion_mult2D(onion1, onion2):
    while onion1.ndim<3:
        onion1 = np.expand_dims(onion1, axis=0)
        onion2 = np.expand_dims(onion2, axis=0)
    N3 = onion1.shape[2]
    if N3>1:
        L = N3//2
        a=onion1[:,:,:L]
        b=onion1[:,:,L:]
        b[:,:,1:] = -b[:,:,1:]

        c=onion2[:,:,:L]
        d=onion2[:,:,L:]
        d[:,:,1:] = -d[:,:,1:]
        if N3==2:
            ris = np.concatenate((a*c-d*b, a*d+c*b), axis=2)
        else:
            ris1=onion_mult2D(a,c)
            ris2=onion_mult2D(d,np.concatenate((b[:,:,0:1],-b[:,:,1:]), axis=2))
            ris3=onion_mult2D(np.concatenate((a[:,:,0:1],-a[:,:,1:]), axis=2),d)
            ris4=onion_mult2D(c,b)
            
            aux1=ris1-ris2
            aux2=ris3+ris4
            ris = np.concatenate((aux1, aux2), axis=2)
            
    else:
        ris = onion1*onion2
    return ris

def onion_mult(onion1,onion2):

    N=(onion1.shape[1])

    if N>1:
        L=N//2
        a=onion1[:,:L]
        b=onion1[:, L:]
        b[:,1:] = -b[:, 1:]
        c=onion2[:,:L]
        d=onion2[:, L:]
        d[:,1:] = -d[:,1:]
        if N==2:
            ris=np.concatenate((a*c-d*b,a*d+c*b), axis=1)
        else:
            ris1=onion_mult(a,c)
            ris2=onion_mult(d,np.concatenate((b[:,0:1],-b[:,1:]), axis=1))
            ris3=onion_mult(np.concatenate((a[:,0:1],-a[:,1:]), axis=1),d)
            ris4=onion_mult(c,b)
            aux1=ris1-ris2
            aux2=ris3+ris4
            ris=np.concatenate([aux1,aux2], axis=1)
    else:
        ris = onion1*onion2
    return ris
##########################################################
# 23-taps interpolation
##########################################################
'''
interpolation with 23-taps
'''

from scipy import ndimage
def upsample_mat_interp23(image, ratio=4):
    '''2 pixel shift compare with original matlab version'''
    shift=2
    h,w,c = image.shape
    basecoeff = np.array([[-4.63495665e-03, -3.63442646e-03,  3.84904063e-18,
     5.76678319e-03,  1.08358664e-02,  1.01980790e-02,
    -9.31747402e-18, -1.75033181e-02, -3.17660068e-02,
    -2.84531643e-02,  1.85181518e-17,  4.42450253e-02,
     7.71733386e-02,  6.70554910e-02, -2.85299239e-17,
    -1.01548683e-01, -1.78708388e-01, -1.60004642e-01,
     3.61741232e-17,  2.87940558e-01,  6.25431459e-01,
     8.97067600e-01,  1.00107877e+00,  8.97067600e-01,
     6.25431459e-01,  2.87940558e-01,  3.61741232e-17,
    -1.60004642e-01, -1.78708388e-01, -1.01548683e-01,
    -2.85299239e-17,  6.70554910e-02,  7.71733386e-02,
     4.42450253e-02,  1.85181518e-17, -2.84531643e-02,
    -3.17660068e-02, -1.75033181e-02, -9.31747402e-18,
     1.01980790e-02,  1.08358664e-02,  5.76678319e-03,
     3.84904063e-18, -3.63442646e-03, -4.63495665e-03]])
    coeff = np.dot(basecoeff.T, basecoeff)
    I1LRU = np.zeros((ratio*h, ratio*w, c))
    I1LRU[shift::ratio, shift::ratio, :]=image
    for i in range(c):
        temp = I1LRU[:, :, i]
        temp = ndimage.convolve(temp, coeff, mode='wrap')
        I1LRU[:, :, i]=temp
    return I1LRU


##########################################################
# Using Gaussian filter matched MTF to degrade HRMS images
##########################################################
def MTF_Filter(hrms, scale, sensor, GNyq=None):
    # while hrms.ndim<4:
    #     hrms = np.expand_dims(hrms, axis=0)
    h,w,c = hrms.shape
    if GNyq is not None:
        GNyq = GNyq
    elif sensor == 'random':
        GNyq = np.random.normal(loc=0.3, scale=0.03, size=c)
    elif sensor=='QB':
        GNyq = [0.34, 0.32, 0.30, 0.22]  # Band Order: B,G,R,NIR
    elif sensor=='IK':
        GNyq = [0.26,0.28,0.29,0.28]    # Band Order: B,G,R,NIR
    elif sensor=='GE' or sensor == 'WV4':
        GNyq = [0.23,0.23,0.23,0.23]    # Band Order: B,G,R,NIR   
    elif sensor=='WV3':
        GNyq = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315]
    elif sensor=='WV2':
        GNyq = ([0.35]*7+[0.27])
    else:
        GNyq = [0.3]*c
    mtf = [GNyq2win(GNyq=tmp) for tmp in GNyq]
    ms_lr = [ndimage.convolve(hrms[:,:,idx], tmp_mtf, mode='wrap') for idx, tmp_mtf in enumerate(mtf)]
    ms_lr = np.stack(ms_lr, axis=2)
    return ms_lr


##########################################################
# No reference metrics for Full Resolution Assesment.
##########################################################

def HQNR(ps_ms, ms, pan, S=32, sensor=None, ratio=4):
    msexp = upsample_mat_interp23(ms, ratio)
    Dl = D_lambda_K(ps_ms, msexp, ratio, sensor, S)
    Ds = D_s(ps_ms, ms, pan, ratio, S, 1)
    HQNR_value = (1-Dl)*(1-Ds)
    return Dl, Ds, HQNR_value


def D_lambda_K(fused, msexp, ratio, sensor, S):
    if fused.shape != msexp.shape:
        raise('The two images must have the same dimensions')
    N, M, _ = fused.shape
    if N % S != 0 or N % S != 0:
        raise('numbers of rows and columns must be multiple of the block size.')

    fused_degraded = MTF_Filter(fused, sensor, ratio, GNyq=None)
    q2n = Q2n(msexp, fused_degraded, S, S)
    return 1-q2n


def D_s(img_fake, img_lm, pan,ratio, S, q=1):
    """Spatial distortion
    img_fake, generated HRMS
    img_lm, LRMS
    pan, HRPan"""
    # fake and lm
    assert img_fake.ndim == img_lm.ndim == 3, 'MS images must be 3D!'
    H_f, W_f, C_f = img_fake.shape
    H_r, W_r, C_r = img_lm.shape
    assert H_f // H_r == W_f // W_r == ratio, 'Spatial resolution should be compatible with scale'
    assert C_f == C_r, 'Fake and lm should have the same number of bands!'
    # fake and pan
    # if pan.ndim == 2: pan = np.expand_dims(pan, axis=2)
    if pan.ndim==3: pan = pan.squeeze(2)
    H_p, W_p = pan.shape
    assert H_f == H_p and W_f == W_p, "Pan's and fake's spatial resolution should be the same"
    # get LRPan, 2D
    pan_lr = Image.fromarray(pan).resize((int(1/ratio*H_p),int(1/ratio*W_p)), resample=Image.BICUBIC)
    pan_lr = np.array(pan_lr)
  
    Q_hr = []
    Q_lr = []
    for i in range(C_f):
        # for HR fake
        band1 = img_fake[..., i]
        Q_hr.append(_qindex(band1, pan, block_size=S))
        band1 = img_lm[..., i]
        Q_lr.append(_qindex(band1, pan_lr, block_size=S))
    Q_hr = np.array(Q_hr)
    Q_lr = np.array(Q_lr)
    D_s_index = (np.abs(Q_hr - Q_lr) ** q).mean()
    return D_s_index ** (1/q)



def pan_calc_metrics_rr(PS, GT, scale, img_range):
    GT = np.array(GT).astype(np.float)
    PS = np.array(PS).astype(np.float)
    RMSE = (GT - PS)/img_range
    RMSE = np.sqrt((RMSE*RMSE).mean())
    # cc = CC(GT,PS)
    sam = SAM(GT, PS)
    ergas = ERGAS(PS, GT, scale=scale)
    # Qave = Q_AVE(GT, PS)
    # scc = sCC(GT, PS)
    q2n = Q2n(GT,PS)
    psnr = peak_signal_noise_ratio(GT, PS, data_range=img_range)
    # return {'SAM':sam, 'ERGAS':ergas, 'Q2n':q2n, 'CC': cc, 'RMSE':RMSE}
    return {'PSNR': psnr, 'SAM':sam, 'ERGAS':ergas, 'Q2n': q2n}

if __name__ == '__main__':

# import numpy as np
# import os
# ArbRPN_dir = '/home/ser606/Documents/LihuiChen/ArbRPN_20200916/results/SR/RNN_RESIDUAL_BI_PAN_FB_MASK/QB-FIX-4/x4/'
# GT_dir = '/home/ser606/Documents/LihuiChen/PanSharp_dataset/QB/test/MTF/4bands/HRMS_npy'
# LRMS_dir = '/home/ser606/Documents/LihuiChen/PanSharp_dataset/QB/test/MTF/4bands/LRMS_npy'
# PAN_dir = '/home/ser606/Documents/LihuiChen/PanSharp_dataset/QB/test/MTF/4bands/LRPAN_npy'
# # save_root = '/home/ser606/Documents/LihuiChen/compare/extra/reduced_resolution/test'

# ArbRPN_files = os.listdir(ArbRPN_dir)
# ArbRPN_files.sort()
# GT_files = os.listdir(GT_dir)
# GT_files.sort()
# LRMS_files = os.listdir(LRMS_dir)
# LRMS_files.sort()
# PAN_files = os.listdir((PAN_dir))
# PAN_files.sort()
# cc = []
# sam = []
# ergas = []
# q_ave = []
# q2n = []
# for i in range(len(ArbRPN_files)):
#     ps = np.load(os.path.join(ArbRPN_dir, ArbRPN_files[i]))
#     ps = ps.astype(np.float)
#     gt = np.load(os.path.join(GT_dir, GT_files[i])).astype(np.float)
#     pan = np.load(os.path.join(PAN_dir, PAN_files[i])).astype(np.float)
#     # print('%s  ||  %s \n'%(ArbRPN_files[i], PAN_files[i]))
#     # print((ps.dtype))
#     cc.append(CC(ps, gt))
#     sam.append(SAM(gt, ps))
#     ergas.append(ERGAS(ps, gt))
#     q_ave.append(Q_AVE(ps, gt))
#     q2n.append(Q2n(gt, ps, 32, 32))

# print(mean(q2n))

    # import os
    # TFNET_dir = '/home/ser606/Documents/LihuiChen/ArbRPN_20200916/results/SR/PARABIRNN/QB-Vanilla-BiRNN-FR/x4'
    # LRMS_dir = '/home/ser606/Documents/LihuiChen/PanSharp_dataset/QB/test/MS_full_resolution/MS_npy'
    # PAN_dir = '/home/ser606/Documents/LihuiChen/PanSharp_dataset/QB/test/PAN_full_resolution/PAN_npy'
    # # save_root = '/home/ser606/Documents/LihuiChen/compare/extra/full_resolution/test'ArbRPN_files = os.listdir(ArbRPN_dir)
    # TFNET_files = os.listdir(TFNET_dir)
    # TFNET_files.sort()
    # # GT_files = os.listdir(GT_dir)
    # # GT_files.sort()
    # LRMS_files = os.listdir(LRMS_dir)
    # LRMS_files.sort()
    # PAN_files = os.listdir((PAN_dir))
    # PAN_files.sort()
    # Dl_results = []
    # Ds_results = []
    # Qnr_results = []
    # for i in range(len(TFNET_files)):
    #     print('processing the %d-th image.\n'%i)
    #     ps = np.load(os.path.join(TFNET_dir, TFNET_files[i]))
    #     ps = ps.astype(np.float)
    #     lrms = np.load(os.path.join(LRMS_dir, LRMS_files[i])).astype(np.float)
    #     pan = np.load(os.path.join(PAN_dir, PAN_files[i])).astype(np.float)
    #     # print('%s  ||  %s \n'%(ArbRPN_files[i], PAN_files[i]))
    #     # print((ps.dtype))
    #     msexp = upsample_mat_interp23(lrms, 4)
    #     dl, ds, hqnr = HQNR(ps, lrms, msexp, pan, 32, 'QB', 4)
    #     Dl_results.append(dl)
    #     Ds_results.append(ds)
    #     Qnr_results.append(hqnr)
    # print(sum(Qnr_results)/len(Qnr_results))
    a = np.random.randn(240, 240, 128)
    b = np.random.randn(240, 240, 128)
    Q2n(a,b)
