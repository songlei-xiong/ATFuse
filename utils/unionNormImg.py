'''
author: Lihui Chen
copywrite: Lihui Chen
email: lihuichenscu@foxmail.com
'''
import numpy as np
import glob
import sys
import scipy.io as sio
# sys.path.append('../dataProc4manuscript/')
from hist_adjust import hist_line_stretch, hist_line_stretchv2
from plot_subfigs_colorbar import plot_subfigs_colorbar
from metrics import pan_calc_metrics_rr
# %matplotlib inline'GSA': '/Users/qianqian/Documents/codeWorkspace/result/result/GSA/*.npy',
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os
# from collections import Order`edDict
def main():
    results_dir = './SOTA_HS_Sharpen'
    dataset = 'result_chikusei'  #'result_HyperALi' 'result_chikusei'
    method_dir = {
        # 'Ours': '',
        # 'GSA': '%s/%s/GSA/*.npy'%(results_dir, dataset),
        'SFIMHS': '%s/%s/SFIMHS/*.npy'%(results_dir, dataset),
        'GLPHS': '%s/%s/GLPHS/*.npy'%(results_dir, dataset),
        'CNMF': '%s/%s/CNMF/*.npy'%(results_dir, dataset),
        # 'ICCV15': '%s/%s/ICCV15/*.npy'%(results_dir, dataset),
        'FUSE': '%s/%s/FUSE/*.npy'%(results_dir, dataset),
        'HySure': '%s/%s/HySure/*.npy'%(results_dir, dataset),
        'MHFnet': '%s/%s/MHFnet/*.npy'%(results_dir, dataset),
        'HSRnet':'%s/%s/HSRnet/*.npy'%(results_dir, dataset),
        # 'MoGCDN':'%s/%s/MoGCDN/*.mat'%(results_dir, dataset),
        'Ours': '%s/%s/Ours/*.npy'%(results_dir, dataset),
    }

    GT_dir = '%s/%s/GT/*.npy'%(results_dir, dataset)
    # MSHR = './'
    GT_file = glob.glob(GT_dir)
    GT_file.sort()
    savedir = '%s/%s/visualization_final/'%(results_dir, dataset)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    rgb_band = (30, 10, 4)
    rad_res = 4095

    # dl_method_files = {key:glob.glob(value) for key, value in DL_method.items()}
    # for key in dl_method_files.keys(): dl_method_files[key].sort()

    # dl_method_files = {
    #     'MHFnet':sio.loadmat(dl_method_files['MHFnet'][0])['chikusei'].astype(np.float),
    #     'HSRnet':sio.loadmat(dl_method_files['HSRnet'][0])['output'].astype(np.float)/12.0*rad_res,
    #     }


    method_files = {key:glob.glob(value) for key, value in method_dir.items()}
    for key in method_files.keys(): method_files[key].sort()
    key_list = list(method_files.keys())
    imgdict = dict()
    for idx_img in range(len(GT_file)):
        # imgdict = {key: np.load(value[idx_img]).astype(np.float) for key, value in method_files.items()}
        for idx,(key, value) in enumerate(method_files.items()):
            if key in ['MogCDN']:
                imgdict[key] = sio.loadmat(value[idx_img])['data'].astype(np.float).squeeze()
            else:
                imgdict[key] = np.load(value[idx_img]).astype(np.float).squeeze()
            # if key=='MHFnet':
            #     dlimgdict[key] = sio.loadmat(value[idx_img])['HyperALi%d'%(idx_img+1)].astype(np.float).squeeze()
            #     # dlimgdict[dlkey] = dlvalue[idx_img,...]
            # if key=='HSRnet':
            #     dlimgdict[key] = sio.loadmat(value[idx_img])['data'].astype(np.float).squeeze()#/12*rad_res
            #     # dlimgdict[dlkey] = dlvalue[idx_img,...]
            # if key == 'Ours':
            #     dlimgdict[key] = sio.loadmat(value[idx_img])['data'].astype(np.float).squeeze()
            
        GT= np.load(GT_file[idx_img]).astype(np.float)
        # GT= sio.loadmat(GT_file[idx_img])['HSHR'].astype(np.float)
        # MSHR = sio.loadmat(GT_file[idx_img])['MSHR'].astype(np.float)
        # MSHR,_,_ = hist_line_stretchv2(MSHR[:,:,(5,3,1)], rad_res, bound=[0.01, 0.99])
        
        res_imgdict = {'%s Res.'%key:np.abs(value-GT).mean(axis=2)/rad_res for key, value in imgdict.items()}

        imgdict = {key: value[:,:, rgb_band] for key, value in imgdict.items()}
        GT, lowThe, highThe = hist_line_stretch(GT[:,:,rgb_band], rad_res, bound=[0.01, 0.995])
        # print('%s'%str(tmpMetric))
        for key, value in imgdict.items():
            imgdict[key] = (value-lowThe)/(highThe-lowThe)
            # for idx, (low, high) in enumerate(zip(lowThe, highThe)):
            #     tmp =  value[:,:,idx]
            #     tmp[np.where(tmp>high)]=high
            #     tmp[np.where(tmp<low)]=low
            #     tmp = (tmp-low)/(high-low)
            #     value[:,:,idx] = tmp
            # if idx_img ==0:
            #     method_metrics[key] = {key: 0 for key in tmpMetric[key].keys()}
            # for key_metric, value_metric in tmpMetric[key].items():
            #     method_metrics[key][key_metric] += value_metric/len(method_files['GT'])

        img_name = savedir + os.path.basename(GT_file[idx_img])[:-4] + '.png'
        total_imgdict = {**imgdict, **{'GT': GT}, **res_imgdict}
        fig = plot_subfigs_colorbar({**total_imgdict}, plotsize=[2, len(total_imgdict.keys())//2+1], bound=[0,0.05], save_name=img_name, ifpdf=True)
    return fig
    # for method_key, metric in method_metrics.items():
    #     print('%s: \n %s'%(method_key, str(metric)))+

if __name__ == '__main__':
    main()
