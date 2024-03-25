'''
author: Lihui Chen
copywrite: Lihui Chen
email: lihuichenscu@foxmail.com
'''
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
################  { for colorbar }  ################
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
color_list = ['#0000FF', '#00FF33', '#FFFF33', '#FF0000', '#FF00FF']
my_cmap = LinearSegmentedColormap.from_list('rain', color_list)
cm.register_cmap(cmap=my_cmap)
################  { for colorbar }  ################

def plot_subfigs_colorbar(imgDict, plotsize=[1, 6], bound=[0, 1], save_name=None, ifpdf=False, fontsize=6):
    '''
    inputs: 
        imgDict: dict of images
        plotsize: size of subplots
        bound: the low- and up- bound for image
    return:
    '''
    plt.rcParams['font.family'] = 'Times'
    fig = plt.figure()      
    for idx, key in enumerate(imgDict.keys()):
        ax = fig.add_subplot(plotsize[0],plotsize[1], idx+1)
        data = imgDict[key]
        if 'res' in key.lower():
            im = ax.imshow(data , vmin = bound[0], vmax = bound[1], cmap='rain')
        else:
            data = np.clip(data, 0, 1)
            ax.imshow(data)
        ax.set_title(key, fontsize=fontsize)
        ax.set_xticks([])
        ax.set_yticks([])
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    cb = plt.colorbar(im, cax=cax)
    # cb = plt.colorbar(im, ax = axes.ravel().tolist())
    # cb = plt.colorbar(im, ax = ax)
    cb.ax.tick_params(labelsize=fontsize)
    # plt.show()
    if save_name is not None:
        plt.savefig(save_name)
        
    if ifpdf:
        plt.savefig(save_name.replace('.png', '.svg'), dpi=300)
    # plt.close() 
    return fig
    
    