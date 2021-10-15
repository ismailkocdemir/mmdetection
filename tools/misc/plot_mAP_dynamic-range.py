import json
import os
from collections import defaultdict
from cycler import cycler

from matplotlib.legend import _get_legend_handles_labels
import matplotlib.pyplot as plt
import numpy as np

'''
exp_names = ["ldr", "hdr", "gamma", "tmqi", "bestexp", "fattal", "durand", "mantiuk", "reinhard", "ashikhmin", "deeptmo",
                    "TMO-GAN + RetinaNet COCO", "TMO-GAN + RetinaNet OOD"]

exp_names_on_plot = ["LDR", "HDR", "Gamma", "Best TMQI", "Std. LDR", "Fattal", "Durand", "Mantiuk", "Reinhard", "Ashikhmin", "TMO-GAN",
                    "TMO-GAN + RetinaNet COCO", "TMO-GAN + RetinaNet OOD"]
'''

exp_names = ["frcnn_ldr", "frcnn_hdr", "frcnn_gamma", "frcnn_tmqi", "frcnn_bestexp", "frcnn_fattal", "frcnn_durand", "frcnn_mantiuk", "frcnn_reinhard", "frcnn_ashikhmin", "frcnn_deeptmo",
                    "TMO-GAN + Faster-RCNN COCO", "TMO-GAN + Faster-RCNN OOD"]

exp_names_on_plot = ["LDR", "HDR", "HDR w Gamma", "Best TMQI", "Std. LDR", "Fattal", "Durand", "Mantiuk", "Reinhard", "Ashikhmin", "TMO-GAN",
                    "TMO-GAN + Faster-RCNN COCO", "TMO-GAN + Faster-RCNN OOD"]

# Cityscapes classes
#classes = ['car', 'person', 'bicycle', 'rider', 'motorcycle', 'truck', 'bus', 'train']

# OOD (pascal-voc) classes
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tv/monitor']

ranges = {
    "area" : [[0, 32], [32, 64], [64,128], [128,896]]
}

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plot_mAP_scores(mode='drange', cat_type=None , y_map_limit=(0., 0.7), y_object_limit=(0,5000)):

    assert mode in ['drange', 'entropy', 'lumin', 'area'], "Undefined metric: {}".format(mode)
    if cat_type:
        assert cat_type in ['all','common', 'rare'] or cat_type in classes, "Udefined category type: {}".format(cat_type)
    
    if mode == 'area':
        str_ranges = [('$(' + str(a[0]) + '^2$, $' + str(a[1]) + '^2)$' ) for a in ranges[mode]]
    else:
        str_ranges = ['low', 'low-med', 'med-high', 'high']
    
    cmap = get_cmap(len(exp_names))
    default_cycler = (cycler(marker=['o', 'v', 's', '^','D']))


    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    xlabel = "dynamic range" if mode == "drange" else "entropy" if mode == "entropy" else "luminance"
    ax1.set_xlabel(mode, fontsize=15)
    
    ax1.set_ylabel('mAP', color=color, fontsize=15)
    
    ax1.set_ylim(y_map_limit)
    ax1.tick_params(axis='x', labelsize=13, rotation=45)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=15)
    ax1.set_prop_cycle(default_cycler)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    
    ax2.set_ylabel('object count', color=color, fontsize=15)  # we already handled the x-label with ax1
    ax2.set_ylim(y_object_limit)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=15)

    area_idx = [i for i in range(len(str_ranges))]
    ax1.set_xticks(area_idx, str_ranges)
    ax1.grid(True)
    
    d_count = json.load(open("/HDD/H3DR/HDR4RTT/0_RESIZED/stats/{}_range_count_{}.txt".format(mode, cat_type)))
    for idx, nm in enumerate(exp_names):
        #if cat_type!='all':
        d_ap = json.load(open("test_outputs/text_data/{}_{}_{}.txt".format(nm, mode, cat_type)))
        #else:
        #    d_ap = json.load(open("/truba/home/ikocdemir/data/HDR4RTT/0_RESIZED/stats/{}_{}.txt".format(nm, mode)))
        print("PLOTTING:", nm, flush=True)
        ax1.plot(area_idx, d_ap["ap"][:len(ranges[mode])], label=exp_names_on_plot[idx], c=cmap(idx)) #, label=nm, marker='o')
        
    counts = d_count["count"][:len(ranges[mode])]
    sc = zip(str_ranges, counts)
    for idx, (x,y) in enumerate(sc):
        if idx == 0:
            ax2.scatter(x, y, color="tab:blue", marker='x', label='object count')
        else:
            ax2.scatter(x, y, color="tab:blue", marker='x')
    
    ax1.legend(*ax1.get_legend_handles_labels(), bbox_to_anchor=(1.04, -0.1), loc="lower left", borderaxespad=4.5, prop={'size': 15})
    ax2.legend(*ax2.get_legend_handles_labels(), bbox_to_anchor=(1.04, -0.2), loc="lower left", borderaxespad=4.5, prop={'size': 15})
    
    fig.tight_layout()  
    
    plot_title = "{}_lineplot_frcnn.png".format(mode, cat_type) if cat_type else "{}_lineplot_frcnn.png".format(mode)
    plt.savefig(plot_title)


if __name__ == "__main__":
    

    plot_mAP_scores('drange', cat_type='all')
    plot_mAP_scores('entropy', cat_type='all')
    plot_mAP_scores('lumin', cat_type='all')

    '''    
    plot_mAP_scores('drange', 'common', y_object_limit=(0, 3000))
    plot_mAP_scores('drange', 'rare', y_object_limit=(0, 3000))
    plot_mAP_scores('entropy', 'common', y_object_limit=(0, 3000))
    plot_mAP_scores('entropy', 'rare', y_object_limit=(0, 3000))
    plot_mAP_scores('lumin', 'common', y_object_limit=(0, 3000))
    plot_mAP_scores('lumin', 'rare', y_object_limit=(0, 3000))
    '''    
    


    
