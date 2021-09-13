from functools import partial
import os.path as osp
import numpy as np
import torch
from six.moves import map, zip

from ..mask.structures import BitmapMasks, PolygonMasks

import mmcv
from scipy.stats import entropy as entropy_scipy

def remove_outliers_from_image(data, m=3.):
    '''
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return np.clip(data, data[s<m].min(), data[s<m].max())
    
    '''
    darkest = np.percentile(data, 1)
    brightest = np.percentile(data, 99)
    data[data>brightest] = brightest
    data[data<darkest] = darkest
    return data


def get_key(image, remove_outliers=False, m=3.):
    _image = image.copy()
    _image += 1e-6
    if remove_outliers:
        _image = remove_outliers_from_image(_image, m)
    
    log_avg = np.mean( np.log(_image) )
    log_min = np.log(np.min(_image))
    log_max = np.log(np.max(_image))

    return (log_avg - log_min)/(log_max - log_min)

def get_dynamic_range(image, remove_outliers=False, m=3.):
    _image = image.copy()
    _image += 1e-6
    if remove_outliers:
        _image = remove_outliers_from_image(_image, m)

    return np.log2(_image.max()) - np.log2(_image.min())

def entropy_stable(values):
    ''' fixes less than or equal to zero values before calculating entropy'''
    values_stable = (values - values.min()) + 1e-10
    return entropy_scipy(values_stable)

def get_histogram_entropy(image, remove_outliers=True, m=3.):
    _image = image.copy()

    if remove_outliers:
        _image = remove_outliers_from_image(_image, m)
    
    #_image = np.log2(_image + 1e-8)
    
    bins =  np.linspace(0.0,1.0, 2**16) #'sqrt'
    hist, _ = np.histogram(_image.flatten(), bins=bins, density=True)
    _ent = entropy_stable(hist)

    K = np.log(len(hist)+1)
    _ent /= K

    return _ent

def get_avg_log_luminance(image, remove_outliers=False, m=3.):
    _image = image.copy()
    if remove_outliers:
        _image = remove_outliers_from_image(_image, m)
    return np.exp(np.mean( np.log(_image + 1e-6) ))

def calculate_bbox_metric(tensor, metas, result, bbox_metric='entropy', out_dir=None, model=None):
    assert bbox_metric in ["entropy", "drange", "lumin"], "Undefiend bbox quality metric: {}".format(bbox_metric)
    num_imgs = tensor.size(0)
    np.seterr('raise')
    if isinstance(result[0], tuple):
        _result = [r[0] for r in result]
    else:
        _result = result[:]

    mean = metas[0]['img_norm_cfg']['mean']
    std = metas[0]['img_norm_cfg']['std']
    to_rgb = metas[0]['img_norm_cfg']['to_rgb']
    
    values = []
    for img_id in range(num_imgs):
        image = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        h, w, _ = metas[img_id]['img_shape']
        image = image[:h, :w]
        #image = mmcv.imdenormalize(image, mean, std)
        
        '''
        if out_dir:
            out_file = osp.join(out_dir, "HDR_version" ,metas[img_id]['ori_filename'])
        
            model.module.show_result(
                (image*65536).astype(np.int16),
                result[img_id],
                show=None,
                out_file=out_file,
                score_thr=0.5)
        '''
        
        min_val = -326.18848 #-12.278792
        max_val = 65504.0 #69482.45
        image = (image - min_val) / (max_val - min_val)
        image = 0.2126*image[:,:,2] + 0.7152*image[:,:,1] + 0.0722*image[:,:,0]
        image = remove_outliers_from_image(image)
        
        bbox_result = _result[img_id]
        for ann in bbox_result:
            for i in range(len(ann)):
                x1,y1,x2,y2 = [int(np.floor(a)) for a in ann[i][:-1]]

                if x2 < x1 or y2 < y1 or y1 < 0 or x1 < 0 or y2 >= image.shape[0] or x2 >= image.shape[1]:
                    values.append(0)
                    continue
                
                box_image = image[y1:y2, x1:x2]    
                if bbox_metric == 'entropy':
                    values.append(get_histogram_entropy(box_image))
                elif bbox_metric == 'drange':
                    values.append(get_dynamic_range(box_image))
                elif bbox_metric == 'lumin':
                    values.append(get_avg_log_luminance(box_image))

    return values

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size
    count)"""
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret


def mask2ndarray(mask):
    """Convert Mask to ndarray..

    Args:
        mask (:obj:`BitmapMasks` or :obj:`PolygonMasks` or
        torch.Tensor or np.ndarray): The mask to be converted.

    Returns:
        np.ndarray: Ndarray mask of shape (n, h, w) that has been converted
    """
    if isinstance(mask, (BitmapMasks, PolygonMasks)):
        mask = mask.to_ndarray()
    elif isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    elif not isinstance(mask, np.ndarray):
        raise TypeError(f'Unsupported {type(mask)} data type')
    return mask
