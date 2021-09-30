import sys
import os
import json
import glob
from shutil import copyfile, move
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy as entropy_scipy
import cv2

ID_2_CAT =  {
    24:"person",
    25:"rider",
    26:"car",
    27:"truck",
    28:"bus",
    31:"train",
    32:"motorcycle",
    33:"bicycle"
}

ranges = {
    "area" : [[0**2, 32**2], [32**2, 64**2], [64**2, 96**2], [96**2, 128**2], [128**2, 192**2], 
                [192**2, 256**2], [256**2, 320**2], [320**2, 480**2], [480**2, 1e5**2]],
    "lumin" : [],
    "drange" : [],
    "entropy" : []
}

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

markers = ["o", "v","^","<",">","s","P","*"]
cat_markers = dict([(val, markers[idx]) for idx, (key,val) in enumerate(ID_2_CAT.items())])
CMAP = get_cmap(len(ID_2_CAT))
cat_colours = dict([(val, CMAP(idx)) for idx, (key,val) in enumerate(ID_2_CAT.items())])

def get_bbox_category(val, mode="area"):
    if mode ==  "area":
        for i,r in enumerate(ranges[mode]):
            if val >= r[0] and val < r[1]:
                return "s" + str(i+1)
            
    range_labels = ["L", "L-M", "M-H", "H"]
    for i,r in enumerate(ranges[mode]):
        if val >= r[0] and val < r[1]:
            mode_abbr = "DR" if mode == "drange" else "LUM" if mode == "lumin" else "ENT"
            return range_labels[i] + "-" + mode_abbr


def is_bbox_valid(bbox_dims, image_dims):
    x,y,w,h = bbox_dims
    im_h, im_w = image_dims

    if w < 8 or h < 8:
        return False
    
    if x+w > im_w or y+h > im_h:
        return False
    
    return True


def get_image_luminance_category(avg_lum):
    return "low_lum" if avg_lum < 0.021 else "high_lum" if avg_lum > 0.029 else "mid_lum"

def get_image_dynamic_range_category(drange):
    return "low_drange" if drange < 8 else "high_drange" 

def get_image_entropy_category(entropy):
    return "low_ent" if entropy < 0.90737 else "high_ent" if entropy > 0.96793 else "mid_ent"

def convert_to_gray(image):
    return 0.2126*image[:,:,2] + 0.7152*image[:,:,1] + 0.0722*image[:,:,0]

def read_image_ldr(image_file):
    rgb = cv2.imread(image_file, cv2.IMREAD_COLOR)
    rgb = cv2.resize(rgb, (1024,512))
    return rgb

def read_image(image_file, remove_outliers=False, m=3.0):
    rgb = cv2.imread(image_file, cv2.IMREAD_COLOR + cv2.IMREAD_ANYDEPTH)
    if rgb is None:
        return rgb
    rgb = cv2.resize(rgb, (1024,512))
    rgb = rgb.astype(np.float32) / 65536
    
    rgb = convert_to_gray(rgb)
    
    if remove_outliers:
        rgb = remove_outliers_from_image(rgb, m)
    return rgb

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

def get_key(image, remove_outliers=True, m=3.):
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
    
    bins =  np.linspace(0.0,1.0, 64) #'sqrt'
    hist, _ = np.histogram(_image.flatten(), bins=bins, density=True)
    _ent = entropy_stable(hist)

    K = np.log(len(hist)+1)
    _ent /= K

    return _ent

def get_avg_luminance(image):
    return np.mean(image)

def get_avg_log_luminance(image, remove_outliers=False, m=3.):
    _image = image.copy()
    if remove_outliers:
        _image = remove_outliers_from_image(_image, m)
    return np.exp(np.mean( np.log(_image + 1e-6) ))

def set_range_and_count(pair_dict, anno_type, metric, data_folder):
    global ranges
    
    if anno_type != "train":
        print("Annotation type is not TRAIN. Not setting the ranges.")
        return
    
    stat_folder = os.path.join(data_folder, "stats")
    if not os.path.exists(stat_folder):
        os.makedirs(stat_folder)

    if metric is not None:
        for bbox_cat, value_list in pair_dict:
            if bbox_cat == "all":
                if metric != "area":
                    range_prc = [0, 5, 50, 95, 100]
                    ranges[metric] = [ [np.percentile(value_list, range_prc[idx]), np.percentile(value_list, range_prc[idx+1])] \
                                            for idx in range(len(range_prc)-1) ]
                else:
                    range_prc = ["S" + str(idx+1) for idx in range(len(ranges["area"]))]
                    
                print("Current measure:", metric)
                print(ranges[metric])
                plt.clf()
                #if metric == "drange":
                #    np_hist = np.histogram(value_list, bins=np.linspace(0,20.0, 200), density=True)
                #    np.save("cityscapes_hist_0.npy", np_hist[0])
                #    np.save("cityscapes_hist_1.npy", np_hist[1])
                fig = plt.hist(value_list, bins='sqrt', density=False, alpha=0.85)

                min_ylim, max_ylim = plt.ylim()
                for i in range(1, len(range_prc)-1):
                    plt.axvline(ranges[metric][i][0], color='red', linestyle='dashed', linewidth=1)
                    plt.text(ranges[metric][i][0]*1.05, max_ylim*0.95, '{}\u1D57\u02B0'.format(range_prc[i]))
                
                plt.title('BBox {}: {} set'.format(metric, anno_type))
                plt.xlabel("bbox {}".format(metric))
                plt.ylabel("frequency")
                plt.savefig(os.path.join(stat_folder, "bbox_{}_hist_{}_99th.png".format(metric, anno_type)))

    for bbox_cat, value_list in pair_dict:
        counts = []
        for r in ranges[metric]:
            counts.append( value_list[ (value_list>=r[0]) & (value_list<r[1]) ].shape[0] )

        json.dump({"count": counts}, 
                open(
                    os.path.join(stat_folder, "{}_range_count_{}.txt".format(metric, bbox_cat)),
                    'w'
                )
            )

def main_image(data_folder, annotate=False, mode="lum"):
    assert mode.lower() in ["lum", "drange"]
    anno_files = glob.glob(os.path.join(data_folder, "annotations_16bit", "*.json"))

    for anno_file in anno_files:
        anno_type = "train" if "train" in anno_file else "val" if "val" in anno_file else "test"
        anno_filename = anno_file.split("/")[-1]
         
        print("Processing:", anno_type)

        anno_data = None
        with open(anno_file) as json_data:
            anno_data = json.load(json_data)
            json_data.close()

        image_info = anno_data["images"]
        categories = anno_data["categories"]
        anno_info = anno_data["annotations"] if "annotations" in anno_data else None
        
        file_list_dr = defaultdict(list)
        file_list_lum = defaultdict(list)
        if annotate:
            levels = ["low_lum", "mid_lum", "high_lum"] if mode=="lum" else ["low_drange", "high_drange"]    
            levels_json_ldr = dict([(item, defaultdict(list)) for item in levels])
            levels_json_hdr = dict([(item, defaultdict(list)) for item in levels])
            for _key in levels_json_ldr.keys():
                anno_folder_ldr = os.path.join(data_folder, "annotations_8bit_image_" + _key)
                anno_folder_hdr = os.path.join(data_folder, "annotations_16bit_image_" + _key)
                os.makedirs(anno_folder_ldr, exist_ok=True)
                os.makedirs(anno_folder_hdr, exist_ok=True)
                levels_json_ldr[_key]["categories"] = categories
                levels_json_hdr[_key]["categories"] = categories
            img_ids = dict([(item, 0) for item in levels])
        
        val_list = defaultdict(list)
        total_images = len(image_info)
        for _idx, _info in enumerate(image_info):
            print("{}%".format(int(100*_idx/total_images)), end="\r")

            file_name = _info["file_name"]
            image = read_image(os.path.join("leftImg16bit", anno_type, file_name))

            if mode == "lum" or not annotate:
                val_lum = get_avg_log_luminance(image, remove_outliers=False) 
                _level_key = get_image_luminance_category(val_lum)
                file_list_lum[_level_key].append(file_name)
                val_list["lum"].append(val_lum)
            if mode == "drange" or not annotate:
                val_drange = get_dynamic_range(image, remove_outliers=True, m=2) 
                _level_key = get_image_dynamic_range_category(val_drange)
                file_list_dr[_level_key].append(file_name)
                val_list["drange"].append(val_drange)

            if annotate:
                _info_hdr = _info.copy()
                _info_hdr["id"] = img_ids[_level_key]
                levels_json_hdr[_level_key]["images"].append(_info_hdr)
                
                _info_ldr = _info_hdr.copy()
                _info_ldr["file_name"] = _info_ldr["file_name"].replace("16bit", "8bit")
                levels_json_ldr[_level_key]["images"].append(_info_ldr)

                if anno_info == None:
                    img_ids[_level_key] += 1
                    continue
                for anno in anno_info:
                    if anno["image_id"] != _info["id"]:
                        continue
                    
                    anno_copy = anno.copy()
                    anno_copy["image_id"] = img_ids[_level_key]
                    levels_json_hdr[_level_key]["annotations"].append(anno_copy)
                    levels_json_ldr[_level_key]["annotations"].append(anno_copy)

                img_ids[_level_key] += 1
        
        if annotate:
            for _level in levels:
                ldr_path = os.path.join(data_folder, "annotations_8bit_image_" + _level, anno_filename)
                with open(ldr_path, 'w', encoding='utf-8') as f:
                    json.dump(levels_json_ldr[_level], f, ensure_ascii=False, indent=4)
                
                hdr_path = os.path.join(data_folder, "annotations_16bit_image_" + _level, anno_filename)
                with open(hdr_path, 'w', encoding='utf-8') as f:
                    json.dump(levels_json_hdr[_level], f, ensure_ascii=False, indent=4) 
            
        
        if len(val_list["lum"]):
            '''
            val_list_lum = np.array(val_list["lum"])
            upper_per = [np.percentile(val_list_lum, p ) for p in range(80, 100, 5)]
            lower_per = [np.percentile(val_list_lum, p ) for p in range(20, 0, -5)]
            
            for idx, pr in enumerate(upper_per):
                print("lum:{1:.4f}, size:{2} ".format( 80 + idx*5, pr, len(val_list_lum[val_list_lum>pr])) )
            
            for idx, pr in enumerate(lower_per):
                print("lum:{1:.4f}, size:{2} ".format( 20 - idx*5, pr, len(val_list_lum[val_list_lum<pr])) )
            '''

            plt.clf()
            fig = plt.hist(val_list["lum"], bins="sqrt", density=False)
            plot_title = "log-luminance"
            plt.title('Image {}: {} set'.format(plot_title, anno_type))
            plt.xlabel(plot_title)
            plt.ylabel("frequency")
            plt.savefig(os.path.join(data_folder, "image_{}_hist_{}.png".format(plot_title, anno_type)))

            cntr = 0
            for k,v in file_list_lum.items():
                with open('images_{}.txt'.format(k), 'w') as f:
                    for item in v:
                        f.write("%s\n" % item)
                
                tmos = ["ReinhardGlob", "ReinhardLoc", "Mantiuk", "Opt"]
                for tmo in tmos:
                    for im in v:
                        src = os.path.join("/home/ismail/H3DR/demo_object_detection/cityscapes/test_outputs/{}".format(tmo), im)
                        dst = os.path.join("/home/ismail/H3DR/demo_object_detection/cityscapes/test_outputs/comparison", k, tmo, im.split("/")[-1])
                        copyfile(src, dst) 

        
        if len(val_list["drange"]):
            '''
            val_list_dr = np.array(val_list["drange"])
            upper_per = [np.percentile(val_list_dr, p ) for p in range(80, 100, 5)]
            lower_per = [np.percentile(val_list_dr, p ) for p in range(20, 0, -5)]
            
            for idx, pr in enumerate(upper_per):
                print("dr:{1:.2f}, size:{2} ".format( 80 + idx*5, pr, len(val_list_dr[val_list_dr>pr])) )
            
            for idx, pr in enumerate(lower_per):
                print("dr:{1:.2f}, size:{2} ".format( 20 - idx*5, pr, len(val_list_dr[val_list_dr<pr])) )
            '''
            plt.clf()
            fig = plt.hist(val_list["drange"], bins="sqrt", density=False)
            plot_title = "dynamic-range"
            plt.title('Image {}: {} set'.format(plot_title, anno_type))
            plt.xlabel(plot_title)
            plt.ylabel("frequency")
            plt.savefig(os.path.join(data_folder, "image_{}_hist_{}.png".format(plot_title, anno_type)))

            for k,v in file_list_dr.items():
                with open('images_{}.txt'.format(k), 'w') as f:
                    for item in v:
                        f.write("%s\n" % item)


def main_bbox(data_folder, annotate=False, skip_train=False):
    
    global ranges

    anno_files_glob = glob.glob(os.path.join(data_folder, "annotations_16bit", "*.json"))
    anno_files = []

    order = ["train", "val", "test"]
    for item in order:
        for anno_file in anno_files_glob:
            if item in anno_file and "backup" not in anno_file: 
                anno_files.append(anno_file)

    for anno_file in anno_files:
        
        anno_type = "train" if "train" in anno_file else "val" if "val" in anno_file else "test"

        if anno_type == "train" and skip_train:
            continue
        
        anno_filename = anno_file.split("/")[-1]

        print("Processing:", anno_type)

        anno_data = None
        with open(anno_file) as json_data:
            anno_data = json.load(json_data)
            json_data.close()

        anno_info = anno_data["annotations"] if "annotations" in anno_data else None
        if anno_info == None:
            continue
        image_info = anno_data["images"]
        categories = anno_data["categories"]

        img_id_2_idx = {}
        image_info_ldr = []
        for idx,_img_info in enumerate(image_info):
            img_id_2_idx[_img_info["id"]] = idx
            _img_info_ldr = _img_info.copy()
            _img_info_ldr["file_name"] = _img_info_ldr["file_name"].replace('16bit', '8bit')
            image_info_ldr.append(_img_info_ldr)

        curr_image = None
        curr_image_ldr_lum = None
        curr_image_ldr_drange = None
        curr_image_ldr_ent = None
        prev_id = None
        category_entropy = defaultdict(list)
        category_drange = defaultdict(list)
        category_lumin = defaultdict(list)
        category_entropy_drange = defaultdict(list)
        category_entropy_lumin = defaultdict(list)
        lumin_list = defaultdict(list)
        drange_list = defaultdict(list)
        entropy_list = defaultdict(list)
        area_list = defaultdict(list)
        total_annos = len(anno_info)
        anno_list = []
        image_list = []

        #plt.clf()
        for _idx, anno in enumerate(anno_info):
            print("{}%".format(int(100*_idx/total_annos)), end="\r")

            img_id = anno["image_id"]
            if img_id != prev_id:
                prev_id = img_id
                image_path = image_info[img_id_2_idx[img_id]]["file_name"]
                image_path = os.path.join(data_folder, "leftImg16bit", anno_type, image_path)
                curr_image = read_image(image_path, remove_outliers=True, m=3.0)

                if curr_image is None:
                    print("could not found:", image_path)
                    continue

                if not annotate and anno_type == 'val':
                    if curr_image_ldr_lum is not None:
                        write_path = image_path_ldr.replace("leftImgOptExp", "leftImgWithBBOX")
                        write_folder = "/".join(write_path.split("/")[:-1])
                        if not os.path.exists(write_folder):
                            os.makedirs(write_folder)
                        cv2.imwrite(write_path.replace(".png", "_LUM.png"), curr_image_ldr_lum)
                        cv2.imwrite(write_path.replace(".png", "_DRANGE.png"), curr_image_ldr_drange)
                        cv2.imwrite(write_path.replace(".png", "_ENT.png"), curr_image_ldr_ent)
                    image_path_ldr = os.path.join(data_folder, "leftImgOptExp", anno_type, image_info[img_id]["file_name"])
                    curr_image_ldr_lum = read_image_ldr(image_path_ldr)
                    curr_image_ldr_drange = curr_image_ldr_lum.copy()
                    curr_image_ldr_ent = curr_image_ldr_lum.copy()
            else:
                if curr_image is None:
                    continue
            
            bbox_cat = ID_2_CAT[anno["category_id"]]
            bbox_dims = [int(item/2.0) for item in anno["bbox"]]

            if not is_bbox_valid(bbox_dims, curr_image.shape):
                continue
            
            x, y, w, h = bbox_dims
            normal_area = w*h*4 
            box_image = np.zeros((w,h)) 
            box_image = curr_image[y:y+h, x:x+w]
            
            area = np.log2(normal_area)
            avg_lum = get_avg_log_luminance(box_image, remove_outliers=False)
            drange = get_dynamic_range(box_image, remove_outliers=False)
            entropy = get_histogram_entropy(box_image, remove_outliers=False)

            if not annotate and anno_type == 'val':
                curr_image_ldr_lum = cv2.rectangle(curr_image_ldr_lum, (x, y), (x + w, y + h), (0,0,255), 1)
                curr_image_ldr_drange = cv2.rectangle(curr_image_ldr_drange, (x, y), (x + w, y + h), (0,0,255), 1)
                curr_image_ldr_ent = cv2.rectangle(curr_image_ldr_ent, (x, y), (x + w, y + h), (0,0,255), 1)
                lum_cat = get_bbox_category(avg_lum, "lumin")
                drange_cat = get_bbox_category(drange, "drange")
                ent_cat = get_bbox_category(entropy, "entropy")

                if lum_cat in ["L-LUM", "H-LUM"]:
                    cv2.putText(curr_image_ldr_lum, lum_cat, 
                                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                if lum_cat in ["L-LUM", "H-LUM"]:
                    cv2.putText(curr_image_ldr_drange, drange_cat, 
                                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                cv2.putText(curr_image_ldr_ent, ent_cat, 
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

            if annotate:
                anno["area"] = normal_area
                anno["lumin"] = avg_lum
                anno["drange"] = drange
                anno["entropy"] = entropy
                anno_list.append(anno)

            lumin_list[bbox_cat].append(avg_lum)            
            drange_list[bbox_cat].append(drange)
            entropy_list[bbox_cat].append(entropy)
            area_list[bbox_cat].append(normal_area)

            
            bbox_cat = "common" if  bbox_cat in ["car", "person", "bicycle"] else "rare"
            lumin_list["all"].append(avg_lum)
            lumin_list[bbox_cat].append(avg_lum)
            
            drange_list["all"].append(drange)
            drange_list[bbox_cat].append(drange)

            entropy_list["all"].append(entropy)
            entropy_list[bbox_cat].append(entropy)
            
            area_list["all"].append(normal_area)
            area_list[bbox_cat].append(normal_area)
            
            #plt.scatter(area, drange, color='black', alpha=0.3)
            category_drange[bbox_cat].append((area, drange))
            category_entropy[bbox_cat].append((area, entropy))
            category_lumin[bbox_cat].append((area, avg_lum))
            category_entropy_drange[bbox_cat].append((entropy, drange))
            category_entropy_lumin[bbox_cat].append((entropy, avg_lum))

        #plt.xlabel("log-area")
        #plt.ylabel("dynamic-range")
        #plt.savefig(os.path.join(data_folder, "log-area_vs_drange_{}.pdf".format(anno_type)))
        
        area_list_np = [(k,np.array(v)) for k,v in area_list.items()]
        set_range_and_count(area_list_np, anno_type, "area", data_folder)
        
        entropy_list_np = [(k,np.array(v)) for k,v in entropy_list.items()]
        set_range_and_count(entropy_list_np, anno_type, "entropy", data_folder)

        lumin_list_np = [(k,np.array(v)) for k,v in lumin_list.items()]
        set_range_and_count(lumin_list_np, anno_type, "lumin", data_folder)

        drange_list_np = [(k,np.array(v)) for k,v in drange_list.items()]
        set_range_and_count(drange_list_np, anno_type, "drange", data_folder)

        if annotate:
            full_intervals = {}
            for metric, values in ranges.items():
                min_val = 0.0
                max_val = 100.0
                if metric == "area":
                    max_val = 1e6
                elif metric == "lumin":
                    max_val = 1.0
                elif metric == "drange":
                    max_val = 20.0
                elif metric == "entropy":
                    max_val == 1.0
                
                fixed_range = values[:]
                fixed_range[-1][1] = max_val
                fixed_range.insert(0 , [min_val, max_val])
                full_intervals[metric] = fixed_range 
            
            anno_folder_ldr = os.path.join(data_folder, "annotations_8bit")
            anno_folder_hdr = os.path.join(data_folder, "annotations_16bit")

            if not os.path.exists(anno_folder_ldr):
                os.makedirs(anno_folder_ldr, exist_ok=True)
            if not os.path.exists(anno_folder_hdr):
                os.makedirs(anno_folder_hdr, exist_ok=True) 
            
            hdr_path = os.path.join(anno_folder_hdr, anno_filename)
            hdr_anno = {
                "bbox_intervals":full_intervals,
                "categories":categories, 
                "images":image_info, 
                "annotations":anno_list
            }
            with open(hdr_path, 'w', encoding='utf-8') as f:
                json.dump(hdr_anno, f, ensure_ascii=False, indent=4)

            ldr_path = os.path.join(anno_folder_ldr, anno_filename)
            ldr_anno = {
                "bbox_intervals":full_intervals,
                "categories":categories, 
                "images":image_info_ldr, 
                "annotations":anno_list
            }
            with open(ldr_path, 'w', encoding='utf-8') as f:
                json.dump(ldr_anno, f, ensure_ascii=False, indent=4)


        '''
        plt.clf()
        fig, axs = plt.subplots(2,3,figsize=(18, 7))
        cat_types = ['car',  'person', 'bicycle', 'rider', 'motorcycle', 'truck']
        #for _,k in ID_2_CAT.items():
        for y_idx, ax_list in enumerate(axs):
            for x_idx, ax1 in enumerate(ax_list):
                cat_type = cat_types[x_idx + 3*y_idx]
                R = np.corrcoef(area_list[cat_type], drange_list[cat_type])[1,0]
                print(cat_type, "R:", R)
                
                ax1.set_title(cat_type, fontsize=17)
                ax1.set_xlabel("log-area")
                ax1.set_ylabel("dynamic range")
                for item in category_drange[cat_type]:
                    ax1.scatter(item[0], item[1], marker='.', color='black') #, marker=cat_markers[k], label=k)
                
                ax1.text(0.1, 0.9, 'p:{:.3f}'.format(R), horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, fontsize=15)
        
                #unzipped = list(zip(*category_drange[k]))
                #R = np.corrcoef(unzipped[0], unzipped[1])[1,0]
                #print("Correlation coefficient for drange-vs-area:", R)
                plt.savefig(os.path.join("/home/ismail/H3DR/demo_object_detection/cityscapes/test_outputs/plots/scatter/", "area_drange_scatter_{}.png".format(k)))
                plt.clf()

                
                plt.xlabel("log-area")
                plt.ylabel("log-luminance")
                for item in category_lumin[k]:
                    plt.scatter(item[0], item[1], marker='.') #, color=cat_colours[k], marker=cat_markers[k], label=k)
                plt.savefig(os.path.join("/home/ismail/H3DR/demo_object_detection/cityscapes/test_outputs/plots/scatter/", "area_lumin_scatter_{}.png".format(k)))
                plt.clf()
                
                plt.xlabel("log-area")
                plt.ylabel("entropy")
                for item in category_entropy[k]:
                    plt.scatter(item[0], item[1], marker='.') #, color=cat_colours[k], marker=cat_markers[k], label=k)
                plt.savefig(os.path.join("/home/ismail/H3DR/demo_object_detection/cityscapes/test_outputs/plots/scatter/", "area_entropy_scatter_{}.png".format(k)))
                plt.clf()

                plt.xlabel("entropy")
                plt.ylabel("dynamic range")
                for item in category_entropy_drange[k]:
                    plt.scatter(item[0], item[1], marker='.') #, color=cat_colours[k], marker=cat_markers[k], label=k)
                plt.savefig(os.path.join("/home/ismail/H3DR/demo_object_detection/cityscapes/test_outputs/plots/scatter/", "entropy_drange_scatter_{}.png".format(k)))
                plt.clf()

                plt.xlabel("entropy")
                plt.ylabel("log-luminance")
                for item in category_entropy_lumin[k]:
                    plt.scatter(item[0], item[1], marker='.') #, color=cat_colours[k], marker=cat_markers[k], label=k)
                plt.savefig(os.path.join("/home/ismail/H3DR/demo_object_detection/cityscapes/test_outputs/plots/scatter/", "entropy_lumin_scatter_{}.png".format(k)))
                plt.clf()       
            '''
        
        #plot_title = "scatterplot.pdf"
        #plt.savefig(os.path.join("/home/ismail/H3DR/demo_object_detection/cityscapes/test_outputs/plots/", plot_title))

def split_anno(anno_folder):
    anno_files = glob.glob(os.path.join(anno_folder, "*train.json"))

    for anno_file in anno_files:
        anno_filename = anno_file.split("/")[-1]

        anno_data = None
        with open(anno_file) as json_data:
            anno_data = json.load(json_data)
            json_data.close()

        anno_info = anno_data["annotations"] if "annotations" in anno_data else None
        if anno_info == None:
            continue
        image_info = anno_data["images"]
        categories = anno_data["categories"]

        images_train = []
        images_val = []
        anno_train = []
        anno_val = []
        
        id_2_fname = {}
        for img in image_info:
            fname = img['file_name']
            img_id = img['id']

            id_2_fname[img_id] = fname

            if 'stuttgart' in fname or 'ulm' in fname or 'bochum' in fname:
                images_val.append(img)
            else:
                images_train.append(img)
        
        for anno in anno_info:
            img_id = anno["image_id"]
            fname = id_2_fname[img_id]

            if 'stuttgart' in fname or 'ulm' in fname or 'bochum' in fname:
                anno_val.append(anno)
            else:
                anno_train.append(anno)


        train_path = os.path.join(anno_folder, anno_filename)
        train_path_backup = os.path.join(anno_folder, anno_filename.replace('train', 'train_backup'))
        val_path = os.path.join(anno_folder, anno_filename.replace('train', 'val'))
        test_path = os.path.join(anno_folder, anno_filename.replace('train', 'test'))
        test_path_backup = os.path.join(anno_folder, anno_filename.replace('train', 'test_backup'))

        copyfile(train_path, train_path_backup)
        copyfile(test_path, test_path_backup)
        copyfile(val_path, test_path)


        full_ann_train = {"categories":categories, "images":images_train, "annotations":anno_train}
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(full_ann_train, f, ensure_ascii=False, indent=4)

        full_ann_val = {"categories":categories, "images":images_val, "annotations":anno_val}
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(full_ann_val, f, ensure_ascii=False, indent=4)



def make_val_split(data_folder, copy_folders=False):  
    split_anno(os.path.join(data_folder, "annotations_8bit"))
    split_anno(os.path.join(data_folder, "annotations_16bit"))

    gt_folder = os.path.join(data_folder, "gtFine")
    train_path = os.path.join(gt_folder, 'train') 
    val_path = os.path.join(gt_folder, 'val')
    test_path = os.path.join(gt_folder, 'test')
    test_path_orig = os.path.join(gt_folder, 'test_orig') 

    move(test_path, test_path_orig)
    move(val_path, test_path)

    os.makedirs(val_path)

    for city in ["stuttgart", "ulm", "bochum"]:
        city_source = os.path.join(train_path, city)
        city_dest = os.path.join(val_path, city)

        move(city_source, city_dest)


    if copy_folders:
        #folders = ["leftImg8bit", "leftImg16bit","leftImg16bitGamma", "leftImgDurand", "leftImgFattal", "leftImgMantiuk", "leftImgOptExp", "leftImgReinhardGlobal", "leftImgReinhardLocal"]
        folders = ["leftImgDurand", "leftImgFattal", "leftImgMantiuk", "leftImgOptExp", "leftImgReinhardGlobal", "leftImgReinhardLocal"]
        for fol in folders:
            img_fold = os.path.join(data_folder, fol)
            
            train_path = os.path.join(img_fold, 'train') 
            val_path = os.path.join(img_fold, 'val')
            test_path = os.path.join(img_fold, 'test')
            test_path_orig = os.path.join(img_fold, 'test_orig') 

            move(test_path, test_path_orig)
            move(val_path, test_path)

            os.makedirs(val_path)

            for city in ["stuttgart", "ulm", "bochum"]:
                city_source = os.path.join(train_path, city)
                city_dest = os.path.join(val_path, city)

                move(city_source, city_dest)


def calc_mean_var(img_folder):
    
    psum    = np.array([0.0, 0.0, 0.0])
    psum_sq = np.array([0.0, 0.0, 0.0])
    idx = 0
    count = 0

    # loop through images
    for r, d, f in os.walk(img_folder):
        for _file in f:
            try:
                image_path = os.path.join(r, _file)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR + cv2.IMREAD_ANYDEPTH).astype(np.float32)

                psum    = psum + image.sum(axis=(0,1))
                psum_sq = psum_sq + (image * image).sum(axis = (0, 1))
                count += (image.shape[0] * image.shape[1])

                if idx % 100 == 0:
                    print(idx, end='\r')
                idx += 1
            except Exception as e:
                print(e)
                idx += 1
                continue

    ####### FINAL CALCULATIONS

    # mean and std
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = np.sqrt(total_var)

    # output
    print(img_folder)
    print("mean", total_mean)
    print("std", total_std)
    print("var", total_var)
    print("==================")



           
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("dataset folder is not provided")
        sys.exit()

    data_folder = str(sys.argv[1])
    #make_val_split(data_folder)
    #main_bbox(data_folder, annotate=True, skip_train=False)
    #main_image(data_folder, annotate=True)
    calc_mean_var(data_folder)
