import argparse
import glob
import os.path as osp

import cityscapesscripts.helpers.labels as CSLabels
import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from collections import Counter

def collect_files(img_dir, gt_dir, hdr=False):
    print(hdr, img_dir)
    suffix = 'leftImg16bit.png' if hdr else 'leftImg8bit.png'
    files = []
    for img_file in glob.glob(osp.join(img_dir, '**/*.png')):
        assert img_file.endswith(suffix), img_file
        inst_file = gt_dir + img_file[
            len(img_dir):-len(suffix)] + 'gtFine_instanceIds.png'
        # Note that labelIds are not converted to trainId for seg map
        segm_file = gt_dir + img_file[
            len(img_dir):-len(suffix)] + 'gtFine_labelIds.png'
        files.append((img_file, inst_file, segm_file))
    assert len(files), f'No images found in {img_dir}'
    print(f'Loaded {len(files)} images from {img_dir}')

    return files

def collect_annotations(files, nproc=1):
    print('Loading annotation images')
    if nproc > 1:
        images = mmcv.track_parallel_progress(
            load_img_info, files, nproc=nproc)
    else:
        images = mmcv.track_progress(load_img_info, files)

    return images


def load_img_info(files):
    img_file, inst_file, segm_file = files
    inst_img = mmcv.imread(inst_file, 'unchanged')
    # ids < 24 are stuff labels (filtering them first is about 5% faster)
    unique_inst_ids = np.unique(inst_img[inst_img >= 24])
    anno_info = []
    for inst_id in unique_inst_ids:
        # For non-crowd annotations, inst_id // 1000 is the label_id
        # Crowd annotations have <1000 instance ids
        label_id = inst_id // 1000 if inst_id >= 1000 else inst_id
        label = CSLabels.id2label[label_id]
        if not label.hasInstances or label.ignoreInEval:
            continue

        category_id = label.id
        iscrowd = int(inst_id < 1000)
        mask = np.asarray(inst_img == inst_id, dtype=np.uint8, order='F')
        mask_rle = maskUtils.encode(mask[:, :, None])[0]

        area = maskUtils.area(mask_rle)
        # convert to COCO style XYWH format
        bbox = maskUtils.toBbox(mask_rle)

        # for json encoding
        mask_rle['counts'] = mask_rle['counts'].decode()

        anno = dict(
            iscrowd=iscrowd,
            category_id=category_id,
            bbox=bbox.tolist(),
            area=area.tolist(),
            segmentation=mask_rle)
        anno_info.append(anno)
    video_name = osp.basename(osp.dirname(img_file))
    img_info = dict(
        # remove img_prefix for filename
        file_name=osp.join(video_name, osp.basename(img_file)),
        height=inst_img.shape[0],
        width=inst_img.shape[1],
        anno_info=anno_info,
        segm_file=osp.join(video_name, osp.basename(segm_file)))

    return img_info

def cvt_annotations(image_infos, out_json_name, ratio=0):
    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['categories'] = []
    out_json['annotations'] = []

    if ratio:
        new_size = int(len(image_infos)*ratio)
        image_infos = np.random.choice(image_infos, new_size, replace=False)

    class_counter = Counter()
    for image_info in image_infos:
        #if divide_size and img_id%divide_size:
        #    img_id += 1
        #    continue
        image_info['id'] = img_id #// 2 if divide_size else img_id
        anno_infos = image_info.pop('anno_info')
        out_json['images'].append(image_info)
        for anno_info in anno_infos:

            label = CSLabels.id2label[anno_info['category_id']]
            class_counter[label] += 1

            anno_info['image_id'] = image_info['id']
            anno_info['id'] = ann_id
            out_json['annotations'].append(anno_info)
            ann_id += 1
        img_id += 1
    for label in CSLabels.labels:
        if label.hasInstances and not label.ignoreInEval:
            cat = dict(id=label.id, name=label.name)
            out_json['categories'].append(cat)

    if len(out_json['annotations']) == 0:
        out_json.pop('annotations')

    mmcv.dump(out_json, out_json_name, indent=4)
    print("Instances per Category")
    for clss, cnt in class_counter.items():
        print("  ", clss, cnt)
    return out_json

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to COCO format')
    parser.add_argument('cityscapes_path', help='cityscapes data path')
    parser.add_argument('--img-dir', type=str)
    parser.add_argument('--gt-dir', default='gtFine', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=8, type=int, help='number of process') 
    parser.add_argument('-r', '--ratio', default=0, type=float, help='ratio of the dataset for training on a subset')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cityscapes_path = args.cityscapes_path
    out_dir = osp.join(cityscapes_path, args.out_dir) if args.out_dir else cityscapes_path
    mmcv.mkdir_or_exist(out_dir)

    img_dir = osp.join(cityscapes_path, args.img_dir)
    gt_dir = osp.join(cityscapes_path, args.gt_dir)
   
    set_name = dict(
        train='instancesonly_filtered_gtFine_train.json',
        val='instancesonly_filtered_gtFine_val.json',
        test='instancesonly_filtered_gtFine_test.json')

    _highbit = "lefImg8bit" not in img_dir
    for split, json_name in set_name.items():
        print(f'Converting {split} into {json_name}')
        with mmcv.Timer(
                print_tmpl='It tooks {}s to convert Cityscapes annotation'):
            files = collect_files(
                osp.join(img_dir, split), osp.join(gt_dir, split), _highbit)
            image_infos = collect_annotations(files, nproc=args.nproc)
            ratio = args.ratio and "train" in json_name
            cvt_annotations(image_infos, osp.join(out_dir, json_name), (args.ratio if ratio else 0))


if __name__ == '__main__':
    main()
