__author__ = 'tsungyi'

import os
import copy
import datetime
import time
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import pycocotools.mask as maskUtils

class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(
            self, cocoGt=None, cocoDt=None, iouType='segm', 
            bbox_metric='area', 
            exp_name=None,
            dump_path=None,
            bbox_intervals=None,
            use_cityscapes=False
        ):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt = cocoDt  # detections COCO API
        self.evalImgs = defaultdict(
            list)  # per-image per-category evaluation results [KxAxI] elements
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params(
            iouType=iouType, 
            bbox_metric=bbox_metric, 
            exp_name=exp_name, 
            bbox_intervals=bbox_intervals,
            use_cityscapes=use_cityscapes
        )
        self.bbox_metric = bbox_metric # luminance metric: log-luminance, dynamic-range or area.
        self.exp_name = exp_name
        self.dump_path = dump_path
        self.bbox_intervals = bbox_intervals

        self.areaRng_backup = [0**2, 1e5**2]
        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.ious = {}  # ious between all gts and dts
        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())
            self.params.common_catIdx = [idx for idx, cat_id in enumerate(self.params.catIds) if cat_id in self.params.common_catIds]
            self.params.rare_catIdx = [idx for idx, cat_id in enumerate(self.params.catIds) if cat_id in self.params.rare_catIds]

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(
                self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(
                self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(
            list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results
         (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if p.useSegm is not None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.
                  format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId)
                     for imgId in p.imgIds for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [
            evaluateImg(imgId, catId, areaRng, maxDet) for catId in catIds
            for areaRng in p.areaRng for imgId in p.imgIds
        ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]
            yg = g[1::3]
            vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]
            x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]
            y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]
                yd = d[1::3]
                if k1 > 0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) &
                    # (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0 - xd), axis=0) + np.max(
                        (z, xd - x1), axis=0)
                    dy = np.max((z, y0 - yd), axis=0) + np.max(
                        (z, yd - y1), axis=0)
                e = (dx**2 + dy**2) / vars / (gt['area'] + np.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            g_area = g[self.bbox_metric]
            if g['ignore'] or (g_area < aRng[0] or g_area > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(
            self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store
                        # appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d[self.bbox_metric] < aRng[0] or d[self.bbox_metric] > aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        
        return {
            'image_id': imgId,
            'category_id': catId,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg
            #'dtArea': np.array([d["bbox"][2]*d["bbox"][3] for d in dt]),
            #'gtArea': np.array([g["bbox"][2]*g["bbox"][3] for g in gt])
        }

    def accumulate(self, p=None):
        '''
        Accumulate per image evaluation results and store the result in
        self.eval

        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones(
            (T, R, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [
            n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng))
            if a in setA
        ]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)

        '''
        tp_by_area = [[] for i in range(T)]
        fp_by_area = [[] for i in range(T)]
        fn_by_area = [[] for i in range(T)]
        #gt_count = [0 for i in range(A)]
        '''
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate(
                        [e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different
                    # results. mergesort is used to be consistent as Matlab
                    # implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate(
                        [e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:,
                                                                        inds]
                    '''
                    if self.exp_name:
                        gtm = np.concatenate([e['gtMatches'] for e in E], axis=1)

                        dtA = np.concatenate([e['dtArea'][0:maxDet] for e in E])[inds]

                        gtA = np.concatenate([e['gtArea'] for e in E])
                    '''
                    dtIg = np.concatenate(
                        [e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]


                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm),
                                         np.logical_not(dtIg))
                    
                    '''
                    if self.exp_name:
                        fns = np.logical_and(np.logical_not(gtm), np.logical_not(gtIg))

                        for tr in range(T):
                            areas_tp = dtA[tps[tr]]
                            #print(dtA.shape, tps[tr].shape, areas_tp.shape)
                            areas_fp = dtA[fps[tr]]
                            areas_fn = gtA[fns[tr]]
                            tp_by_area[tr] += areas_tp.tolist()
                            fp_by_area[tr] += areas_fp.tolist()
                            fn_by_area[tr] += areas_fn.tolist()
                    '''

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R, ))
                        ss = np.zeros((R, ))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for
                        # accessing elements use python array gets significant
                        # speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:  # noqa: E722
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)

        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
            'scores': scores,
        }
        
        '''
        json.dump({"count": gt_count[1:]}, open("/home/ismail/H3DR/text_data/{}_count.txt".format(self.exp_name),'w'))
        if self.exp_name:
            for tr in range(T):
                bin_edges = [(16*i)**2 for i in range(6)]
                plt.clf()
                bins, edges, _ = plt.hist(tp_by_area[tr], bins=bin_edges, density=False)
                plot_title = "TP_{:.2f}-IOU_{}".format(0.5 + tr*0.05, self.exp_name)
                json.dump({"bins":bins.tolist()}, open("/home/ismail/H3DR/{}.txt".format(plot_title),'w'))
                plt.title(plot_title)
                plt.xlabel("bbox area")
                plt.ylabel("frequency")
                plt.savefig(os.path.join("/home/ismail/H3DR/plots", "{}_hist.png".format(plot_title)))
                plt.clf()
                bins, edges, _ = plt.hist(fp_by_area[tr], bins=bin_edges, density=False)
                plot_title = "FP_{:.2f}-IOU_{}".format(0.5 + tr*0.05, self.exp_name)
                json.dump({"bins":bins.tolist()}, open("/home/ismail/H3DR/{}.txt".format(plot_title),'w'))
                plt.title(plot_title)
                plt.xlabel("bbox area")
                plt.ylabel("frequency")
                plt.savefig(os.path.join("/home/ismail/H3DR/plots", "{}_hist.png".format(plot_title)))
                plt.clf()
                bins, edges, _ = plt.hist(fn_by_area[tr], bins=bin_edges, density=False)
                plot_title = "FN_{:.2f}-IOU_{}".format(0.5 + tr*0.05, self.exp_name)
                json.dump({"bins":bins.tolist()}, open("/home/ismail/H3DR/{}.txt".format(plot_title),'w'))
                plt.title(plot_title)
                plt.xlabel("bbox area")
                plt.ylabel("frequency")
                plt.savefig(os.path.join("/home/ismail/H3DR/plots", "{}_hist.png".format(plot_title)))
        '''
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter
        setting
        '''
        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100, kinds=None):
            p = self.params
            iStr = '{:<18} {} @[ IoU={:<9} | interval={:>6s} | maxDets={:>3d} ] = {:0.3f}'  # noqa: E501
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [
                i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng
            ]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if kinds is not None:
                    s = s[:, :, kinds, :, :]
                #if isinstance(kinds, int):
                #    s = s[:, :, aind, mind]
                #else:
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(
                iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets,
                            mean_s))
            return mean_s

        def _summarizeDets():
            _f = self.bbox_metric == "area"
            
            if _f:
                stats = np.zeros((14, ))
                _kinds = None
                stats[0] = _summarize(1, kinds=_kinds)
                stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], kinds=_kinds)
                stats[2] = _summarize(1,
                                    iouThr=.75,
                                    maxDets=self.params.maxDets[2], kinds=_kinds)
                stats[3] = _summarize(1,
                                    areaRng='s1',
                                    maxDets=self.params.maxDets[2], kinds=_kinds)
                stats[4] = _summarize(1,
                                    areaRng='s2',
                                    maxDets=self.params.maxDets[2], kinds=_kinds)
                stats[5] = _summarize(1,
                                    areaRng='s3',
                                    maxDets=self.params.maxDets[2], kinds=_kinds)
                stats[6] = _summarize(1,
                                    areaRng='s4',
                                    maxDets=self.params.maxDets[2], kinds=_kinds)
                stats[7] = _summarize(1,
                                    areaRng='s5',
                                    maxDets=self.params.maxDets[2], kinds=_kinds)
                stats[8] = _summarize(1,
                                    areaRng='s6',
                                    maxDets=self.params.maxDets[2], kinds=_kinds)
                stats[9] = _summarize(1,
                                    areaRng='s7',
                                    maxDets=self.params.maxDets[2], kinds=_kinds)
                stats[10] = _summarize(1,
                                    areaRng='s8',
                                    maxDets=self.params.maxDets[2], kinds=_kinds)
                stats[11] = _summarize(0, maxDets=self.params.maxDets[0], kinds=_kinds)
                stats[12] = _summarize(0, maxDets=self.params.maxDets[1], kinds=_kinds)
                stats[13] = _summarize(0, maxDets=self.params.maxDets[2], kinds=_kinds)
                if self.exp_name:
                    json.dump({"ap": stats[3:11].tolist()}, 
                            open(
                                os.path.join(self.dump_path, "{}_{}.txt".format(self.exp_name, self.bbox_metric))
                                ,'w'
                            )
                        )

            else:
                total_classes = len(self.params.catIds)
                stats = np.zeros((30, ))
                for i in range(total_classes + 3):
                    if i < 3:
                        _kinds = None if i==0 else self.params.common_catIdx if i==1 else self.params.rare_catIdx
                        stats[0+i] = _summarize(1, kinds=_kinds)
                        stats[3+i] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], kinds=_kinds)
                        stats[6+i] = _summarize(1,
                                            iouThr=.75,
                                            maxDets=self.params.maxDets[2], kinds=_kinds)
                        stats[9+i] = _summarize(1,
                                            areaRng="low",
                                            maxDets=self.params.maxDets[2], kinds=_kinds)
                        stats[12+i] = _summarize(1,
                                            areaRng="med_low",
                                            maxDets=self.params.maxDets[2], kinds=_kinds)
                        stats[15+i] = _summarize(1,
                                            areaRng="med_high",
                                            maxDets=self.params.maxDets[2], kinds=_kinds)
                        stats[18+i] = _summarize(1,
                                            areaRng="high",
                                            maxDets=self.params.maxDets[2], kinds=_kinds)
                        
                        stats[21+i] = _summarize(0, maxDets=self.params.maxDets[0], kinds=_kinds)
                        stats[24+i] = _summarize(0, maxDets=self.params.maxDets[1], kinds=_kinds)
                        stats[27+i] = _summarize(0, maxDets=self.params.maxDets[2], kinds=_kinds)

                    else:
                        cat_id = self.params.catIds[i-3]
                        cat_name = self.params.Id2Label[cat_id]
                        _kinds = [self.params.catIds.index(cat_id),]
                        stat_list = []
                        stat_list.append(_summarize(1,
                                        areaRng="low",
                                        maxDets=self.params.maxDets[2], kinds=_kinds))
                        stat_list.append(_summarize(1,
                                        areaRng="med_low",
                                        maxDets=self.params.maxDets[2], kinds=_kinds))
                        stat_list.append(_summarize(1,
                                        areaRng="med_high",
                                        maxDets=self.params.maxDets[2], kinds=_kinds))
                        stat_list.append(_summarize(1,
                                        areaRng="high",
                                        maxDets=self.params.maxDets[2], kinds=_kinds))
                        if self.exp_name:
                            json.dump({"ap": stat_list}, 
                                    open(
                                        os.path.join(
                                            self.dump_path,
                                            "{}_{}_{}.txt".format(self.exp_name, self.bbox_metric, cat_name)
                                        )
                                        ,'w'
                                    )
                                )
                        
                    if self.exp_name:
                        json.dump({"ap": stats[9:21:3].tolist()}, 
                                    open(
                                        os.path.join(
                                            self.dump_path,
                                            "{}_{}_all.txt".format(self.exp_name, self.bbox_metric)
                                            )
                                        ,'w'
                                    )
                                )
                        json.dump({"ap": stats[10:21:3].tolist()}, 
                                open("{}_{}_common.txt".format(
                                    self.exp_name, self.bbox_metric),'w'))
                        json.dump({"ap": stats[11:21:3].tolist()}, 
                                open("{}_{}_rare.txt".format(
                                    self.exp_name, self.bbox_metric),'w'))
            return stats

        def _summarizeKps():
            stats = np.zeros((10, ))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(
            self, 
            bbox_metric, 
            exp_name, 
            bbox_intervals,
            use_cityscapes
        ):
        assert bbox_metric in ["lumin", "drange", "area", "entropy"]

        self.imgIds = []
        self.catIds = []

        if not use_cityscapes:
            self.common_catIds = list(range(1,7))
            self.rare_catIds = list(range(7,21))
            self.Id2Label = {
                1: "pottedplant",
                2: "person",
                3: "car",
                4: "bottle",
                5: "diningtable",
                6: "chair",
                7: "boat",
                8: "motorbike",
                9: "sofa",
                10: "tv-monitor",
                11: "aeroplane",
                12: "dog",
                13: "bicycle",
                14: "bus",
                15: "bird",
                16: "horse",
                17: "train",
                18: "sheep",
                19: "cat",
                20: "cow"
            }
        else:
            self.common_catIds = [24, 26, 33]
            self.rare_catIds = [25, 27, 28, 31, 32]
            self.Id2Label = {
                24:"person",
                25:"rider",
                26:"car",
                27:"truck",
                28:"bus",
                31:"train",
                32:"motorcycle",
                33:"bicycle"
            }


        self.common_catIdx = []
        self.rare_catIdx = []

        self.exp_name = exp_name
        # np.arange causes trouble.  the data point on arange is slightly
        # larger than the true value
        self.iouThrs = np.linspace(.5,
                                   0.95,
                                   int(np.round((0.95 - .5) / .05)) + 1,
                                   endpoint=True)
        self.recThrs = np.linspace(.0,
                                   1.00,
                                   int(np.round((1.00 - .0) / .01)) + 1,
                                   endpoint=True)
        self.maxDets = [1, 10, 100]
        
        old_intervals = {
            "area":[[0**2, 1e5**2], [0**2, 32**2], [32**2, 64**2],
                            [64**2, 128**2], [128**2, 1e5**2]],
            "lumin":[[0.0, 1.0], 
                            [0.0, 0.004961234983056784], 
                            [0.004961234983056784, 0.00503345625475049], 
                            [0.00503345625475049, 0.005887560360133647], 
                            [0.005887560360133647, 1.0]
                        ],
            "drange":[[0.0, 20.0],
                            [0.0, 0.011953067779541016], 
                            [0.011953067779541016, 0.17247676849365234], 
                            [0.17247676849365234, 1.6741512298583983], 
                            [1.6741512298583983, 6.005528926849365]
                        ],
            "entropy": [[0.0, 1.0], 
                            [0.0, 0.04788953263725463], 
                            [0.04788953263725463, 0.20370051357384036], 
                            [0.20370051357384036, 0.39132539690074014], 
                            [0.39132539690074014, 1.0]
                        ]
        }

        if bbox_intervals is not None:
            self.areaRng = bbox_intervals[bbox_metric]
        else:
            self.areaRng = old_intervals[bbox_metric]

        if bbox_metric == "area":
            if bbox_intervals is None:
                self.areaRngLbl = ['all', 'small', 'med_small', 'med_large', 'large']
            else:
                ['all', 's1', 's2', 's3', 's4','s5','s6', 's7', 's8']
        else:
            self.areaRngLbl = ['all', 'low', 'med_low', 'med_high', 'high']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly
        # larger than the true value
        self.iouThrs = np.linspace(.5,
                                   0.95,
                                   int(np.round((0.95 - .5) / .05)) + 1,
                                   endpoint=True)
        self.recThrs = np.linspace(.0,
                                   1.00,
                                   int(np.round((1.00 - .0) / .01)) + 1,
                                   endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0**2, 1e5**2], [64**2, 256**2], [256**2, 1e5**2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
            .87, .87, .89, .89
        ]) / 10.0

    def __init__(self, iouType='segm', bbox_metric="area", exp_name="default-exp", bbox_intervals=None, use_cityscapes=False):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams(bbox_metric, exp_name, bbox_intervals, use_cityscapes)
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
