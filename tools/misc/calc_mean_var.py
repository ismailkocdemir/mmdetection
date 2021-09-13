import os
import cv2
import numpy as np

def calc_mean_var(img_folder):
    # placeholders
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

if __name__ == "__main__":
    folders = ["leftImg8bit", "leftImg16bit","leftImg16bitGamma"]
    #folders = [ "leftImgDurand", "leftImgFattal", "leftImgMantiuk"]
    #folders = ["leftImgOptExp", "leftImgReinhardGlobal", "leftImgReinhardLocal"]

    for fol in folders:
        img_fol = os.path.join(fol, "train")
        calc_mean_var(img_fol)
        print("==========")
