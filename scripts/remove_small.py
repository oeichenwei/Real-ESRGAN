import cv2
import os
import glob
import numpy as np

input_dir = 'datasets/Face_finetune/'
img_paths_gt = sorted(glob.glob(os.path.join(input_dir, 'Face_HR/*')))
img_paths_lq = sorted(glob.glob(os.path.join(input_dir, 'Face_LQ/*')))

for i in range(len(img_paths_lq)):
    img_lq = cv2.imread(img_paths_lq[i])
    img_hq = cv2.imread(img_paths_gt[i])
    if img_lq is None or img_hq is None:
        continue

    if (img_hq.shape[0] != 4 * img_lq.shape[0]) or (img_hq.shape[1] != 4 * img_lq.shape[1]):
        hq_new = img_hq[0: img_lq.shape[0] * 4, 0:img_lq.shape[1] * 4]
        hq_new = np.ascontiguousarray(hq_new)
        cv2.imshow("ss", hq_new)
        print(hq_new.shape)
        print(img_hq.shape, img_lq.shape, i)
        cv2.imwrite(img_paths_gt[i], hq_new)
