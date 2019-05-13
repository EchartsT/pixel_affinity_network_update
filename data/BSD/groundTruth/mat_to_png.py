import os
from scipy.io import loadmat
import numpy as np
import cv2

gt_dir = './train'

gt_path = os.path.join(os.getcwd(), 'groundtruth')
if not os.path.exists(gt_path):
    os.makedirs(gt_path)

for gtfile in os.listdir(gt_dir):
    if gtfile.endswith('.mat'):
        # read .mat
        gtfile_path = os.path.join(os.getcwd(), 'train', gtfile)
        gt = loadmat(gtfile_path)
        gt_mat = gt['groundTruth']

        for i in range(gt_mat.shape[1]):
            gt_label = gt_mat[0, i][0, 0][0]
            # save labels as uint16 png files
            img_name = gtfile.split('.')[0] + '_' + str(i) + '.png'
            save_path = os.path.join(gt_path, img_name)
            cv2.imwrite(save_path, np.uint16(gt_label))

print(gt_label.shape)
print(gt_label)

