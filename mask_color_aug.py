from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import time

import argparse
import cv2
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms


torch.multiprocessing.set_start_method("spawn", force=True)


IMAGE_DIR = r"D:\Dataset\CelebAMask-HQ-mask\CelebAMask-HQ-maskRendering_473\images"
image_list = os.listdir(IMAGE_DIR)
MASK_DIR = r"D:\Dataset\CelebAMask-HQ-mask\CelebAMask-HQ-maskRendering_473\mask_only"
mask_list = os.listdir(MASK_DIR)
SAVE_DIR = r"D:\Dataset\CelebAMask-HQ-mask\CelebAMask-HQ-maskRendering_473\images_mask_green"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


if __name__ == '__main__':

    for image_name in image_list:
        image = cv2.imread(os.path.join(IMAGE_DIR, image_name), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(MASK_DIR, image_name[:-4]+".png"), cv2.IMREAD_GRAYSCALE)

        index = np.where(mask == 255)
        # print(mask[index[0][0], index[1][0]])

        """
        green: (161, 184, 76)
        blue: (238, 202, 105)
        pink: (255, 217, 255)
        """
        for x, y in zip(index[0], index[1]):
            # mask_seg[x][y] = 0
            # print(image[x][y].shape)  # (3,)
            image[x][y] = (161, 184, 76)
            # print(x, y)



        cv2.imwrite(os.path.join(SAVE_DIR, image_name), image)
        #
        # cv2.imshow("img_raw", image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break





