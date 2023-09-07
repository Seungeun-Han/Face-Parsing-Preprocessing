import time
import torch
import os, numpy as np
import os.path as osp
import shutil
import cv2

def generate_edge(label_dir, edge_dir, size = 473):
    #  Generate edges for labels in label_dir and save them to edge_dir

    print('Generating edges from {} to {}'.format(label_dir, edge_dir))
    if not os.path.exists(edge_dir):
        # shutil.rmtree(edge_dir)
        os.makedirs(edge_dir)
    ll = os.listdir(label_dir)
    saved_list = os.listdir(edge_dir)
    ll = [i for i in ll if i[:-4] + ".png" not in saved_list]

    ll = [x for x in ll if int(x[:5]) < 10000 and int(x[:5]) >= 5000]  #  and int(x[:-4]) >= 10000)
    print(len(ll))

    for filename in ll:
        print(filename)
        label = cv2.imread(osp.join(label_dir, filename), cv2.IMREAD_GRAYSCALE)
        # label = cv2.resize(label, (size, size))
        edge = np.zeros_like(label)
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                flag = 1
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    x = i + dx
                    y = j + dy
                    if 0 <= x < label.shape[0] and 0 <= y < label.shape[1]:
                        if label[i,j] != label[x,y]:
                            edge[i,j] = 1
        cv2.imwrite(osp.join(edge_dir, filename), edge)

generate_edge(r"D:\Dataset\CelebAMask-HQ-hand\CelebAMask-HQ-hand_473\labels/", r"D:\Dataset\CelebAMask-HQ-hand\CelebAMask-HQ-hand_473\edges/", size=256)
