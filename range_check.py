import os
import cv2
import numpy as np

upper_path = "D:/Dataset/CelebAMask-HQ/CelebA-HQ_Aug_flip+crop_bgr_473/"

img_path = upper_path + "valid/cat_edges/"
img_list = os.listdir(img_path)

txt_path = upper_path + "train_cat_edges_error_list.txt"
error_list = open(txt_path, 'w')

"""if "Mask" in img_path:
    if "cat_edges" in img_path or "labels" in img_path:
        print("Check if it is in the range of 0 to 19")
        for img in img_list:
            image = cv2.imread(img_path+img, cv2.IMREAD_GRAYSCALE)
            image = np.reshape(image, (1, -1))
            pixel = [i for i in range(len(image[0])) if image[0][i] > 19]
            if len(pixel) > 0:
                print(img)
                error_list.write("{0}\n".format(img))
    elif "edges" in img_path:
        print("Check if it is in the range of 0 to 1")
        for img in img_list:
            image = cv2.imread(img_path+img, cv2.IMREAD_GRAYSCALE)
            image = np.reshape(image, (1, -1))
            pixel = [i for i in range(len(image[0])) if image[0][i] != 0 and image[0][i] != 255]
            if len(pixel) > 0:
                print(img)
                error_list.write("{0}\n".format(img))
else:"""
if "cat_edges" in img_path or "labels" in img_path:
    print("Check if it is in the range of 0 to 18")
    for img in img_list:
        image = cv2.imread(img_path+img, cv2.IMREAD_GRAYSCALE)
        image = np.reshape(image, (1, -1))
        pixel = [i for i in range(len(image[0])) if image[0][i] > 18]
        if len(pixel) > 0:
            print(img)
            error_list.write("{0}\n".format(img))
elif "edges" in img_path:
    print("Check if it is in the range of 0 or 255")
    for img in img_list:
        image = cv2.imread(img_path+img, cv2.IMREAD_GRAYSCALE)
        image = np.reshape(image, (1, -1))
        pixel = [i for i in range(len(image[0])) if image[0][i] != 0 and image[0][i] != 255]
        if len(pixel) > 0:
            print(img)
            error_list.write("{0}\n".format(img))

error_list.close()



