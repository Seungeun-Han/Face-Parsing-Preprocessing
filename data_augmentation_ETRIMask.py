import albumentations as A
import cv2
import os
import time

# Declare an augmentation pipeline
transform1 = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
])

transform2 = A.Compose([
    A.HorizontalFlip(p=1.0),
])

transform_option = 2  # 1: transform1, 2: transform2

im_path = os.path.join(r'D:\Dataset\ETRI_MaskDB\ETRI_MaskDB_473\labels_necklace_Last/')
image_list = os.listdir(im_path)

save_dir = r'D:\Dataset\ETRI_MaskDB\ETRI_MaskDB_473\labels_necklace_Last_aug/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for image_name in image_list:
    label = cv2.imread(im_path + image_name, cv2.IMREAD_GRAYSCALE)

    if transform_option == 1:
        # Augment an image
        # transformed = transform1(image=label)
        # transformed_image = transformed["image"]

        cv2.imwrite(save_dir + image_name[:-4] + "_RandomBrightnessContrast.png", label)

    elif transform_option == 2:
        # Augment an image
        transformed = transform2(image=label)
        transformed_image = transformed["image"]

        cv2.imwrite(save_dir + image_name[:-4] + "_HorizontalFlip.png", transformed_image)
