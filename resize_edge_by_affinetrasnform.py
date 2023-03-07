"""
473
왼눈[183, 180], 오눈[296, 180], 오입[289, 300]

256
average_left_eye_rocation: (98, 122)
average_right_eye_rocation: (160, 122)
average_nose_rocation: (127.96376666666667, 161.1341)
average_left_mouth_rocation: (102.08783333333334, 187.6695)
average_right_mouth_rocation: (157, 188)
"""
import cv2
import os
import numpy as np

image_path = 'D:/Dataset/ETRI_MaskDB/labels'  # train, test, valid
image_list = os.listdir(image_path)
save_dir = 'D:/Dataset/ETRI_MaskDB/mask_dataset_th3_px1_256/labels'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for images in image_list:
    print(images)
    image = cv2.imread(os.path.join(image_path, images), cv2.IMREAD_GRAYSCALE)

    # resize
    inPoint = np.array([[0, 0], [0, 472], [472, 472]], dtype=np.float32)
    outPoint = np.array([[0, 0], [0, 255], [255, 255]], dtype=np.float32)
    affine_matrix = cv2.getAffineTransform(inPoint, outPoint)
    face_align = cv2.warpAffine(image, affine_matrix, (256, 256))

    cv2.imwrite(os.path.join(save_dir, images), face_align)


