import cv2
import os

input_path = 'D:/Dataset/CelebAMask-HQ-mask/seg_1024png'
image_list = os.listdir(input_path)
image_list = [i for i in image_list if "kf94" in i]

save_dir = 'D:/Dataset/CelebAMask-HQ-mask/seg_1024png'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


for images in image_list:
    print(images)
    image = cv2.imread(os.path.join(input_path, images), cv2.IMREAD_GRAYSCALE)
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    image = cv2.imwrite(os.path.join(save_dir, images), image)

