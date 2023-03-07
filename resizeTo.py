import cv2
import os

input_path = '../CelebA-HQ+Mask/train/'  # train / valid / test

save_dir = '../CelebA-HQ+Mask_224/train/'  # train / valid / test

target_size = (224, 224) # (473, 473)

type_list = os.listdir(input_path) # ['edges', 'images', 'cat_edges', 'labels']
for type in type_list:
    print(type)
    if type in ['images']:
        image_list = os.listdir(input_path+type)
        for images in image_list:
            print(images)
            image = cv2.imread(os.path.join(input_path, type, images))
            image = cv2.resize(image, target_size)
            save_path = os.path.join(save_dir, type)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            image = cv2.imwrite(os.path.join(save_path, images), image)
    else:
        image_list = os.listdir(input_path + type)
        for images in image_list:
            print(images)
            image = cv2.imread(os.path.join(input_path, type, images))
            image = cv2.resize(image, target_size, cv2.INTER_NEAREST)
            save_path = os.path.join(save_dir, type)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            image = cv2.imwrite(os.path.join(save_path, images), image)

