import cv2
import os

input_path = r'D:\Dataset\CelebAMask-HQ\CelebAMask-HQ\CelebA-HQ-img'  # train, test, valid

save_dir = r'D:\Dataset\CelebAMask-HQ\CelebA-HQ_112/images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
"""# type_list = os.listdir(input_path) # ['edges', 'images', 'cat_edges', 'labels']
# type_list = ['edges', 'cat_edges', 'labels']
# type_list = ['images']
type_list = ['edges']
for type in type_list:
    print(type)
    #image_path = input_path+"/"+type
    image_list = os.listdir(input_path)
    for images in image_list:
        print(images)
        image = cv2.imread(os.path.join(input_path, images), cv2.IMREAD_GRAYSCALE)  # type, cv2.IMREAD_COLOR
        # image = cv2.resize(image, (256, 256))
        image = cv2.resize(image, (256, 256) , cv2.INTER_NEAREST)
        save_path = os.path.join(save_dir, type) # , type
        if not os.path.exists(save_path):
          os.makedirs(save_path)
        # images
        # cv2.imwrite(os.path.join(save_path, images[:-4].zfill(5)+'.jpg'), image)

        # labels
        cv2.imwrite(os.path.join(save_path, images), image)

        # image = cv2.imwrite(os.path.join(save_dir, images), image)"""

image_list = os.listdir(input_path)
for images in image_list:
    print(images)
    image = cv2.imread(os.path.join(input_path, images), cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR), cv2.IMREAD_GRAYSCALE
    image = cv2.resize(image, (112, 112)) # , cv2.INTER_NEAREST
    cv2.imwrite(os.path.join(save_dir, images[:-4].zfill(5))+".jpg", image)