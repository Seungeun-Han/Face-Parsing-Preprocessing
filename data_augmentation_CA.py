import albumentations as A
import cv2
import os
import time

# Declare an augmentation pipeline
transform1 = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=1.0),
])

transform2 = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
])

transform3 = A.Compose([
    A.HorizontalFlip(p=1.0),
])

transform4 = A.Compose([
    A.RandomCrop(width=256, height=256),
])

"""type = ['test', 'train', 'valid']
for t in type:
    print(t)
    im_path = os.path.join('../CelebA-HQ/' + t + '/images/')
    label_path = os.path.join('../CelebA-HQ/' + t + '/labels/')
    edge_path = os.path.join('../CelebA-HQ/' + t + '/edges/')
    cat_edge_path = os.path.join('../CelebA-HQ/' + t + '/cat_edges/')
    save_dir = '../CelebA-HQ_Augmentation/' + t
    image_list = os.listdir(im_path)

    folder_list = ['/images/', '/labels/', '/edges/', '/cat_edges/']
    for f in folder_list:
        if not os.path.exists(save_dir + f):
            os.makedirs(save_dir + f)

    for image_name in image_list:
        start = time.time()
        print(image_name)
        image = cv2.imread(im_path + image_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(label_path + image_name, cv2.IMREAD_GRAYSCALE)
        edge = cv2.imread(edge_path + image_name, cv2.IMREAD_GRAYSCALE)
        cat_edge = cv2.imread(cat_edge_path + image_name, cv2.IMREAD_GRAYSCALE)

        masks = [label, edge, cat_edge]

        # Augment an image
        transformed = transform1(image=image, masks=masks)
        transformed_image = transformed["image"]
        transformed_label = transformed['label']
        transformed_edge = transformed['edge']
        transformed_cat_edge = transformed['cat_edge']

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(save_dir + '/images/' + image_name, transformed_image)
        cv2.imwrite(save_dir + '/labels/' + image_name, transformed_label)
        cv2.imwrite(save_dir + '/edges/' + image_name, transformed_edge)
        cv2.imwrite(save_dir + '/cat_edges/' + image_name, transformed_cat_edge)"""

type = 'test'  # test, train, valid
transform_option = 1  # 1: transform1, 2: transform2, 3: transform3, 4: transform4

im_path = os.path.join('D:/Dataset/CelebAMask-HQ/CelebA-HQ/' + type + '/images/')
label_path = os.path.join('D:/Dataset/CelebAMask-HQ/CelebA-HQ/' + type + '/labels/')
edge_path = os.path.join('D:/Dataset/CelebAMask-HQ/CelebA-HQ/' + type + '/edges/')
cat_edge_path = os.path.join('D:/Dataset/CelebAMask-HQ/CelebA-HQ/' + type + '/cat_edges/')
save_dir = 'D:/Dataset/CelebAMask-HQ/CelebA-HQ_Aug_flip+crop_bgr/' + type
image_list = os.listdir(im_path)

folder_list = ['/images/', '/labels/', '/edges/', '/cat_edges/']
for f in folder_list:
    if not os.path.exists(save_dir + f):
        os.makedirs(save_dir + f)

for image_name in image_list:
    start = time.time()
    print(image_name)
    label_name = image_name[:-4] + ".png"
    image = cv2.imread(im_path + image_name, cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if transform_option == 1:
        label = cv2.imread(label_path + label_name, cv2.IMREAD_GRAYSCALE)
        edge = cv2.imread(edge_path + label_name, cv2.IMREAD_GRAYSCALE)
        cat_edge = cv2.imread(cat_edge_path + label_name, cv2.IMREAD_GRAYSCALE)

        masks = [label, edge, cat_edge]

        # Augment an image
        transformed = transform1(image=image, masks=masks)
        transformed_image = transformed["image"]
        transformed_masks = transformed["masks"]

        cv2.imwrite(save_dir + '/images/' + image_name, transformed_image)
        cv2.imwrite(save_dir + '/labels/' + label_name, transformed_masks[0])
        cv2.imwrite(save_dir + '/edges/' + label_name, transformed_masks[1])
        cv2.imwrite(save_dir + '/cat_edges/' + label_name, transformed_masks[2])

        print("time :", time.time() - start)

    elif transform_option == 2:
        # Augment an image
        transformed = transform2(image=image)
        transformed_image = transformed["image"]
        cv2.imwrite(save_dir + '/images/' + image_name, transformed_image)
        print("time :", time.time() - start)

    elif transform_option == 3:
        label = cv2.imread(label_path + label_name, cv2.IMREAD_GRAYSCALE)
        edge = cv2.imread(edge_path + label_name, cv2.IMREAD_GRAYSCALE)
        cat_edge = cv2.imread(cat_edge_path + label_name, cv2.IMREAD_GRAYSCALE)

        masks = [label, edge, cat_edge]

        # Augment an image
        transformed = transform3(image=image, masks=masks)
        transformed_image = transformed["image"]
        transformed_masks = transformed["masks"]

        cv2.imwrite(save_dir + '/images/' + image_name, transformed_image)
        cv2.imwrite(save_dir + '/labels/' + label_name, transformed_masks[0])
        cv2.imwrite(save_dir + '/edges/' + label_name, transformed_masks[1])
        cv2.imwrite(save_dir + '/cat_edges/' + label_name, transformed_masks[2])

        print("time :", time.time() - start)

    elif transform_option == 4:
        label = cv2.imread(label_path + label_name, cv2.IMREAD_GRAYSCALE)
        edge = cv2.imread(edge_path + label_name, cv2.IMREAD_GRAYSCALE)
        cat_edge = cv2.imread(cat_edge_path + label_name, cv2.IMREAD_GRAYSCALE)

        masks = [label, edge, cat_edge]

        # Augment an image
        transformed = transform4(image=image, masks=masks)
        transformed_image = transformed["image"]
        transformed_masks = transformed["masks"]

        cv2.imwrite(save_dir + '/images/' + image_name, transformed_image)
        cv2.imwrite(save_dir + '/labels/' + label_name, transformed_masks[0])
        cv2.imwrite(save_dir + '/edges/' + label_name, transformed_masks[1])
        cv2.imwrite(save_dir + '/cat_edges/' + label_name, transformed_masks[2])

        print("time :", time.time() - start)