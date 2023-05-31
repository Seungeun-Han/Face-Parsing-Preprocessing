import os

upper_path = os.path.join('D:/Dataset/CelebAMask-HQ/necklace_CelebAMask-HQ_473_HoriFlip')
# type_list = os.listdir(upper_path) # ['edges', 'images', 'cat_edges', 'labels']
type_list = ['edges', 'images', 'labels', 'labels_necklace_Last']

for type in type_list:
    image_path = os.path.join(upper_path, type)
    image_list = os.listdir(image_path)
    for image in image_list:
        # print(os.path.join(image_path, image))
        # print(os.path.join(image_path, image[:-4] + "_flip+crop256" + image[-4:]))
        # print(image[:-4] + "_kf94" + image[-4:])
        print(image[:-4] + "_HoriFlip" + image[-4:])
        os.renames(os.path.join(image_path, image), os.path.join(image_path, image[:-4] + "_HoriFlip" + image[-4:]))
    print("end")

"""image_list = os.listdir(upper_path)
for image in image_list:
    if "kf94" not in image:
        # print(image, image[:-4] + "_kf94" + image[-4:])
        os.renames(os.path.join(upper_path, image), os.path.join(upper_path, image[:-4] + "_kf94" + image[-4:]))
print("end")"""