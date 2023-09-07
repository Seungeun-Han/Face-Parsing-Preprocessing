import os

label_path = r"D:\Dataset\CelebAMask-HQ\CelebA-HQ_partial_256\labels"
label_list = os.listdir(label_path)

anno_path = r"D:\Dataset\CelebAMask-HQ\CelebAMask-HQ\CelebAMask-HQ-mask-anno_acc"
remove_list = [i[:5]+".png" for i in os.listdir(anno_path) if i[6:-4] == "hat"]
remove_list = [i for i in remove_list if i in label_list]
print(len(remove_list))
print(remove_list)

# save_dir = r"D:\Dataset\CelebAMask-HQ-hand\CelebAMask-HQ-mask-anno_acc/"
# saved_list = os.listdir(save_dir)

for subject in remove_list:
    """if len(subject) > 10:
        print(subject)
        os.remove(save_dir+subject)"""
    # if "hand" in subject: # HorizontalFlip RandomBrightnessContrast
    print(subject)
    os.remove(os.path.join(label_path, subject))