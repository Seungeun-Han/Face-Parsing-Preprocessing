import os
import cv2

# im_path = 'D:/Dataset/CelebAMask-HQ/CelebA-HQ_473/images'
lb_path = "D:/Dataset/CelebAMask-HQ/CelebA-HQ_473/labels_necklace_Last"
# edge_path = "D:/Dataset/CelebAMask-HQ/CelebA-HQ_473/edges"
parsing_anno_path = os.path.join('D:/Dataset/CelebAMask-HQ/CelebAMask-HQ/CelebAMask-HQ-mask-anno_acc/')
annotation_list = os.listdir(parsing_anno_path)

# im_save_dir = "D:/Dataset/CelebAMask-HQ/earring_CelebAMask-HQ/images"
# if not os.path.exists(im_save_dir):
#     os.makedirs(im_save_dir)
lb_save_dir = "D:/Dataset/CelebAMask-HQ/necklace_CelebAMask-HQ/necklaceLast_labels"
if not os.path.exists(lb_save_dir):
    os.makedirs(lb_save_dir)
# edge_save_dir = "D:/Dataset/CelebAMask-HQ/earring_CelebAMask-HQ/edges"
# if not os.path.exists(edge_save_dir):
#     os.makedirs(edge_save_dir)

for i in annotation_list:
    if "neck_l" in i:  # neck_l
        print(i)
        # img = cv2.imread(os.path.join(im_path, i[:5]+".jpg"), cv2.IMREAD_COLOR)
        lb = cv2.imread(os.path.join(lb_path, i[:5] + ".png"), cv2.IMREAD_GRAYSCALE)
        # edge = cv2.imread(os.path.join(edge_path, i[:5] + ".png"), cv2.IMREAD_GRAYSCALE)

        # cv2.imwrite(os.path.join(im_save_dir, i[:5]+".jpg"), img)
        cv2.imwrite(os.path.join(lb_save_dir, i[:5] + ".png"), lb)
        # cv2.imwrite(os.path.join(edge_save_dir, i[:5] + ".png"), edge)
        """cv2.imshow("parsing_anno", parsing_anno)
        cv2.waitKey(10)"""

