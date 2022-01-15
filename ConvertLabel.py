import cv2
import os
import numpy as np
from FranjiFilter import GetVesshancementMask

IMAGE_SIZE = [672, 752]

# input: 
# 	1: CT img
# 	2: Lung label
# 	3: Lesion label
# output:
# 	refined multi-category label
#	1: r1
#   2: r2
#   3: multi-category label (g=3, interval=255//3=85)  
#		gray-scale range: [0, 85, 170, 255]
#		
mark_bgr = [ 
			[201, 230,  253], # 类别 1      粉色
			[0, 128, 255], # 类别 2         橘色
			[0, 0, 255] # 类别 3           红色
		   ]

def getRefinedLabel(img, lung_label, lesion_label):

	img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)
	lung_label = cv2.resize(lung_label, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)
	lesion_label = cv2.resize(lesion_label, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)

	_, lung_label = cv2.threshold(lung_label, 250, 1, cv2.THRESH_BINARY)
	_, lesion_label = cv2.threshold(lesion_label, 250, 1, cv2.THRESH_BINARY)

	healthy_mask = cv2.subtract(lung_label, lesion_label)
	healthy_set = img[healthy_mask>0]
	healthy_pixel_mean = int(np.mean(healthy_set))

	lesion_img = cv2.multiply(img, lesion_label)

	# remove the lung parenchyma pixels
	_, r1 = cv2.threshold(lesion_img, healthy_pixel_mean, 1, cv2.THRESH_BINARY)

	# remove the vesll pixels
	lesion_img_new = cv2.multiply(lesion_img, r1)
	vess_mask = GetVesshancementMask(lesion_img_new)
	r2 = cv2.subtract(r1, vess_mask)

	r2_img = cv2.multiply(img, r2)

	ranges = [0, 85, 170, 255]
	r2_m = np.zeros([r2.shape[0], r2.shape[1], 3], dtype="uint8")
	for r in range(r2.shape[0]):
		for c in range(r2.shape[1]):
			if r2_img[r][c] == 0:
				r2_m[r][c] = [0,0,0]
			elif r2_img[r][c] < ranges[1]:
				r2_m[r][c] = mark_bgr[0]
			elif r2_img[r][c] < ranges[2]:
				r2_m[r][c] = mark_bgr[1]
			elif r2_img[r][c] < ranges[3]:
				r2_m[r][c] = mark_bgr[2]

	return r1*255, r2*255, r2_m


if __name__ == '__main__':
    img_name = '015_axial0079.png'
    lung_label_name = '015_axial0079_lung.png'
    lesion_label_name = '015_axial0079_lesion.png'

    img = cv2.imread(img_name, 0)
    lung_label = cv2.imread(lung_label_name, 0)
    lesion_label = cv2.imread(lesion_label_name, 0)

    r1, r2, r2_m = getRefinedLabel(img, lung_label, lesion_label)

    cv2.imwrite('r1.jpg', r1)
    cv2.imwrite('r2.jpg', r2)
    cv2.imwrite('r2_m.jpg', r2_m)






