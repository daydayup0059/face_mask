import sys
import numpy as np
import cv2

img = cv2.imread("./5.jpeg")
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_hsv_1 = np.array([0, 30, 30])  # 颜色范围低阈值
upper_hsv_1 = np.array([40, 255, 255])  # 颜色范围高阈值
lower_hsv_2 = np.array([140, 30, 30])  # 颜色范围低阈值
upper_hsv_2 = np.array([180, 255, 255])  # 颜色范围高阈值
    #cv2.inRange函数设阈值，去除背景部分第一个参数：hsv指的是原图第二个参数：lower_red指的是图像中低于这个lower_red的值，图像值变为0第三个参数：upper_red指的是图像中高于这个upper_red的值，图像值变为0,而在lower_red～upper_red之间的值变成255
mask1 = cv2.inRange(hsv_img, lower_hsv_1, upper_hsv_1)
mask2 = cv2.inRange(hsv_img, lower_hsv_2, upper_hsv_2)
cv2.imshow("mask1", mask1)
cv2.imshow("mask2", mask2)
#
mask = mask1 + mask2
#滤波
mask_all = cv2.blur(mask, (3, 3))
cv2.imshow("mask_all", mask_all)
mask_all = mask_all[80:280, 150:330]
cv2.imshow("mask", mask_all)
#_,= 就是拆解列表，元组，集合,可迭代对象的特例，要求其只能有一个元素
 #_, result = detectFaceOpenCVDnn(net, img)
#cv2.imshow("face_detection", result)
#print(If_Have_Mask(img))
cv2.waitKey()
cv2.destroyAllWindows()