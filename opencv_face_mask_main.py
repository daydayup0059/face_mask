#来自https://mp.weixin.qq.com/s/2t_UwsM4-qEF4rWSq1mAlw
import sys
import numpy as np
import cv2

modelFile = "opencv_face_detector_uint8.pb"
configFile = "opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
conf_threshold = 0.7   #置信阈值0.7


def cnt_area(cnt):
    '''求轮廓面积'''
    area = cv2.contourArea(cnt)
    return area


def If_Have_Mask(img):
    '''我使用的是HSV颜色空间的H值匹配来得到遮挡物（如：帽子，眼镜和口罩）的颜色。
    对于这部分，说说我的想法吧。我是将RGB颜色空间转化为HSV颜色空间，
    提取出待测图片的H值，因为不同颜色的H值不同，所以可以通过H值的确定来确定颜色'''
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_hsv_1 = np.array([0, 30, 30])  # 颜色范围低阈值
    upper_hsv_1 = np.array([40, 255, 255])  # 颜色范围高阈值
    lower_hsv_2 = np.array([140, 30, 30])  # 颜色范围低阈值
    upper_hsv_2 = np.array([180, 255, 255])  # 颜色范围高阈值
    #cv2.inRange函数设阈值，去除背景部分第一个参数：hsv指的是原图第二个参数：lower_red指的是图像中低于这个lower_red的值，图像值变为0第三个参数：upper_red指的是图像中高于这个upper_red的值，图像值变为0,而在lower_red～upper_red之间的值变成255
    mask1 = cv2.inRange(hsv_img, lower_hsv_1, upper_hsv_1)
    mask2 = cv2.inRange(hsv_img, lower_hsv_2, upper_hsv_2)
    #
    mask = mask1 + mask2
    #滤波
    mask = cv2.blur(mask, (3, 3))

    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    #只显示方框内的部分
    # mask = mask[80:280, 150:330]
    cv2.imshow("mask", mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) < 1:
        return "No Mask"
    contours=list(contours)
    contours.sort(key=cnt_area, reverse=True)
    print('cv2.contourArea(contours[0]):',cv2.contourArea(contours[0]))
    area = cv2.contourArea(contours[0])
    print('img.shape[0]:', img.shape[0])
    print('img.shape[1]:', img.shape[1])
    mask_rate = area / (img.shape[0] * img.shape[1])
    print('mask_rate:',mask_rate)
    if mask_rate < 0.65:
        return "Have Mask"
    else:
        return "No Mask"
def detectFaceOpenCVDnn(net, frame):
    #cv2.dnn.blobFromImage用法https://blog.csdn.net/weixin_43077628/article/details/114981613
    #blobFromImage# 对图像进行预处理，包括减均值，比例缩放，裁剪，交换通道等，返回一个4通道的blob(blob可以简单理解为一个N维的数组，用于神经网络的输入
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    print('blob.shape:', blob.shape)  # (1, 3, 300, 300)
    #cv2.imshow("blob", blob)
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    ## setInput设置模型的入参，将blob放入神经网络。计算输入的前向传递，将结果存储为 detections
    net.setInput(blob)
    #磁盘加载模型
    detections = net.forward()
    bboxes = []
    ret = 0
    print('detections.shape[2]:', detections.shape[2])  #200
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            #通过肤色轮廓面积与ROI面积比值判断是否有佩戴口罩，这里设置比值为0.65，上面三幅图的比例分别如下：
            ROI = frame[y1:y2, x1:x2].copy()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 框出人脸区域
            text = If_Have_Mask(img)
            print('口罩检测结果:', text)
            cv2.putText(frame, text, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                        cv2.LINE_AA)
    #If_Have_Mask(frame)
    return ret, frame


if __name__ == '__main__':
    img = cv2.imread("./4.jpeg")
    #_,= 就是拆解列表，元组，集合,可迭代对象的特例，要求其只能有一个元素
    _, result = detectFaceOpenCVDnn(net, img)
    cv2.imshow("face_detection", result)
    #print(If_Have_Mask(img))
    cv2.waitKey()
    cv2.destroyAllWindows()

