import cv2
import numpy as np

image = cv2.imread("./img/000012.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      # 转为灰度图
blurred = cv2.GaussianBlur(gray, (15, 15), 0)
edged = cv2.Canny(blurred, 30, 120)         # 用Canny算子提取边缘
# cv2.imshow('1', edged)
# cv2.waitKey()
image2, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)      # 轮廓检测
# cv2.imshow('1', image)
# cv2.waitKey()
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)    # 绘制轮廓
# cv2.imshow('1', image)
# cv2.waitKey()
k = contours[0]  # 合并所有轮廓
for i in range(len(contours)):
    if i == 0:
        continue
    m = contours[i]
    k = np.vstack((k, m))

rect = cv2.minAreaRect(k)   # 检测轮廓最小外接矩形，得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
box = np.int0(cv2.boxPoints(rect))   # 获取最小外接矩形的4个顶点坐标
cv2.drawContours(image, [box], 0, (255, 0, 0), 2)     # 绘制轮廓最小外接矩形
cv2.imshow('1', image)
cv2.waitKey()