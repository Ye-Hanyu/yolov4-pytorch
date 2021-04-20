import cv2
import numpy as np

image = cv2.imread("./img/3.jpg")
image2 = cv2.imread("./result/3.jpgresult.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      # 转为灰度图
blurred = cv2.GaussianBlur(gray, (9, 9), 0)
edged = cv2.Canny(blurred, 30, 90)
cv2.imwrite('./result/edged.jpg', edged)         # 用Canny算子提取边缘
kernel = np.ones((17, 17), np.uint8)
closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('./result/closing.jpg', closing)
image3, contours, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for i in range(len(contours)):
    if len(contours[i]) <= 50:
        continue
    hull_points = cv2.convexHull(contours[i])
    cv2.polylines(image, [hull_points], True, (0, 255, 0), 2)
    cv2.imshow('1', image)
    cv2.waitKey(100)
cv2.imwrite('./result/hull.jpg', image)
for i in range(len(contours)):
    if len(contours[i]) <= 50:
        continue
    hull_points = cv2.convexHull(contours[i])
    cv2.polylines(image2, [hull_points], True, (0, 255, 0), 2)
    cv2.imshow('1', image2)
    cv2.waitKey(100)
cv2.imwrite('./result/hull2.jpg', image2)
# rect = cv2.minAreaRect(k)   # 检测轮廓最小外接矩形，得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
# box = np.int0(cv2.boxPoints(rect))   # 获取最小外接矩形的4个顶点坐标
# cv2.drawContours(image, [box], 0, (255, 0, 0), 2)     # 绘制轮廓最小外接矩形
# cv2.imshow('1', image)
# cv2.waitKey()