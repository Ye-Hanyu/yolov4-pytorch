import cv2
import numpy as np

image = cv2.imread("./img/3.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      # 转为灰度图
blurred = cv2.GaussianBlur(gray, (15, 15), 0)
edged = cv2.Canny(blurred, 30, 60)
cv2.imwrite('./result/edged.jpg', edged)         # 用Canny算子提取边缘
kernel = np.ones((21,21),np.uint8)
closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('./result/closing.jpg', closing) 

rect = cv2.minAreaRect(k)   # 检测轮廓最小外接矩形，得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
box = np.int0(cv2.boxPoints(rect))   # 获取最小外接矩形的4个顶点坐标
cv2.drawContours(image, [box], 0, (255, 0, 0), 2)     # 绘制轮廓最小外接矩形
cv2.imshow('1', image)
cv2.waitKey()