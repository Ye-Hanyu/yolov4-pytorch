#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from yolo import YOLO
from PIL import Image
import time
import os
import cv2
import numpy as np

path = "./img"  # 文件夹目录
outpath = "./result"
files = os.listdir(path)

yolo = YOLO()


def edge(img, post):
    crop = img[post[1]:post[3], post[0]:post[2]]
    # cv2.imshow('1', edged)
    # cv2.waitKey()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)      # 转为灰度图
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    edged = cv2.Canny(blurred, 30, 90)
    # cv2.imwrite(outpath + '/' + file + "edged.jpg", edged)         # 用Canny算子提取边缘
    kernel = np.ones((17, 17),np.uint8)
    closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite(outpath + '/' + file + "closing.jpg", closing) 
    # cv2.imshow('1', closing)
    # cv2.waitKey()
    image2, contours, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)      # 轮廓检测
    # cv2.imshow('1', image)
    # cv2.waitKey()
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 2)    # 绘制轮廓
    # cv2.imshow('1', image)
    # cv2.waitKey()
    k = []  # 合并所有轮廓
    for i in range(len(contours)):
        if len(contours[i]) > len(k):
            k = contours[i]

    rect = cv2.minAreaRect(k)   # 检测轮廓最小外接矩形，得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    box = np.int0(cv2.boxPoints(rect))   # 获取最小外接矩形的4个顶点坐标
    for i in range(4):
        box[i] = box[i] + [post[0], post[1]]
    # cv2.drawContours(img, [box], 0, (255, 0, 0), 2)     # 绘制轮廓最小外接矩形
    # cv2.imshow('1', img)
    # cv2.waitKey()
    return box


for file in files:  # 遍历文件夹
    image = Image.open(path + "/" + file)
    r_image = image.copy()
    time_start = time.time()
    list = []  # x1, y1, x2, y2
    allbox = []
    yolo.detect_image(r_image, list)
    time_end = time.time()
    imagecv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    print('Time cost:', time_end-time_start)
    print(list)   
    for i in range(len(list)):
        pos = (list[i][1], list[i][2], list[i][3], list[i][4])
        # for j in range(len(list)):
        #     if j is not i:
        #         allpos.append((list[j][1], list[j][2], list[j][3], list[j][4]))
        # image.show()
        # r_image.show()
        allbox.append(edge(imagecv, pos))
    for i in range(len(allbox)):
        print(allbox[i])
        cv2.drawContours(imagecv, [allbox[i]], 0, (255, 0, 0), 3)     # 绘制轮廓最小外接矩形
        a = tuple(allbox[i][1])
        
        cv2.putText(imagecv, list[i][0], a, cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)
        cv2.imshow('1', imagecv)
        cv2.waitKey(1000)

        
        # img_part.show()
    
    cv2.imwrite(outpath + '/' + file + "result.jpg", imagecv)
    r_image.save(outpath + '/' + file + "result2.jpg")
