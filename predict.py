#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from yolo import YOLO
from PIL import Image
import time
import os
import cv2
import numpy as np

path = "/home/ye/img/rgb"  # 文件夹目录
outpath = "/home/ye/img/mask"
files = os.listdir(path)

yolo = YOLO()


def edge(img, post, name):
    crop = img[post[1]:post[3], post[0]:post[2]]
    # cv2.imwrite(outpath + '/' + name + "-crop.jpg", crop)
    # cv2.imshow('1', crop)
    # cv2.waitKey()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)      # 转为灰度图
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)       # 高斯模糊
    # cv2.imwrite(outpath + '/' + name + "-blur.jpg", blurred)
    edged = cv2.Canny(blurred, 30, 90)
    cv2.imwrite(outpath + '/' + name + "-edge.jpg", edged)
    # 用Canny算子提取边缘
    kernel = np.ones((17, 17), np.uint8)
    closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)    # 膨胀腐蚀处理
    cv2.imwrite(outpath + '/' + name + "-closing.jpg", closing)
    # cv2.imshow('1', closing)
    # cv2.waitKey()
    image2, contours, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # opencv 4.0 轮廓检测 3.x版本最前面需要加一个image2参数
    # cv2.imshow('1', image)
    # cv2.waitKey()
    k = []  # 合并所有轮廓
    for i in range(len(contours)):
        if len(contours[i]) > len(k):
            k = contours[i]
    h, w, c = crop.shape
    finaledge = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(finaledge, k, -1, (255, 255, 255), 2)   # 绘制轮廓
    cv2.imwrite(outpath + '/' + name + "-finaledge.jpg", finaledge)
    crop2 = crop.copy()
    cv2.drawContours(crop2, k, -1, (0, 255, 0), 2)    # 在原图上绘制轮廓
    # cv2.imwrite(outpath + '/' + name + "-contours.jpg", crop2)

    l = k.copy()   # 将轮廓坐标还原到原图
    for i in range(len(k)):
        l[i] = k[i] + [post[0], post[1]]

    h, w, c = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)  # 创建空白掩膜图
    cv2.fillPoly(mask, [l], (255, 255, 255))  # 利用轮廓生成掩膜
    cv2.imwrite(outpath + '/' + name + "-mask.jpg", mask)
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    result = cv2.bitwise_and(img, img, mask=mask)  # 利用掩膜提取原图零件
    # cv2.imshow("result", result)
    # cv2.waitKey(0)
    cv2.imwrite(outpath + '/' + name + "-cut.jpg", result)

    rect = cv2.minAreaRect(k)   # 检测轮廓最小外接矩形，得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    box = np.int0(cv2.boxPoints(rect))   # 获取最小外接矩形的4个顶点坐标
    for i in range(4):
        box[i] = box[i] + [post[0], post[1]]
    # cv2.drawContours(img, [box], 0, (255, 0, 0), 2)     # 绘制轮廓最小外接矩形
    # crop3 = img[box[1][1]:box[2][2], box[0]:box[2]]
    # cv2.imwrite(outpath + '/' + file + "box.jpg", crop3)
    # cv2.imshow('1', img)
    # cv2.waitKey()
    return box


for file in files:  # 遍历文件夹
    image = Image.open(path + "/" + file)
    r_image = image.crop((640, 300, 1280, 780))
    l_image = r_image.copy()
    # r_image.show()
    time_start = time.time()
    list = []  # x1, y1, x2, y2
    allbox = []
    yolo.detect_image(l_image, list)
    time_end = time.time()
    imagecv = cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR)
    print('Time cost:', time_end-time_start)
    print(list)
    for i in range(len(list)):  # 逐个零件进行检测轮廓
        pos = (list[i][1], list[i][2], list[i][3], list[i][4])
        # for j in range(len(list)):
        #     if j is not i:
        #         allpos.append((list[j][1], list[j][2], list[j][3], list[j][4]))
        # image.show()
        # r_image.show()
        allbox.append(edge(imagecv, pos, list[i][0]))  # 将坐标统一进一个集合
    
    for i in range(len(allbox)):
        print(allbox[i])
        cv2.drawContours(imagecv, [allbox[i]], 0, (255, 0, 0), 3)     # 绘制轮廓最小外接矩形
        a = tuple(allbox[i][1])
        
        cv2.putText(imagecv, list[i][0], a, cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)
        cv2.imshow('1', imagecv)
        cv2.waitKey(1000)

        
        # img_part.show()
    
    cv2.imwrite(outpath + '/' + file + "-result.png", imagecv)
    r_image.save(outpath + '/' + file + "-cut.png")
    l_image.save(outpath + '/' + file + "-result2.png")
