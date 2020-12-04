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


for file in files:  # 遍历文件夹
    image = Image.open(path + "/" + file)
    r_image = image.copy()
    time_start = time.time()
    list = []  # x1, y1, x2, y2
    yolo.detect_image(r_image, list)
    time_end = time.time()
    print('Time cost:', time_end-time_start)
    print(list)
    print(list[0][1])
    for i in range(len(list)):
        pos = (list[i][1], list[i][2], list[i][3], list[i][4])
        # image.show()
        # r_image.show()
        img_part = image.crop(pos)
        img_gray = cv2.cvtColor(np.asarray(img_part), cv2.COLOR_RGB2GRAY)
        lower = np.array([120])  # 过滤下限
        upper = np.array([256])  # 过滤下限
        mask = cv2.inRange(img_gray, lower, upper)
        img_gray[mask != 0] = [0]  # 将指定像素颜色修改为黑色
        cv2.imshow("OpenCV", img_gray)
        cv2.waitKey()
        # img_part.show()
    r_image.save(outpath + '/' + file + "result.jpg")
