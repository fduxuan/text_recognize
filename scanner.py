import cv2 as cv
from math import *
import numpy as np
import time

def make_photo():
    """使用opencv拍照"""
    cap = cv.VideoCapture(0)  # 默认的摄像头
    while True:
        ret, frame = cap.read()
        if ret:
            cv.imshow("capture", frame)  # 弹窗口
            # 等待按键q操作关闭摄像头
            if cv.waitKey(1) & 0xFF == ord('q'):
                file_name = "aqian.jpeg"
                cv.imwrite(file_name, frame)
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()





def main():
    src = cv.imread('test_img/car.png')
    kkk = src
    #cv.imshow('input_image', src)
    # 二值化
    img = cv.cvtColor(src, cv.COLOR_RGB2GRAY);
    #cv.imwrite("ppt/grey.jpg", img)
    # 高斯滤波
    img_gaussian = cv.GaussianBlur(img, (5, 5), 0)
    #cv.imwrite("ppt/gauss.jpg", img_gaussian)
    # 获取矩形自定义核
    element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # 膨胀操作
    img_dilate = cv.dilate(img_gaussian, element)
    #cv.imwrite("ppt/dilate.jpg", img_dilate)
    # 边缘提取
    img_canny = cv.Canny(img_dilate, 30, 120, 3)
    #cv.imwrite("ppt/car.jpg", img_canny)


    # 找外轮廓
    contours, hierarchy = cv.findContours(img_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    max_area = 0
    index = 0

    for i in range(len(contours)):
        #cv.polylines(src, [contours[i]], True, (0, 255, 0), 2)
        tmparea = fabs(cv.contourArea(contours[i]));

        if tmparea > max_area:
            index = i
            max_area = tmparea
    #print(max_area)
    contour = contours[index]
    cv.polylines(src, [contour], True, (0, 255, 0), 2)
    # cv.imwrite("ppt/car.jpg", src)

    rect = cv.minAreaRect(contour)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    print(rect)
    h = rect[1][1]
    w = rect[1][0]
    # 计算变换后的面积
    area = rect[1][0]*rect[1][1]

    box = np.int0(cv.boxPoints(rect))  # 通过box得到矩形框
    print(box)
    # cv.drawContours(src, contour, -1, (0, 0, 255), 3)
    cv.polylines(src, [box], True, (0, 255, 0), 2)
    #cv.imwrite("ppt/min_area.jpg", src)


    approx = cv.approxPolyDP(contour, 50, True)

    # cv.polylines(src, [approx], True, (0, 255, 0), 2)
    # cv.imshow("dd", src)
    pts_before = []
    left = 0
    length = len(approx)

    print(length)

    for i in range(length):
        pts_before.append(approx[i][0])
    index = 0
    min_dist = 1000000
    for i in range(4):
        x = pts_before[i][0]
        y = pts_before[i][1]
        if sqrt(x*x + y*y) < min_dist:
            min_dist = sqrt(x*x+y*y)
            index = i

    pts = []
    print(pts_before)
    for i in range(4):
        pts.append(pts_before[-4+index+i])
    pts = np.float32(pts)

    pts_change = np.float32([[0,0],[0, w],[h,w],[h,0]])
    print(pts)
    print(pts_change)
    M = cv.getPerspectiveTransform(pts,pts_change)
    dst = cv.warpPerspective(kkk, M, (int(h), int(w)))
    #cv.imwrite("ppt/dst.jpg", dst)


    img = cv.cvtColor(dst, cv.COLOR_RGB2GRAY);
    blockSize = 25;
    constValue = 10;
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blockSize, constValue);

    cv.imshow("ddd", img)
    cv.imwrite("ppt/car.jpg", img)
    cv.waitKey(0)

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()