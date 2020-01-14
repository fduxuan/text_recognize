import cv2 as cv

#边缘检测

def main():


    src = cv.imread('test_img/car.jpg')
    img = cv.cvtColor(src, cv.COLOR_RGB2GRAY);
    #cv.imshow("原始图", src);

    # dst = cv.Sobel(img, cv.CV_64F, 1, 1, ksize=3)
    x = cv.Sobel(img, cv.CV_16S, 1, 0)
    y = cv.Sobel(img, cv.CV_16S, 0, 1)

    absX = cv.convertScaleAbs(x)  # 转回uint8
    absY = cv.convertScaleAbs(y)

    dst = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

    # cv.imshow("absX", absX)
    # cv.imshow("absY", absY)

    # cv.imshow("Result", dst)
    cv.imwrite("result.jpg",dst)



    cv.waitKey(0);


if __name__ == "__main__":
    main()