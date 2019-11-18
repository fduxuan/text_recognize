import argparse
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from split import *

# 随机字母:
def rndChar():
    return chr(random.randint(65, 90))
# 随机颜色1:
def rndColor():
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))
# 随机颜色2:
def rndColor2():
    return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))

def rotation():
    return random.randint(-15,15)



def gen():
    string = [chr(i) for i in range(33, 127)]
    num = len(string)
    width = 64
    height = 80
    write_txt = ""
    for k in range(num):
        # 每个字母循环100次
        for i in range(20):
            image = Image.new('RGB', (width, height), (255, 255, 255))
            image = image.convert('RGBA')
            # 创建Font对象:

            font = ImageFont.truetype('font_test/Geneva.dfont', 60)
            # 创建Draw对象:
            draw = ImageDraw.Draw(image)
            # 输出文字:
            draw.text((0, 0), string[k], font=font, fill=(0, 0, 0))
            image = image.rotate(rotation(), expand=1)
            fff = Image.new('RGBA', image.size, (255,) * 4)
            # 使用alpha层的rot作为掩码创建一个复合图像
            out = Image.composite(image, fff, image)

            image = np.array(out)
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY);
            row, col = split_c(image)
            new_img = image[row[0] - 1: row[1] + 1, col[0] - 1: col[1] + 1]
            filename = "character_test/" + str(k) + "_"+str(i)+  ".jpg"
            write_txt+=filename+" "+str(k)+"\n"
            cv.imwrite(filename, new_img)
    # with open("dataset/test.txt", "w") as f:
    #     f.write(write_txt)







if __name__ == "__main__":
    gen()