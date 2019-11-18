import cv2 as cv
import numpy as np

def split_c(img):
    pos_row = find_pos(img, 0)
    pos_col = find_pos(img, 1)

    split_row = split_pos(pos_row, 0, 1)
    split_col = split_pos(pos_col, 0, 1)

    return [split_row[0][0], split_row[-1][1]], [split_col[0][0], split_col[-1][1]]


# min_thresh: 阈值  min_range: 跨度
def split_pos(pos, min_thresh, min_range):
    split_res = []
    length = 0
    for i in range(len(pos)):
        if pos[i] > min_thresh:
            length += 1
        else:
            if length > min_range:
                split_res.append([i - length, i])
            length = 0
    if length != 0:
        split_res.append([len(pos) - length, len(pos)])
    return split_res


# mode:0 row, mode:1 col
def find_pos(img, mode=0):
    # print(img.shape)
    shape = img.shape
    w = shape[1]
    h = shape[0]
    pos_row = []
    pos_col = []
    img2 = np.zeros(shape,dtype=np.uint8)
    # print(img2.shape)
    if mode == 0:
        for i in range(h):
            r = 0
            for j in range(w):
                if img[i][j] < 2:
                    r += 1
            pos_row.append(r)
        for i in range(h):
            for j in range(pos_row[i]):
                img2[i][j] = 255
        cv.imwrite("ppt/row_split.jpg", img2)
        return pos_row
    if mode == 1:
        for j in range(w):
            r = 0
            for i in range(h):
                if img[i][j] < 2:
                    r += 1
            pos_col.append(r)
        return pos_col





def show():
    src = cv.imread('res.jpg', 0)
    pos_row = find_pos(src, 0)

    split_row = split_pos(pos_row, 6, 10)
    x = split_row[0]
    src_row = src[x[0]: x[1] + 1]
    cv.line(src, (0, x[0]), (src.shape[1], x[0]), (0, 0, 255), 1, 1)
    cv.imwrite("ppt/src_one_row.jpg", src)
    cv.imshow("ppt/row.jpg", src_row)
    cv.waitKey(0)

def main():
    src = cv.imread('res.jpg', 0)
    # cv.imshow("dddd", src)
    pos_row = find_pos(src, 0)

    split_row = split_pos(pos_row, 6, 10)
    write_txt = ""
    for j in range(len(split_row)):
        x = split_row[j]
        # cv.line(src, (0, x[0]),(src.shape[1], x[0]) ,(0,0,255), 1, 1)
        # cv.imshow("dddd", src)

        src_row = src[x[0]: x[1] + 1]
        pos_col = find_pos(src_row, 1)
        split_col = split_pos(pos_col, 1, 10)
        for i in range(len(split_col)):
            img = src_row[:, split_col[i][0]: split_col[i][1]]
            row, col = split_c(img)
            file_name = "result/"+ str(j)+"_" + str(i) + ".jpg"
            cv.imwrite(file_name, img[row[0]:row[1], col[0]:col[1]])
            write_txt += file_name + " "+str(j) +"\n"
    with open("result/c.txt", "w") as f:
        f.write(write_txt)
    # cv.imshow("Dddd", src_row)
    cv.waitKey(0)
    cv.destroyAllWindows()









if __name__ == "__main__":
    #main()
    show()