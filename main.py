from train import *
import torch.nn.functional as F


def find_max(k):
    m = 0
    index = 0
    for i in range(len(k)):
        if k[i] > m:
            m = k[i]
            index = i
    return index

def read_txt(file):
    path = []
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            path.append([words[0], int(words[1])])
    return path


def main():
    cnn = CNN()
    cnn.load_state_dict(torch.load("params.pkl"))
    path = read_txt("result/c.txt")
    write_txt = [""]*10000
    for i, row in path:
        img = Image.open(i)
        img = np.array(img)
        img = cv.resize(img, (28, 28), interpolation=cv.INTER_NEAREST)
        img = Image.fromarray(img, mode='L')
        tran = torchvision.transforms.ToTensor()
        img = tran(img)
        # print(img.shape)
        b_x = img.unsqueeze(0)
        b_x = Variable(b_x)
        out, x = cnn.forward(b_x)
        # print(out.shape, out)
        k = F.softmax(out[0], 0)
        char_k = find_max(k.detach().numpy())
        # print(chr(33 + char_k))
        write_txt[row] += chr(33+char_k)
    with open("result.txt", "w") as f:
        for line in write_txt:
            if line == "":
                break
            f.write(line+"\n")


def show():
    cnn = CNN()
    cnn.load_state_dict(torch.load("params.pkl"))
    img = Image.open("result/0_3.jpg")
    img = np.array(img)
    img = cv.resize(img, (28, 28), interpolation=cv.INTER_NEAREST)
    img = Image.fromarray(img, mode='L')
    tran = torchvision.transforms.ToTensor()
    img = tran(img)
    # print(img.shape)
    b_x = img.unsqueeze(0)
    b_x = Variable(b_x)
    out, x = cnn.forward(b_x)
    # print(out.shape, out)
    k = F.softmax(out[0], 0)
    char_k = find_max(k.detach().numpy())
    print(chr(33 + char_k))



if __name__ == "__main__":
    #main()
    show()