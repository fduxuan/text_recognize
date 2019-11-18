import os
from skimage import io
import torchvision.datasets.mnist as mnist
import torchvision
import numpy



def main():
    root = "mnist/MNIST/raw/"

    train_set = (
        mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
    )

    test_set = (
        mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
    )

    print("train set:", train_set[0].size())
    print("test set:", test_set[0].size())

    def convert_to_img(train=True):
        if (train):
            f = open(root + 'train.txt', 'w')
            data_path = root + '/train/'
            if (not os.path.exists(data_path)):
                os.makedirs(data_path)
            for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
                img_path = data_path + str(i) + '.jpg'
                io.imsave(img_path, img.numpy())
                int_label = str(label).replace('tensor(', '')
                int_label = int_label.replace(')', '')
                f.write(img_path + ' ' + str(int_label) + '\n')
            f.close()
        else:
            f = open(root + 'test.txt', 'w')
            data_path = root + '/test/'
            if (not os.path.exists(data_path)):
                os.makedirs(data_path)
            for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
                img_path = data_path + str(i) + '.jpg'
                io.imsave(img_path, img.numpy())
                int_label = str(label).replace('tensor(', '')
                int_label = int_label.replace(')', '')
                f.write(img_path + ' ' + str(int_label) + '\n')
            f.close()

    convert_to_img(True)
    convert_to_img(False)


def down():
    DOWNLOAD_MNIST = False

    # Mnist digits dataset
    if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
        # not mnist dir or mnist is empyt dir
        DOWNLOAD_MNIST = True

    train_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,  # this is training data
        transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        download=DOWNLOAD_MNIST,
    )

if __name__ == "__main__":
    main()