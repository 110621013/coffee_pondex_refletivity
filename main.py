import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def cal_normalize_distribution_y(img, intevel):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (1000,750))

    sum = 0
    y = np.zeros((intevel))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < intevel:
                y[img[i, j]] += 1
                sum += 1
    y /= sum
    return y
    # 先用入射角跟高度算出一張白紙應有的反射率
    # 再透過計算 粉反射率/白紙反射率的值
def plot_distribution(name, intevel):
    x = np.linspace(0, intevel-1, intevel)
    for img_name in img_list:
        img = cv2.imread(img_name)
        y = cal_normalize_distribution_y(img, intevel)
        plt.plot(x, y, label=img_name)
    plt.xlabel("reflection")
    plt.ylabel("number/sum = rate")
    plt.legend()
    plt.grid()
    plt.savefig(name)
    plt.close()

if __name__ == '__main__':
    img_list = [os.path.join('.', 'pic', img_name) for img_name in os.listdir(os.path.join('.', 'pic'))]
    print(img_list)
    
    plot_distribution('all#_distribution', intevel=256)
    plot_distribution('thr_distribution', intevel=150)
