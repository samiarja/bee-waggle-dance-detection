# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
from os import listdir
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat

image_list = []
frameCounter = 0
annotation = loadmat('final_labels/20210803t1727d200m_cropped/20210803t1727d200m_ground_truth.mat')
ground_truth = annotation["td_gt"]
ground_truth_x = ground_truth["x"][0][0][0]
ground_truth_y = ground_truth["y"][0][0][0]
ground_truth_angle = ground_truth["angle"][0][0][0]

# images = [cv2.imread(file) for file in glob.glob('final_labels/test/*.png')]
folder_dir = "final_labels/test/"

fig = plt.figure(figsize=(18, 18))
for images in os.listdir(folder_dir):
    if (images.endswith(".png")):
        frameCounter = frameCounter + 1
        print(images)
        # print(idx)
        # image = 'final_labels/20210803t1727d200m_cropped/png/' + images + '.png'
        img = cv2.imread(folder_dir + images,0)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))

        rows, cols = img.shape
        crow,ccol = rows/2 , cols/2
        fshift[int(crow)-30:int(crow)+30, int(ccol)-30:int(ccol)+30] = 0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        plt.subplot(131),plt.imshow(img, cmap = 'gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
        plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
        plt.subplot(133),plt.imshow(img_back)
        plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.savefig('./frames/frames' + str(frameCounter) + '.png')
        # plt.show()

        # plt.subplot(121),plt.imshow(img, cmap = 'gray')
        # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
        # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        # plt.show()


    