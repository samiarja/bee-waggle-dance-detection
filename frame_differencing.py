# import the opencv module
import cv2
import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

tau = 36
kernelDepth = 18
kernelArray = np.arange(1,kernelDepth+1)
path = "final_labels/20210803t1727d200m_cropped/png/"
images_path = glob.glob(path + "*png")

for idx in tqdm(range(len(images_path))):
    img_1 = cv2.imread(images_path[idx], 0)
    img_2 = cv2.imread(images_path[idx+1], 0)
    # find difference between two frames
    diff = cv2.absdiff(img_1, img_2)
    # to convert the frame to grayscale
    diff_gray = diff #cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # apply some blur to smoothen the frame
    diff_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
    
    #### 1D Conv #####
    waggleKernel_1 = np.exp(-kernelArray/tau)*np.sin(5.75/(2*np.pi)*(kernelArray+5.2))
    waggleKernel_2 = np.exp(-kernelArray/tau)*np.sin(5/(2*np.pi)*(kernelArray+5.2))
    waggleKernel_3 = np.exp(-kernelArray/tau)*np.sin(4/(2*np.pi)*(kernelArray+6.5))
    
    waggleMap1 = np.reshape(np.convolve(diff_blur.flatten(),waggleKernel_1,'same'),(diff_blur.shape[0],diff_blur.shape[1]))
    waggleMap2 = np.reshape(np.convolve(diff_blur.flatten(),waggleKernel_2,'same'),(diff_blur.shape[0],diff_blur.shape[1]))
    waggleMap3 = np.reshape(np.convolve(diff_blur.flatten(),waggleKernel_3,'same'),(diff_blur.shape[0],diff_blur.shape[1]))
    
    waggleMap_max = waggleMap2
    # to get the binary image
    _, thresh_bin = cv2.threshold(diff_blur, 50, 255, cv2.THRESH_BINARY)
    # to find contours
    contours, hierarchy = cv2.findContours(thresh_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # to draw the bounding box when the motion is detected
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 50:
            cv2.rectangle(img_1, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.drawContours(img_1, contours, -1, (0, 255, 0), 2)
    
    cv2.imshow("Detecting Motion...", diff)
    cv2.waitKey(100)