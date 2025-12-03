import numpy as np
import cv2
import matplotlib.pyplot as plt


# # 讀取影像並轉灰階
img = cv2.imread('lll.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 轉換前，都先將圖片轉換成灰階色彩

# 計算直方圖
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

#1: optimal threshold value=127

k1 = 114.06
h1 = 100000
ret, output1 = cv2.threshold(img_gray, 114.06, 255, cv2.THRESH_BINARY)  
ret, output2 = cv2.threshold(img_gray, 115.80, 255, cv2.THRESH_BINARY)
ret, output3 = cv2.threshold(img_gray, 113.57, 255, cv2.THRESH_BINARY)
ret, output4 = cv2.threshold(img_gray, 113.68, 255, cv2.THRESH_BINARY) 

# 畫直方圖
# plt.plot(hist)
# plt.stem(k1 , h1)
# plt.title("Grayscale Histogram")
# plt.xlabel("Gray Level")
# plt.ylabel("Frequency")
# plt.show()

# cv2.imshow('oxxostudio', img)

cv2.imshow('GMM', output1)
cv2.imshow('SkewMix', output2)
cv2.imshow('GenNormMix', output3)
cv2.imshow('PowerGaussMix', output4)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Model	Threshold
# GMM	114.06
# SkewMix	115.80
# GenNormMix	113.57
# PowerGaussMix	113.68



