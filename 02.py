import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from PIL import Image #沒用到

# 讀取灰階影像
img = cv2.imread("lll.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_np = np.array(img_gray)

# hist = cv2.calcHist([img], [0], None, [256], [0, 256])
# plt.plot(hist)

# 將像素展開成 1D，作為 GMM 的輸入
pixels = img_np.reshape(-1, 1)

# 設定 GMM 裡 Gaussian 的個數（可選 2、3、4...）
gmm = GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(pixels)

# 取出 GMM 參數
weights = gmm.weights_         # 各 Gaussian 的權重
means = gmm.means_.flatten()   # 平均 (μ)
sigmas = np.sqrt(gmm.covariances_.flatten())  # 標準差 (σ)

print("Weights:", weights)
print("Means:", means)
print("Sigmas:", sigmas)

# -------------------------------------------------------
# 建立 x 軸（灰階 0~255）
x = np.arange(0, 256)

# 高斯函數
def gaussian(x, mu, sigma, weight):
    return weight * (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-(x-mu)**2 / (2*sigma**2))

# 計算每個 Gaussian 的 PDF
gdfs = [gaussian(x, means[i], sigmas[i], weights[i]) for i in range(len(weights))]
gdf_sum = np.sum(gdfs, axis=0)

# -------------------------------------------------------
# 畫直方圖與 GMM 拟合曲線
hist, bins = np.histogram(pixels, bins=256, range=[0, 256], density=True)

plt.figure(figsize=(10, 6))
plt.plot(x, hist, label="Histogram")
plt.plot(x, gdf_sum, label="GMM Fit", linewidth=2)

# 個別 Gaussian
for i, g in enumerate(gdfs):
    plt.plot(x, g, linestyle="--", label=f"G{i+1} (μ={means[i]:.1f}, σ={sigmas[i]:.1f})")

plt.title("Histogram + GMM Gaussian Fit")
plt.xlabel("Gray Level")
plt.ylabel("Probability")
plt.legend()
plt.show()





