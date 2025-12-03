import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from PIL import Image 

def fit_gmm(pixels, n_components=3, random_state=0):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=random_state)
    gmm.fit(pixels.reshape(-1,1))
    weights = gmm.weights_
    means = gmm.means_.flatten()
    covs = gmm.covariances_.flatten()  # for 1D full cov, flatten works
    sigmas = np.sqrt(covs)
    # sort by mean ascending (keep permutation)
    order = np.argsort(means)
    return {
        'gmm': gmm,
        'weights': weights[order],
        'means': means[order],
        'sigmas': sigmas[order],
        'order': order
    }

def gaussian_pdf(x, mu, sigma):
    return (1.0 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-(x-mu)**2 / (2*sigma**2))

def solve_intersections(mu1, sigma1, w1, mu2, sigma2, w2):
    # Solve w1 * N(x|mu1,s1) = w2 * N(x|mu2,s2)
    # Equivalent to quadratic: a x^2 + b x + c = 0
    # derive coefficients as in analysis
    s1sq = sigma1**2
    s2sq = sigma2**2
    a = -1.0/s1sq + 1.0/s2sq
    b = 2.0*mu1/s1sq - 2.0*mu2/s2sq
    c = -mu1**2/s1sq + mu2**2/s2sq + 2.0 * np.log((w1 * sigma2) / (w2 * sigma1))
    # handle near-linear (a=0 -> linear eq)
    if np.isclose(a, 0.0):
        if np.isclose(b, 0.0):
            return np.array([])  # no solution or infinite
        x = -c / b
        return np.array([x])
    disc = b*b - 4*a*c
    if disc < 0:
        return np.array([])
    sqrt_disc = np.sqrt(disc)
    x1 = (-b + sqrt_disc) / (2*a)
    x2 = (-b - sqrt_disc) / (2*a)
    return np.array([x1, x2])

def compute_thresholds_from_gmm(params, gray_min=0, gray_max=255):
    means = params['means']
    sigmas = params['sigmas']
    weights = params['weights']
    n = len(means)
    intersections = []
    # compute intersections for every pair, but we'll pick the ones between adjacent means
    for i in range(n):
        for j in range(i+1, n):
            roots = solve_intersections(means[i], sigmas[i], weights[i],
                                        means[j], sigmas[j], weights[j])
            # keep real roots within range
            valid = [r for r in roots if np.isreal(r) and (gray_min <= r <= gray_max)]
            intersections.extend(valid)
    intersections = np.array(intersections)
    # choose intersections that lie between adjacent sorted means
    thresholds = []
    for k in range(n-1):
        left = means[k]
        right = means[k+1]
        # among intersections, pick those in (left, right)
        candidates = intersections[(intersections > left) & (intersections < right)]
        if candidates.size > 0:
            # if multiple, choose the one closest to midpoint (or choose smallest)
            mid = 0.5*(left+right)
            chosen = candidates[np.argmin(np.abs(candidates - mid))]
            thresholds.append(float(chosen))
    thresholds = sorted(thresholds)
    return thresholds

def thresholds_to_segmentation(img_np, thresholds):
    # single-channel uint8 grayscale image
    out = np.zeros_like(img_np, dtype=np.uint8)
    thr = [0] + [int(round(t)) for t in thresholds] + [256]
    for i in range(len(thr)-1):
        mask = (img_np >= thr[i]) & (img_np < thr[i+1])
        out[mask] = int(round(255 * (i / (len(thr)-2))) ) if len(thr)>2 else (255 if i==1 else 0)
    return out

# -------------------------
# Example usage (直接執行)
# -------------------------
if __name__ == "__main__":
    # 讀影像並轉灰階 (替換路徑)
    # 讀取灰階影像
    img = cv2.imread("lll.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_np = np.array(img_gray)

    pixels = img_np.reshape(-1,1)

    # Fit GMM (改 n_components 試試 2 或 3)
    params = fit_gmm(pixels, n_components=2, random_state=42)

    print("Means:", params['means'])
    print("Sigmas:", params['sigmas'])
    print("Weights:", params['weights'])

    # compute analytic intersections -> thresholds
    thresholds = compute_thresholds_from_gmm(params, gray_min=0, gray_max=255)
    print("Thresholds (from intersections):", thresholds)

    # segmentation by thresholds
    seg_by_thresholds = thresholds_to_segmentation(img_np, thresholds)

    # segmentation by GMM posterior (assign each pixel to component with highest posterior)
    gmm = params['gmm']
    labels = gmm.predict(pixels)  # labels correspond to sorted order? careful: gmm was fit before sorting
    # Because we sorted parameters, need to map original component indices to sorted order:
    # create mapping from original component index -> position in sorted array
    # but simpler: compute posteriors and assign by highest posterior with sorted-means tie-breaker
    post = gmm.predict_proba(pixels)  # shape (n_pixels, n_components)
    # reorder columns according to params['order']
    order = params['order']
    post_sorted = post[:, order]
    labels_sorted = np.argmax(post_sorted, axis=1)
    seg_by_gmm = labels_sorted.reshape(img_np.shape).astype(np.uint8)
    # normalize seg_by_gmm to 0..255 for display
    seg_by_gmm_vis = (seg_by_gmm / seg_by_gmm.max() * 255).astype(np.uint8) if seg_by_gmm.max()>0 else seg_by_gmm*0

    # Plot results
    x = np.arange(0, 256)
    hist, _ = np.histogram(pixels, bins=256, range=[0,256], density=True)
    # build GMM fit curve (using sorted params)
    gdfs = []
    for w, m, s in zip(params['weights'], params['means'], params['sigmas']):
        gdfs.append(w * (1.0/(np.sqrt(2*np.pi)*s)) * np.exp(-(x-m)**2 / (2*s*s)))
    gdf_sum = np.sum(gdfs, axis=0)

    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1); plt.imshow(img_np, cmap='gray'); plt.title("Original")
    plt.axis('off')
    plt.subplot(2,2,2); plt.plot(x, hist, label='hist (density)'); plt.plot(x, gdf_sum, label='GMM fit')
    for i,g in enumerate(gdfs):
        plt.plot(x, g, '--', label=f'G{i+1} μ={params["means"][i]:.1f} σ={params["sigmas"][i]:.1f}')
    for t in thresholds:
        plt.axvline(t, color='k', linestyle=':', linewidth=1)
    plt.legend(); plt.title("Histogram + GMM + Thresholds")
    plt.subplot(2,2,3); plt.imshow(seg_by_thresholds, cmap='gray'); plt.title("Segmentation from thresholds")
    plt.axis('off')
    plt.subplot(2,2,4); plt.imshow(seg_by_gmm_vis, cmap='gray'); plt.title("Segmentation from GMM posteriors")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
