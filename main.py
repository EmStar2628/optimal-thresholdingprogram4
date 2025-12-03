import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.mixture import GaussianMixture
from scipy.stats import skewnorm, gennorm
from scipy.optimize import curve_fit, brentq
from scipy.integrate import simpson

# -------------------------
# Utilities
# -------------------------
def hist_density(img_np, bins=256):
    pixels = img_np.flatten()
    hist, bin_edges = np.histogram(pixels, bins=bins, range=(0,256), density=True)
    x = 0.5*(bin_edges[:-1] + bin_edges[1:])
    return x, hist

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def find_intersections_numeric(func1, func2, xgrid):
    xs = []
    for i in range(len(xgrid)-1):
        a, b = xgrid[i], xgrid[i+1]
        fa, fb = func1(a)-func2(a), func1(b)-func2(b)
        if fa == 0:
            xs.append(a)
        elif fa*fb < 0:
            try:
                r = brentq(lambda t: func1(t)-func2(t), a, b)
                xs.append(r)
            except ValueError:
                pass
    return xs

def expected_error_from_components(comp_pdfs, xgrid):
    # comp_pdfs: list of pdf arrays (same xgrid) that sum to mixture
    # expected classification error = integral over x of (mixture - max_i p_i(x)) dx
    mix = np.sum(comp_pdfs, axis=0)
    max_comp = np.max(comp_pdfs, axis=0)
    residual = mix - max_comp
    # numeric integrate over xgrid
    err = simpson(residual, xgrid)
    return err

# -------------------------
# Models definitions
# -------------------------
def fit_gmm_hist(x, hist, n_components=3):
    # Fit GMM to raw pixels (better) but here we approximate by sampling according to hist
    # Simpler: sample from histogram to create pixel-like samples
    probs = hist / hist.sum()
    # create synthetic sample set proportional to hist
    sample_counts = (probs * 20000).astype(int)  # ~20k samples
    samples = []
    for xi, c in zip(x, sample_counts):
        if c>0:
            # create c samples at value xi (approx)
            samples.append(np.full(c, xi))
    if len(samples)==0:
        raise RuntimeError("too few histogram mass")
    samples = np.concatenate(samples).reshape(-1,1)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gmm.fit(samples)
    weights = gmm.weights_
    means = gmm.means_.flatten()
    covs = gmm.covariances_.flatten()
    sigmas = np.sqrt(covs)
    # sort by mean
    order = np.argsort(means)
    weights = weights[order]; means = means[order]; sigmas = sigmas[order]
    # build component pdfs on x
    comp_pdfs = []
    for w,m,s in zip(weights, means, sigmas):
        pdf = w * (1.0/(np.sqrt(2*np.pi)*s)) * np.exp(-(x-m)**2/(2*s*s))
        comp_pdfs.append(pdf)
    comp_pdfs = np.array(comp_pdfs)
    mixture = comp_pdfs.sum(axis=0)
    return {'name':'GMM', 'weights':weights, 'means':means, 'sigmas':sigmas,
            'comp_pdfs':comp_pdfs, 'mixture':mixture, 'order':order, 'model':gmm}

def mixture_skewnorm_pdf(x, params):
    # params: [w,a,mu,s] * K
    K = len(params)//4
    y = np.zeros_like(x, dtype=float)
    comps = []
    for k in range(K):
        w = params[4*k+0]; a = params[4*k+1]; mu = params[4*k+2]; s = params[4*k+3]
        pdf = w * skewnorm.pdf(x, a, loc=mu, scale=s)
        comps.append(pdf)
        y += pdf
    return y, np.array(comps)

def fit_skewmix_hist(x, hist, K=3):
    # initial params from GMM for stability
    g = fit_gmm_hist(x, hist, n_components=K)
    p0 = []
    bounds_low=[]; bounds_up=[]
    for i in range(K):
        w = float(g['weights'][i])
        mu = float(g['means'][i])
        s = float(max(1.0, g['sigmas'][i]))
        p0 += [w, 0.0, mu, s]  # a init 0
        bounds_low += [0.0, -80.0, 0.0, 0.5]
        bounds_up  += [1.0, 80.0, 255.0, 200.0]
    def f_to_fit(xvals, *params):
        y,_ = mixture_skewnorm_pdf(xvals, params)
        return y
    popt, pcov = curve_fit(f_to_fit, x, hist, p0=p0, bounds=(bounds_low,bounds_up), maxfev=20000)
    mixture, comps = mixture_skewnorm_pdf(x, popt)
    weights=[]; means=[]; sigmas=[]
    for k in range(K):
        weights.append(popt[4*k+0]); means.append(popt[4*k+2]); sigmas.append(popt[4*k+3])
    # normalize weights so sum to 1 (curve_fit may not enforce exactly)
    weights = np.array(weights); weights = weights/weights.sum()
    # scale comps accordingly
    comps = np.array(comps)
    # recompute comps using normalized weights
    comps = []
    for k in range(K):
        w = weights[k]; a = popt[4*k+1]; mu = popt[4*k+2]; s = popt[4*k+3]
        comps.append(w * skewnorm.pdf(x, a, loc=mu, scale=s))
    comps = np.array(comps)
    return {'name':'SkewMix', 'weights':weights, 'means':np.array(means), 'sigmas':np.array(sigmas),
            'comp_pdfs':comps, 'mixture':comps.sum(axis=0), 'popt':popt}

def mixture_gennorm_pdf(x, params):
    K = len(params)//4
    y = np.zeros_like(x, dtype=float); comps=[]
    for k in range(K):
        w = params[4*k+0]; beta = params[4*k+1]; mu = params[4*k+2]; scale = params[4*k+3]
        pdf = w * gennorm.pdf(x, beta, loc=mu, scale=scale)
        comps.append(pdf); y+=pdf
    return y, np.array(comps)

def fit_gennormmix_hist(x, hist, K=3):
    g = fit_gmm_hist(x, hist, n_components=K)
    p0=[]; lb=[]; ub=[]
    for i in range(K):
        p0 += [float(g['weights'][i]), 2.0, float(g['means'][i]), float(max(1.0,g['sigmas'][i]))]
        lb += [0.0, 0.1, 0.0, 0.5]
        ub += [1.0, 10.0, 255.0, 200.0]
    def f_to_fit(xvals, *params):
        y,_ = mixture_gennorm_pdf(xvals, params)
        return y
    popt, pcov = curve_fit(f_to_fit, x, hist, p0=p0, bounds=(lb,ub), maxfev=20000)
    _, comps = mixture_gennorm_pdf(x, popt)
    weights = np.array([popt[4*k+0] for k in range(K)])
    weights /= weights.sum()
    # recompute comps using normalized weights
    comps = []
    for k in range(K):
        w = weights[k]; beta = popt[4*k+1]; mu = popt[4*k+2]; s = popt[4*k+3]
        comps.append(w * gennorm.pdf(x, beta, loc=mu, scale=s))
    comps = np.array(comps)
    means = np.array([popt[4*k+2] for k in range(K)])
    sigmas = np.array([popt[4*k+3] for k in range(K)])
    return {'name':'GenNormMix', 'weights':weights, 'means':means, 'sigmas':sigmas,
            'comp_pdfs':comps, 'mixture':comps.sum(axis=0), 'popt':popt}

def mixture_powergauss_pdf(x, params):
    # params per comp: [w, mu, sigma, p]
    K = len(params)//4
    y = np.zeros_like(x, dtype=float); comps=[]
    for k in range(K):
        w = params[4*k+0]; mu = params[4*k+1]; s = params[4*k+2]; p = params[4*k+3]
        g = (1.0/(np.sqrt(2*np.pi)*s)) * np.exp(-(x-mu)**2/(2*s*s))
        gp = g**p
        # re-normalize gp to area 1
        gp = gp / (np.trapz(gp, x))
        comps.append(w * gp); y += w*gp
    return y, np.array(comps)

def fit_powergaussmix_hist(x, hist, K=3):
    g = fit_gmm_hist(x, hist, n_components=K)
    p0=[]; lb=[]; ub=[]
    for i in range(K):
        p0 += [float(g['weights'][i]), float(g['means'][i]), float(max(1.0,g['sigmas'][i])), 1.0]
        lb  += [0.0, 0.0, 0.5, 0.2]
        ub  += [1.0, 255.0, 200.0, 5.0]
    def f_to_fit(xvals, *params):
        y,_ = mixture_powergauss_pdf(xvals, params)
        return y
    popt, pcov = curve_fit(f_to_fit, x, hist, p0=p0, bounds=(lb,ub), maxfev=20000)
    _, comps = mixture_powergauss_pdf(x, popt)
    weights = np.array([popt[4*k+0] for k in range(K)]); weights /= weights.sum()
    comps2=[]
    for k in range(K):
        w = weights[k]; mu = popt[4*k+1]; s = popt[4*k+2]; p = popt[4*k+3]
        g = (1.0/(np.sqrt(2*np.pi)*s)) * np.exp(-(x-mu)**2/(2*s*s))
        gp = g**p; gp = gp/np.trapz(gp,x)
        comps2.append(w*gp)
    comps2 = np.array(comps2)
    means = np.array([popt[4*k+1] for k in range(K)])
    sigmas = np.array([popt[4*k+2] for k in range(K)])
    return {'name':'PowerGaussMix', 'weights':weights, 'means':means, 'sigmas':sigmas,
            'comp_pdfs':comps2, 'mixture':comps2.sum(axis=0), 'popt':popt}

# -------------------------
# Compare function
# -------------------------
def compute_thresholds_from_comp_pdfs(x, comp_pdfs, means):
    # compute intersections between adjacent means
    K = comp_pdfs.shape[0]
    thresholds=[]
    for k in range(K-1):
        f1 = lambda t: np.interp(t, x, comp_pdfs[k])
        f2 = lambda t: np.interp(t, x, comp_pdfs[k+1])
        xs = find_intersections_numeric(f1, f2, x)
        # keep intersections between means[k] and means[k+1]
        cand = [v for v in xs if v>means[k] and v<means[k+1]]
        if len(cand)>0:
            # pick the one closest to midpoint
            mid = 0.5*(means[k]+means[k+1])
            chosen = min(cand, key=lambda v: abs(v-mid))
            thresholds.append(chosen)
    return thresholds

def compare_models_on_image(image_path, K=3, bins=256):
    img = Image.open(image_path).convert('L')
    img_np = np.array(img)
    x, hist = hist_density(img_np, bins=bins)

    models = []
    # 1) Gaussian Mixture (sklearn)
    try:
        m1 = fit_gmm_hist(x, hist, n_components=K)
        models.append(m1)
    except Exception as e:
        print("GMM fit error:", e)
    # 2) Skew mix
    try:
        m2 = fit_skewmix_hist(x, hist, K=K)
        models.append(m2)
    except Exception as e:
        print("Skew fit error:", e)
    # 3) GenNorm mix
    try:
        m3 = fit_gennormmix_hist(x, hist, K=K)
        models.append(m3)
    except Exception as e:
        print("Gennorm fit error:", e)
    # 4) Power gauss
    try:
        m4 = fit_powergaussmix_hist(x, hist, K=K)
        models.append(m4)
    except Exception as e:
        print("PowerGauss fit error:", e)

    # Evaluate models
    results = []
    for m in models:
        comp_pdfs = m['comp_pdfs']
        mixture = m['mixture']
        means = m['means']
        # ensure components aligned by mean
        order = np.argsort(means)
        comp_pdfs = comp_pdfs[order]
        means = means[order]
        # RMSE to hist
        hist_rmse = rmse(hist, mixture)
        # thresholds
        thresholds = compute_thresholds_from_comp_pdfs(x, comp_pdfs, means)
        # expected classification error
        exp_err = expected_error_from_components(comp_pdfs, x)
        # AIC/BIC approximations: use RSS and k parameters
        rss = np.sum((hist - mixture)**2)
        n = len(hist)
        # parameter count k: approx 4*K for skew/gennorm/power, for GMM approx 3*K-1 (weights sum constraint)
        if m['name']=='GMM':
            k = 3*len(means)-1
        else:
            k = 4*len(means)
        aic = 2*k + n * np.log(rss/n + 1e-12)
        bic = np.log(n)*k + n * np.log(rss/n + 1e-12)
        results.append({'name':m['name'], 'rmse':hist_rmse, 'thresholds':thresholds,
                        'exp_err':exp_err, 'aic':aic, 'bic':bic, 'model':m, 'x':x, 'hist':hist})
    # print summary
    for r in results:
        print("Model:", r['name'])
        print("  RMSE:", r['rmse'])
        print("  thresholds:", r['thresholds'])
        print("  expected error:", r['exp_err'])
        print("  AIC:", r['aic'], "BIC:", r['bic'])
        print("-----")
    # plot comparison
    plt.figure(figsize=(12, 8))
    plt.plot(x, hist, label='hist', linewidth=2)
    colors = ['C0','C1','C2','C3']
    for idx, r in enumerate(results):
        m = r['model']
        mixture = r['model']['mixture']
        plt.plot(x, mixture, label=f"{r['name']} mixture", linestyle='--', color=colors[idx])
        for comp in r['model']['comp_pdfs']:
            plt.plot(x, comp, alpha=0.6, linestyle=':', color=colors[idx])
        for t in r['thresholds']:
            plt.axvline(t, color=colors[idx], linestyle=':', alpha=0.8)
    plt.legend(); plt.title('Model fits and thresholds'); plt.xlabel('Gray level'); plt.ylabel('Density')
    plt.show()

    return results

# -------------------------
# Example run (替換 image path)
# -------------------------
if __name__ == "__main__":
    results = compare_models_on_image("lll.png", K=2, bins=256)
