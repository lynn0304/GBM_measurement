from cmath import cosh
from distutils.command.config import config
from pathlib import Path
import argparse
import numpy as np
from skimage import io, img_as_ubyte
from scipy.ndimage import convolve
from skimage.color import rgb2gray
from skimage.morphology import medial_axis, binary_opening, binary_closing, remove_small_objects, disk
from scipy.ndimage import distance_transform_edt, binary_erosion
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pandas as pd
import os
import json
try:
    from skan import Skeleton as SkanSkeleton, summarize as skan_summarize
    SKAN_OK = True
except Exception:
    SKAN_OK = False

# keep skeleton away from border(px=2)
def keep_away_from_border(mask, skel, border_px=2):
    safe = binary_erosion(mask, disk(border_px))
    return skel & safe

# for skeleton length calculation
def _path_length_from_coords(coords, metric="geodesic"):
    if metric == "euclidean": 
        diffs = np.diff(coords.astype(float), axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))
    return float(len(coords))

# calculate skeleton length
def prune_skeleton_by_length(skel_bool, min_len_px=15, metric="geodesic"):
    skel_bool = skel_bool.astype(bool)

    if SKAN_OK:
        sk = SkanSkeleton(skel_bool)
        stats = skan_summarize(sk, separator="_")
        candidate_cols = [
            "path_length", "geodesic_length", "euclidean_length",
            "path-length", "geodesic-length", "euclidean-length",
        ]
        length_col = next((c for c in candidate_cols if c in stats.columns), None)

        if length_col is None:
            lengths = []
            for idx in stats.index:
                coords = sk.path_coordinates(int(idx))
                lengths.append(_path_length_from_coords(coords, metric=metric))
            stats["path_length"] = lengths
            length_col = "path_length"
        keep_idx = stats.index[stats[length_col] >= float(min_len_px)]

        pruned = np.zeros_like(skel_bool, dtype=bool)
        for i in keep_idx:
            coords = sk.path_coordinates(int(i))
            rr, cc = coords[:, 0].astype(int), coords[:, 1].astype(int)
            pruned[rr, cc] = True
        return pruned

    k = np.ones((3, 3), np.uint8)
    pruned = skel_bool.copy()
    for _ in range(64):
        if not pruned.any():
            break
        nb = convolve(pruned.astype(np.uint8), k, mode='constant', cval=0)
        endpoints = pruned & (nb == 2)
        if endpoints.sum() == 0:
            break
        pruned[endpoints] = False
    return pruned

def read_mask(path, threshold=0):
    img = io.imread(path)
    if img.ndim == 3:
        img = rgb2gray(img)
        mask = img > 0.5
    else:
        mask = img > threshold
    return mask.astype(bool)

# fill some incomplete radius
def clean_mask(mask, open_radius=0, close_radius=0, min_obj=0):
    m = mask.copy()
    if open_radius > 0:
        m = binary_opening(m, footprint=disk(open_radius))
    if close_radius > 0:
        m = binary_closing(m, footprint=disk(close_radius))
    if min_obj and min_obj > 0:
        m = remove_small_objects(m, min_size=min_obj)
    return m

# skeletonize and calculate
def skeleton(mask, save=True):
    skel, dist_to_skel = medial_axis(mask, return_distance=True)
    min_len_px = 199
    skel = prune_skeleton_by_length(skel, min_len_px=min_len_px)
    skel = keep_away_from_border(mask, skel, border_px=2)
    if save:
        outdir = Path('skeleton')
        outdir.mkdir(parents=True, exist_ok=True)
        base = (mask.astype(np.uint8) * 128) 
        overlay = base.copy()
        overlay[skel] = 255                        
        io.imsave(outdir / "skeleton.png", overlay)
    return skel, dist_to_skel

# bootstrap ci
def ci(id, x, n=1000, ratio=97.5, seed=0):
    def plot_bootstrap_distribution(boot_means, percentiles, unit="nm", outname="bootstrap_means.png"):
        p5, p50, p95 = percentiles
        plt.figure(dpi=300)
        sns.histplot(boot_means, bins=40, kde=True, color="skyblue")
        plt.axvline(p5, color="red", linestyle="--", label=f"2.5%={p5:.2f} {unit}")
        plt.axvline(p50, color="green", linestyle=":", label=f"50%={p50:.2f} {unit}")
        plt.axvline(p95, color="red", linestyle="--", label=f"97.5%={p95:.2f} {unit}")
        plt.xlabel(f"Mean thickness ({unit})")
        plt.ylabel("Bootstrap count")
        plt.title("Bootstrap distribution of mean thickness")
        plt.legend()
        plt.savefig(outname)
    LEN = len(x)
    rng = np.random.default_rng(seed)
    boots = []
    for i in range(n):
        sample = x[rng.integers(0, LEN, LEN)]
        boots.append(sample.mean())
    low, high = np.percentile(boots, [100-ratio, ratio])
    boots_mean = np.array(boots)
    p5, p50, p95 = np.percentile(boots_mean, [100-ratio, 50, ratio])
    # plot_bootstrap_distribution(id, boots_mean, (p5, p50, p95))
    return low, high

# choose stride length
def rand_choose(id, t_nm_org, strides=[12, 10, 8, 6, 4], ci_max = 0.05, seed=0):
    choose=None
    rng = np.random.default_rng(seed)
    for s in strides:
        start = rng.integers(0, s)
        t_nm = t_nm_org[start::s]
        if len(t_nm)<30:
            continue
        low, high = ci(id, t_nm)
        mean_new = np.mean(t_nm)
        width = (high-low) / mean_new if mean_new > 0 else np.inf
        choose = (s, mean_new, (low, high), width)
        if width<ci_max:
            break
    if choose is None:
        s = strides[-1]
        start = 0
        t_nm = t_nm_org[start::s]
        low, high = ci(id, t_nm)
        mean_new = np.mean(t_nm)
        width = (high-low) / mean_new if mean_new > 0 else np.inf
        choose = (s, mean_new, (low, high), width)
    return choose, t_nm

######PLOTING#######
def hist(id, thickness, mean, median, unit='nm'):
    plt.figure(dpi=300)
    sns.histplot(thickness, bins=40, kde=True, color="skyblue")
    plt.axvline(mean, color="red", linestyle="--", label=f"Mean={mean:.1f} {unit}")
    plt.axvline(median, color="green", linestyle=":", label=f"Median={median:.1f} {unit}")
    plt.xlabel(f"Thickness ({unit})")
    plt.ylabel("Count")
    plt.title("GBM Thickness Distribution")
    plt.legend()
    plt.savefig(f'hist/{id}_thickness_{unit}.png')
    plt.close()

def plot_ci(id, strides, ci_widths, chosen_stride, unit='nm'):
    plt.figure(dpi=300)
    plt.plot(strides, ci_widths, marker="o")
    plt.axhline(0.05, color="red", linestyle="--", label="Target 5%")
    if chosen_stride is not None and chosen_stride in strides:
        idx = strides.index(chosen_stride)
        plt.scatter(strides[idx], ci_widths[idx], color="red", s=80, zorder=5, label=f"Chosen stride={chosen_stride}")
    plt.ylabel("95% CI relative width")
    plt.title(f"CI convergence by stride ({unit})")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.savefig(f"ci/{id}_CI_{unit}.png")
    plt.close()

def overlay(id, mask_path, img_path, thickness_nm, skel, outname="overlay.png"):
    img = io.imread(img_path)
    mask = io.imread(mask_path) > 0
    
    thickness_map = np.zeros_like(mask, dtype=float)
    coords = np.argwhere(skel)
    thickness_map[coords[:,0], coords[:,1]] = thickness_nm
    
    plt.figure(dpi=300)
    plt.imshow(img, cmap="gray")
    im = plt.imshow(thickness_map, cmap="jet", alpha=0.6,
                    norm=Normalize(vmin=np.percentile(thickness_nm, 5),
                                   vmax=np.percentile(thickness_nm, 95)))
    cbar = plt.colorbar(im)
    cbar.set_label("Thickness (nm)")
    plt.title("Thickness Heatmap Overlay")
    plt.axis("off")
    plt.savefig(outname)
    plt.savefig('overlay/{id}_overlay.png')
    plt.close()

def plot_bootstrap_distribution(id, boot_means, percentiles, unit="nm", outname="bootstrap_means.png"):
    p5, p50, p95 = percentiles
    plt.figure(dpi=300)
    sns.histplot(boot_means, bins=40, kde=True, color="skyblue")
    plt.axvline(p5, color="red", linestyle="--", label=f"5%={p5:.2f} {unit}")
    plt.axvline(p50, color="green", linestyle=":", label=f"50%={p50:.2f} {unit}")
    plt.axvline(p95, color="red", linestyle="--", label=f"95%={p95:.2f} {unit}")
    plt.xlabel(f"Mean thickness ({unit})")
    plt.ylabel("Bootstrap count")
    plt.title("Bootstrap distribution of mean thickness")
    plt.legend()
    plt.savefig('bs/{id}_bootstrap_means')
    plt.close()

def main(mask_path, img_path, id):
    with open("ratio.json", "r", encoding="utf-8") as f:
        RATIO = json.load(f)
    MASK_PATH = mask_path
    IMG_PATH = img_path
    try:
        NM_PER_PIXEL=RATIO.get(id, 1)
    except:
        NM_PER_PIXEL=1
    THETA = 0
    mask = read_mask(MASK_PATH)
    mask = clean_mask(mask)
    skel, dist_to_skel = skeleton(mask)
    n_skel = int(skel.sum())
    if n_skel > 0:
        theta_rad = np.deg2rad(THETA)
        cos_theta = np.cos(theta_rad)
        thickness_px = 2.0 * dist_to_skel[skel]*cos_theta
        thickness_nm = thickness_px * NM_PER_PIXEL
        mean_px = float(np.mean(thickness_px))
        med_px  = float(np.median(thickness_px))
        mean_nm = float(np.mean(thickness_nm))
        med_nm  = float(np.median(thickness_nm))
        p10_px, p90_px = np.percentile(thickness_px, [10, 90])
        p10_nm, p90_nm = np.percentile(thickness_nm, [10, 90])
        print(f"厚度(px):mean={mean_px:.3f}, median={med_px:.3f}, P10={p10_px:.3f}, P90={p90_px:.3f}")
        print(f"厚度(µm):mean={mean_nm:.3f}, median={med_nm:.3f}, P10={p10_nm:.3f}, P90={p90_nm:.3f}")
        hist(id, thickness_px, mean_px, med_px, unit='px')
        hist(id, thickness_nm, mean_nm, med_nm, unit='nm')
        # 加入CI
        (s, mean_new, (low, high), width), ci_thickness_px = rand_choose(id, thickness_px)
        print(f"重複採樣厚度(px):mean={mean_new:.3f}, mean 95% CI width={width}, mean 95% CI=[{low}, {high}], stride={s}")
        mean_px = mean_new


        stride_list = []
        width_list = []
        chosen_stride = None
        for s in [12, 10, 8, 6, 4, 2]:
            (_, mean_new, (low, high), width), _ = rand_choose(id, thickness_px, strides=[s])
            stride_list.append(s)
            width_list.append(width)
            if width < 0.05 and chosen_stride is None:
                chosen_stride = s
        plot_ci(id, stride_list, width_list, chosen_stride, unit='px')
        
        
        (s, mean_new, (low, high), width), ci_thickness_nm = rand_choose(id, thickness_nm)
        print(f"重複採樣厚度(µm):mean={mean_new:.3f}, mean 95% CI width={width}, mean 95% CI=[{low}, {high}], stride={s}")
        mean_nm = mean_new
        mean_95_width = width
        ci_low = low
        ci_high = high
        stride = s
        chosen_stride = None
        stride_list = []
        width_list = []
        for s in [12, 10, 8, 6, 4, 2]:
            (_, mean_new, (low, high), width), _ = rand_choose(id, thickness_nm, strides=[s])
            stride_list.append(s)
            width_list.append(width)
            if width < 0.05 and chosen_stride is None:
                chosen_stride = s
        plot_ci(id, stride_list, width_list, chosen_stride, unit='µm')
        
        overlay(id, MASK_PATH, IMG_PATH, thickness_nm, skel)
        return mean_nm, mean_95_width, ci_low, ci_high, stride

#####PARAMETER#####
MASK_PATH='slide_pick/combined'
IMG_PATH='slide_pick/overlay'
OUTPUT_CSV='result_1014_nm.csv'
###################

mean_px = []
ids = []
conf_95 = []
low = []
high = []
s = []
for mask in os.listdir(MASK_PATH):
    mask_path = MASK_PATH + '/' + mask
    img_path = IMG_PATH + '/' + mask.replace('_mask', '_overlay')
    id = mask.replace('_mask.png', '')
    try:
        mean_px_, mean_95, cilow, cihigh, stride = main(mask_path, img_path, id)
        mean_px.append(mean_px_)
        ids.append(id)
        conf_95.append(mean_95)
        low.append(cilow)
        high.append(cihigh)
        s.append(stride)
    except:
        continue
dict = {
    "id": ids,
    "mean_nm": mean_px,
    "conf_95_width": conf_95,
    "ci_low": low,
    "ci_high": high,
    "stride": s
}
df = pd.DataFrame(dict)
df.to_csv(OUTPUT_CSV)
