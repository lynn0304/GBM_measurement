import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress

####PARAMETERS####
res_path = Path("result_1014_nm.csv") # csv from measure.py
per_image_csv = "per_image_pixel_avg.csv" # output csv file 
gt_csv = pd.read_excel("/Users/lynnchang/Desktop/EM/GBM_thickness.xlsx", sheet_name='工作表2') # ground truth file
##################

# aggregate all patches' data into per case data 
df = pd.read_csv(res_path)

def to_image_id(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    return s.split("_")[0]

if "id" not in df.columns or "mean_nm" not in df.columns:
    raise ValueError("Expected columns ['id','mean_px'] in result_0930.csv")

df["image_id"] = df["id"].map(to_image_id)
agg = (
    df.groupby("image_id", as_index=False)
      .agg(mean_px=("mean_nm","mean"),
           median_px=("mean_nm","median"),
           n_patches=("mean_nm","size"))
      .sort_values("image_id")
)

agg.to_csv(per_image_csv, index=False)

agg = pd.read_csv(per_image_csv)


# calculate AI/measure ratio
ratio = []
id = []
test = []
gt = []
for i in range(len(agg['image_id'])):
    for j in range(len(gt_csv['ID'])):
        if agg['image_id'][i] == gt_csv['ID'][j]:
            if agg["n_patches"][i] < 5:
            # if 0.9 < gt_csv['thickness'][j]/agg["mean_px"][i] <1.1:
                ratio.append(gt_csv['thickness'][j]/agg['mean_px'][i])
                test.append(agg['mean_px'][i])
                gt.append(gt_csv['thickness'][j])
                id.append(agg['image_id'][i])


print(ratio)
plt.figure(dpi=150)
plt.plot([x for x in range(1, len(ratio)+1)], ratio)
plt.xticks([x for x in range(1, len(ratio)+1)], [f"case {x}" for x in range(1, len(ratio)+1)], rotation=90)
# plt.xticks([x for x in range(1, len(ratio)+1)], id, rotation=90)
plt.plot([0.5, len(ratio)+0.5], [1, 1], linestyle='--', color="red")
plt.tight_layout()
plt.savefig('ratio.png')

# lolipop plot
def plot_paired_dot_lollipop(categories, a, b, labels=("AI","Manual"), filename="lollipop.png"):
    categories = list(categories)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    y = np.arange(len(categories))

    fig, ax = plt.subplots()
    for i in range(len(categories)):
        ax.plot([a[i], b[i]], [y[i], y[i]], marker=None)
    ax.plot(a, y, 'o', label=labels[0])
    ax.plot(b, y, 'o', label=labels[1])

    ax.set_yticklabels(categories)
    ax.set_xlabel("Value")
    ax.set_yticks([x for x in range(len(ratio))], [f"case {x}" for x in range(1, len(ratio)+1)])
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.show()

plot_paired_dot_lollipop(id, test, gt)


# delta bar of AI-measure
def plot_delta_bar(categories, a, b, filename="delta_bar.png"):
    categories = list(categories)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    delta = b - a

    fig, ax = plt.subplots()
    ax.bar(categories, delta)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_ylabel("Δ (Manual - AI)")
    # ax.set_title("Delta bar chart (Increase/Decrease)")
    ax.set_xticklabels([f"case {x}" for x in range(1, len(ratio)+1)], rotation=90)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.show()
    return delta

plot_delta_bar(id, test, gt)



# linear-correlation
def plot_corr_scatter(y_pred, y_true, filename="corr.png", title="AI vs. Manual", offset_nm=63.0):
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    mask_finite = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask_finite.sum() < len(y_pred):
        bad_idx = np.where(~mask_finite)[0]
        print(f"[warn] dropped {len(bad_idx)} non-finite points at indices: {bad_idx.tolist()}")
    x = y_pred[mask_finite]
    y = y_true[mask_finite]
    if x.size < 2:
        return None
    if np.allclose(x, x.mean()) or np.allclose(y, y.mean()):
        _scatter_only(x, y, filename, title)
        return None
    r, p = pearsonr(x, y)
    slope, intercept, r_lin, p_lin, stderr = linregress(x, y)
    r2 = r**2
    x_corr = x - float(offset_nm)
    r_c, p_c = pearsonr(x_corr, y)
    slope_c, intercept_c, *_ = linregress(x_corr, y)
    r2_c = r_c**2
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    xs = np.linspace(lims[0], lims[1], 200)

    plt.figure(dpi=150)
    plt.scatter(x, y, s=25, label="Samples")
    plt.plot(xs, xs, ls="--", lw=1, label="Identity")
    plt.plot(xs, slope*xs + intercept, ls="-", lw=1.5, color="#ff7f0e",
             label=f"Fit: y={slope:.2f}x{intercept:+.2f}")
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("AI-derived measurement")
    plt.ylabel("Manual measurement")
    plt.title(title)
    plt.legend(loc="lower right")

    ax = plt.gca()
    ax.text(0.05, 0.95, f"r = {r:.3f}\nR² = {r2:.3f}\np = {p:.2e}\nN = {x.size}",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    # ax.text(0.95, 0.95, f"after −{offset_nm:.0f} nm:\n"
    #                     f"r = {r_c:.3f}\nR² = {r2_c:.3f}",
    #         transform=ax.transAxes, va="top", ha="right",
    #         bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    return dict(
        r=r, r2=r2, p=p, slope=slope, intercept=intercept, n=int(x.size),
        r_after=r_c, r2_after=r2_c, slope_after=slope_c, intercept_after=intercept_c,
        offset_nm=float(offset_nm)
    )

# Bland-Altman
def plot_bland_altman(y_pred, y_true, filename="bland_altman.png", title="Bland-Altman"):
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    avg = (y_pred + y_true) / 2.0
    diff = y_pred - y_true

    md = np.mean(diff)
    sd = np.std(diff, ddof=1)
    loa_low  = md - 1.96 * sd
    loa_high = md + 1.96 * sd

    plt.figure(figsize=(5.6, 4.5), dpi=150)
    plt.scatter(avg, diff, s=25)
    plt.axhline(md, color="k", lw=1, label=f"Mean diff = {md:.2f}")
    plt.axhline(loa_low,  color="r", ls="--", lw=1, label=f"LoA low = {loa_low:.2f}")
    plt.axhline(loa_high, color="r", ls="--", lw=1, label=f"LoA high = {loa_high:.2f}")
    plt.xlabel("Average of AI and Manual")
    plt.ylabel("AI − Manual")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return dict(mean_diff=md, loa_low=loa_low, loa_high=loa_high, sd=sd)

stats_corr = plot_corr_scatter(test, gt, filename="corr.png", title="AI vs. Manual")
stats_ba   = plot_bland_altman(test, gt, filename="bland_altman.png")

print("Correlation:", stats_corr)
print("Bland-Altman:", stats_ba)


# bar plot
def plot_ratio_bar(categories, mean_m, gt, filename="ratio_bar.png"):
    cats = list(categories)
    mean_m = np.asarray(mean_m, dtype=float)
    gt     = np.asarray(gt, dtype=float)
    ratio  = gt / mean_m

    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.35*len(cats))))
    ax.bar(cats, ratio)
    ax.axhline(1.0, ls="--", color="k", lw=1, label="ratio = 1")
    ax.set_ylabel("GT / AI mean")
    ax.set_xticklabels(cats, rotation=90)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    return ratio

plot_ratio_bar(id, test, gt)






