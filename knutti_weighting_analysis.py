"""
knutti_weighting_analysis.py
============================
Simplified GSAT-only Knutti-style performance + interdependence weighting
of CMIP6 models against ERA5 observations.

Scope:
  - Performance criterion: RMSE of model ensemble-mean GSAT anomaly vs ERA5 GSAT
    anomaly over the historical calibration period (1940–2014).
  - Interdependence criterion: pairwise RMSE between model ensemble-mean
    anomaly trajectories over the same period.
  - IMPORTANT: This is NOT the full multi-diagnostic Knutti methodology.
    It is a GSAT-only simplification for educational demonstration.

Data notes:
  - CMIP6 Temperature column is in °C (absolute); ERA5 GSAT_era5 is in K.
  - Anomaly computation subtracts the 1981–2014 mean from each time series,
    which cancels the 273.15 K offset. All anomalies are in °C / K (equivalent).
  - We use ENSEMBLE MEANS per model (not individual members) to avoid
    models with many members having undue influence.

References:
  - Knutti et al. (2017), Geophysical Research Letters
  - Tokarska et al. (2020), Science Advances

Output: all figures saved to ./presentation_figures/
"""

# =============================================================================
# 0. Imports and setup
# =============================================================================
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ── output directory ──────────────────────────────────────────────────────────
OUT_DIR = "presentation_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── global plot style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
})

# ── key periods ────────────────────────────────────────────────────────────────
REF_START, REF_END       = 1981, 2014   # anomaly reference period
CALIB_START, CALIB_END   = 1940, 2014   # calibration for weights (within historical)
HINDCAST_TRAIN_START     = 1940
HINDCAST_TRAIN_END       = 1989
HINDCAST_TEST_START      = 1990
HINDCAST_TEST_END        = 2014         # stay within CMIP6 historical experiment
FUTURE_START             = 2025
EOC_START, EOC_END       = 2071, 2100   # end-of-century definition

# =============================================================================
# 1. Load data
# =============================================================================
DATAPATH = "data"

with open(os.path.join(DATAPATH, "cmip6_surface_temperature_5mems.pkl"), "rb") as f:
    df_raw = pickle.load(f)

with open(os.path.join(DATAPATH, "era5_GSAT.pkl"), "rb") as f:
    df_era5_raw = pickle.load(f)

# Verify columns
assert {"Temperature", "Model", "run", "Year"}.issubset(df_raw.columns)
assert {"GSAT_era5", "Year"}.issubset(df_era5_raw.columns)

print(f"CMIP6: {df_raw.Model.nunique()} models, years {df_raw.Year.min()}–{df_raw.Year.max()}")
print(f"ERA5:  years {df_era5_raw.Year.min()}–{df_era5_raw.Year.max()}")

# =============================================================================
# 2. Preprocessing
# =============================================================================

# ── 2a. Compute per-model ensemble means (one trajectory per model) ──────────
# Rationale: prevents models with 50 members from having 10× the influence
#            of models with 5 members.
df_emean = (df_raw
            .groupby(["Model", "Year"])["Temperature"]
            .mean()
            .reset_index()
            .rename(columns={"Temperature": "T_emean"}))

MODELS = sorted(df_emean.Model.unique())
N_MODELS = len(MODELS)
print(f"Working with {N_MODELS} model ensemble means.")

# ── 2b. Compute GSAT anomalies relative to 1981–2014 ─────────────────────────
# Each model: subtract its own 1981–2014 mean.
def compute_anomaly(series, years, ref_start=REF_START, ref_end=REF_END):
    """Subtract reference-period mean from a time series."""
    ref_mask = (years >= ref_start) & (years <= ref_end)
    ref_mean = series[ref_mask].mean()
    return series - ref_mean

# CMIP6 anomalies
ref_means_cmip6 = {}
for model in MODELS:
    mask_model = df_emean.Model == model
    mask_ref   = mask_model & df_emean.Year.between(REF_START, REF_END)
    ref_means_cmip6[model] = df_emean.loc[mask_ref, "T_emean"].mean()

df_emean["Anomaly"] = df_emean.apply(
    lambda row: row["T_emean"] - ref_means_cmip6[row["Model"]], axis=1
)

# ERA5 anomalies (ERA5 is in Kelvin; subtracting its own mean gives K anomaly = °C anomaly)
era5_ref_mask = df_era5_raw.Year.between(REF_START, REF_END)
era5_ref_mean = df_era5_raw.loc[era5_ref_mask, "GSAT_era5"].mean()
df_era5 = df_era5_raw.copy()
df_era5["Anomaly"] = df_era5["GSAT_era5"] - era5_ref_mean

print(f"ERA5 reference mean ({REF_START}–{REF_END}): {era5_ref_mean:.3f} K")

# ── 2c. Build a model × year matrix for convenience ─────────────────────────
# Rows = Years, Columns = Models
df_pivot = (df_emean
            .pivot(index="Year", columns="Model", values="Anomaly")
            .sort_index())

# ERA5 as a Series indexed by Year
era5_series = df_era5.set_index("Year")["Anomaly"].sort_index()

# =============================================================================
# 3. Knutti-style weighting
# =============================================================================
# Simplified GSAT-only performance+interdependence weighting following:
#   Knutti et al. (2017), GRL  –  Equations 1–3
#
# w_i  ∝  w_perf_i  ×  w_indep_i
#
# w_perf_i   = exp( -D_i²  / sigma_D² )        [performance against ERA5]
# w_indep_i  = 1 / (1 + Σ_{j≠i} exp(-S_ij² / sigma_S²))  [independence]
#
# D_i   = RMSE(model_i_anom, ERA5_anom)     over calibration period
# S_ij  = RMSE(model_i_anom, model_j_anom) over calibration period
#
# sigma_D, sigma_S are scale parameters (set to the median of all D_i / S_ij as default)

def compute_rmse(a, b):
    """Root mean squared error between two aligned arrays."""
    diff = np.array(a) - np.array(b)
    return np.sqrt(np.mean(diff**2))


def compute_knutti_weights(df_pivot, era5_series, calib_start, calib_end,
                           sigma_D=None, sigma_S=None):
    """
    Compute simplified Knutti-style weights.

    Parameters
    ----------
    df_pivot    : DataFrame (year × model, anomalies)
    era5_series : Series indexed by year (anomalies)
    calib_start, calib_end : calibration period bounds
    sigma_D     : performance scale; if None, use median(D_i)
    sigma_S     : independence scale; if None, use median(S_ij)

    Returns
    -------
    dict with keys: D, S_matrix, sigma_D, sigma_S,
                    w_perf, w_indep, weights (normalized final)
    """
    models = df_pivot.columns.tolist()
    n = len(models)

    # Align calibration period
    calib_years = [y for y in df_pivot.index
                   if calib_start <= y <= calib_end and y in era5_series.index]
    era5_calib  = era5_series.loc[calib_years].values
    cmip_calib  = df_pivot.loc[calib_years]          # shape (T, n)

    # ── Performance: D_i = RMSE(model_i, ERA5) ─────────────────────────────
    D = np.array([compute_rmse(cmip_calib[m].values, era5_calib) for m in models])

    # ── Interdependence: S_ij = RMSE(model_i, model_j) ─────────────────────
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                S[i, j] = compute_rmse(cmip_calib[models[i]].values,
                                       cmip_calib[models[j]].values)

    # ── Scale parameters ────────────────────────────────────────────────────
    if sigma_D is None:
        sigma_D = np.median(D)
    if sigma_S is None:
        # median of off-diagonal S values
        off_diag = S[S > 0]
        sigma_S = np.median(off_diag)

    # ── Performance weights ─────────────────────────────────────────────────
    w_perf = np.exp(-(D**2) / (sigma_D**2))

    # ── Independence weights ────────────────────────────────────────────────
    # For each model i: sum of exp(-S_ij²/sigma_S²) over j≠i
    w_indep = np.zeros(n)
    for i in range(n):
        redundancy = sum(np.exp(-(S[i, j]**2) / (sigma_S**2))
                         for j in range(n) if j != i)
        w_indep[i] = 1.0 / (1.0 + redundancy)

    # ── Final weights (unnormalized product, then normalize) ─────────────────
    w_raw = w_perf * w_indep
    weights = w_raw / w_raw.sum()

    return {
        "models":   models,
        "D":        D,           # RMSE to ERA5 per model
        "S_matrix": S,           # pairwise RMSE matrix
        "sigma_D":  sigma_D,
        "sigma_S":  sigma_S,
        "w_perf":   w_perf / w_perf.sum(),    # normalized for display
        "w_indep":  w_indep / w_indep.sum(),  # normalized for display
        "w_perf_raw":  w_perf,
        "w_indep_raw": w_indep,
        "weights":  weights,     # final normalized weights
    }


# ── Compute default weights ───────────────────────────────────────────────────
wt = compute_knutti_weights(df_pivot, era5_series, CALIB_START, CALIB_END)

print(f"\nDefault sigma_D = {wt['sigma_D']:.4f}degC  |  sigma_S = {wt['sigma_S']:.4f}degC")
print("\nFinal Perf.+Indep. weights:")
for m, w, d in zip(wt["models"], wt["weights"], wt["D"]):
    print(f"  {m:<25s}  weight={w:.4f}  RMSE_to_ERA5={d:.4f}")

# ── Equal weights ─────────────────────────────────────────────────────────────
equal_weights = np.ones(N_MODELS) / N_MODELS


# =============================================================================
# 4. Weighted ensemble statistics (helper functions)
# =============================================================================

def weighted_mean(values, weights):
    return np.average(values, weights=weights)


def weighted_percentile(values, weights, p):
    """Weighted percentile using sorted-CDF interpolation."""
    idx    = np.argsort(values)
    vs     = np.array(values)[idx]
    ws     = np.array(weights)[idx]
    cum_w  = np.cumsum(ws)
    cum_w /= cum_w[-1]
    return np.interp(p / 100.0, cum_w, vs)


def weighted_median(values, weights):
    return weighted_percentile(values, weights, 50)


def ensemble_trajectory(df_pivot, weights, years=None):
    """Compute weighted ensemble mean trajectory."""
    models = df_pivot.columns.tolist()
    if years is not None:
        data = df_pivot.loc[years]
    else:
        data = df_pivot
    traj = data.values @ weights   # (T,) weighted mean
    return pd.Series(traj, index=data.index)


# ── Full trajectories ─────────────────────────────────────────────────────────
traj_equal   = ensemble_trajectory(df_pivot, equal_weights)
traj_knutti  = ensemble_trajectory(df_pivot, wt["weights"])
models_list  = wt["models"]

# ── End-of-century (2071–2100) per-model means ───────────────────────────────
eoc_values = {
    m: df_pivot.loc[EOC_START:EOC_END, m].mean()
    for m in models_list
}
eoc_arr = np.array([eoc_values[m] for m in models_list])

eoc_equal_mean   = weighted_mean(eoc_arr, equal_weights)
eoc_knutti_mean  = weighted_mean(eoc_arr, wt["weights"])
eoc_equal_med    = weighted_median(eoc_arr, equal_weights)
eoc_knutti_med   = weighted_median(eoc_arr, wt["weights"])
eoc_equal_lo     = weighted_percentile(eoc_arr, equal_weights,  17)
eoc_equal_hi     = weighted_percentile(eoc_arr, equal_weights,  83)
eoc_knutti_lo    = weighted_percentile(eoc_arr, wt["weights"],  17)
eoc_knutti_hi    = weighted_percentile(eoc_arr, wt["weights"],  83)

print(f"\nEnd-of-century warming ({EOC_START}–{EOC_END}):")
print(f"  Equal-weight mean:    {eoc_equal_mean:.2f}degC"
      f"  (17–83%: {eoc_equal_lo:.2f}–{eoc_equal_hi:.2f}°C)")
print(f"  Perf.+Indep. mean:   {eoc_knutti_mean:.2f}degC"
      f"  (17–83%: {eoc_knutti_lo:.2f}–{eoc_knutti_hi:.2f}°C)")

# =============================================================================
# 5. FIGURE 4 – Historical performance (RMSE ranking)
# =============================================================================
# Lollipop chart: model RMSE to ERA5 over calibration period (1940–2014)
# Lower RMSE → better performance → higher weight

fig, ax = plt.subplots(figsize=(8, 5))

sort_idx   = np.argsort(wt["D"])
d_sorted   = wt["D"][sort_idx]
m_sorted   = [models_list[i] for i in sort_idx]
w_sorted   = wt["weights"][sort_idx]

# Color by final weight
norm  = mcolors.Normalize(vmin=w_sorted.min(), vmax=w_sorted.max())
cmap  = cm.RdYlGn                      # red=low weight, green=high weight
colors_lollipop = [cmap(norm(w)) for w in w_sorted]

for i, (m, d, c) in enumerate(zip(m_sorted, d_sorted, colors_lollipop)):
    ax.hlines(i, 0, d, colors="lightgray", linewidth=1.2)
    ax.scatter(d, i, color=c, s=80, zorder=3)

ax.set_yticks(range(N_MODELS))
ax.set_yticklabels(m_sorted, fontsize=8)
ax.set_xlabel("RMSE to ERA5 anomaly (°C)  [calibration period 1940–2014]")
ax.set_title("Historical performance: RMSE to ERA5\n"
             "(simplified GSAT-only metric; color = final Perf.+Indep. weight)")
ax.axvline(wt["sigma_D"], color="steelblue", linestyle="--", linewidth=1.0,
           label=f"sigma_D = {wt['sigma_D']:.3f}°C")
ax.legend(frameon=False, fontsize=9)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Final Perf.+Indep. weight", shrink=0.6, pad=0.02)

fig.savefig(os.path.join(OUT_DIR, "fig04_historical_performance.png"),
            bbox_inches="tight")
plt.close(fig)
print("Saved fig04_historical_performance.png")

# =============================================================================
# 6. FIGURE 5 – Redesigned interdependence heatmap (presentation-ready)
# =============================================================================
# Goal: the audience should immediately see that some models form tight
# similarity clusters, making equal weighting problematic.
#
# Redesign vs. original:
#   (a) Models reordered by Ward hierarchical clustering → block-diagonal
#       structure makes clusters pop out instantly.
#   (b) All 400 cell annotations removed; only the most similar pairs
#       (bottom 15th percentile of pairwise RMSE) are annotated in the
#       lower triangle in white bold text. Much less visual noise.
#   (c) Colored rectangles outline each cluster so the eye goes straight
#       to the groups.
#   (d) Tick labels are coloured by cluster for immediate group recognition.
#   (e) A dashed marker on the colorbar shows sigma_S — the scale below
#       which models are treated as strongly redundant by the weighting.
#
# FIGURE 5b (backup): ranked bar chart of the top-12 most similar pairs.
#   More direct than the matrix and useful as a backup slide.
# =============================================================================

S_mat = wt["S_matrix"].copy()        # (N_MODELS × N_MODELS) symmetric RMSE matrix
N = N_MODELS

# ── 1. Cluster models by pairwise RMSE using Ward's method ───────────────────
# Ward minimises total within-cluster variance; produces compact clusters.
S_cond   = squareform(S_mat)         # condensed upper-triangle distance vector
Z_link   = hierarchy.linkage(S_cond, method="ward")

# Leaf ordering: models in the same subtree are placed adjacent so that
# the resulting heatmap shows a clear block-diagonal structure.
dendro_info = hierarchy.dendrogram(Z_link, no_plot=True)
leaf_order  = dendro_info["leaves"]  # reordered model indices
models_ord  = [models_list[i] for i in leaf_order]
S_ord       = S_mat[np.ix_(leaf_order, leaf_order)]  # reordered matrix

# ── 2. Identify clusters ──────────────────────────────────────────────────────
# Cut the dendrogram at 55 % of the maximum merge height; typically gives
# 4–6 clusters for 20 CMIP6 models.
cut_height = 0.55 * Z_link[-1, 2]
clust_raw  = hierarchy.fcluster(Z_link, t=cut_height, criterion="distance")
clust_ord  = clust_raw[leaf_order]   # cluster label per model in reordered list

# Find contiguous spans [start, end) in the reordered sequence.
# Because the dendrogram ordering puts each subtree together, same-cluster
# models will (almost always) be adjacent.
spans = []
i = 0
while i < N:
    start = i
    while i < N and clust_ord[i] == clust_ord[start]:
        i += 1
    spans.append((start, i))

N_CLUST        = len(spans)
CLUSTER_COLORS = ["#2980B9", "#E74C3C", "#27AE60", "#F39C12",
                  "#8E44AD", "#16A085", "#E67E22"]

# ── 3. Colour scale parameters ────────────────────────────────────────────────
off_diag     = S_ord[~np.eye(N, dtype=bool)]
VMIN         = 0.0
VMAX         = np.percentile(off_diag, 95)   # compress range to emphasise similarities
annot_thresh = np.percentile(off_diag, 15)   # annotate only the most similar 15 %

# ── 4. Main heatmap ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 9))

S_df = pd.DataFrame(S_ord, index=models_ord, columns=models_ord)

sns.heatmap(
    S_df, ax=ax,
    mask=np.eye(N, dtype=bool),          # hide uninformative diagonal
    cmap="YlOrRd_r",                     # red = low RMSE = similar; fades to yellow
    vmin=VMIN, vmax=VMAX,
    linewidths=0.3, linecolor="#e8e8e8",
    cbar_kws={
        "label":  "Pairwise RMSE (°C)   ←   lower = more similar / redundant",
        "shrink": 0.55, "aspect": 25, "pad": 0.02,
    },
    annot=False,   # annotations are added manually below (selective only)
    square=True,
)

# ── 5. Selective annotations: only the most similar pairs ────────────────────
# Annotate only the lower triangle so each pair is labelled once.
for row in range(N):
    for col in range(row):                # strict lower triangle
        val = S_ord[row, col]
        if val <= annot_thresh:
            ax.text(col + 0.5, row + 0.5, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold")

# ── 6. Cluster boundary rectangles ───────────────────────────────────────────
for idx, (start, end) in enumerate(spans):
    size  = end - start
    color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
    ax.add_patch(Rectangle(
        (start, start), size, size,
        fill=False, edgecolor=color, linewidth=3,
        linestyle="-", zorder=10,
    ))

# ── 7. Colour tick labels by cluster ─────────────────────────────────────────
# Build index → cluster-color lookup
idx_color = {}
for idx, (start, end) in enumerate(spans):
    for k in range(start, end):
        idx_color[k] = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7.5)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=7.5)
for i, (xl, yl) in enumerate(zip(ax.get_xticklabels(), ax.get_yticklabels())):
    c = idx_color.get(i, "black")
    xl.set_color(c); xl.set_fontweight("bold")
    yl.set_color(c); yl.set_fontweight("bold")

# ── 8. Mark sigma_S on the colorbar ──────────────────────────────────────────
# Models with pairwise RMSE < sigma_S are treated as strongly redundant.
sigma_S_val = wt["sigma_S"]
cbar = ax.collections[0].colorbar
if VMIN <= sigma_S_val <= VMAX:
    # colorbar y-axis runs from VMIN (bottom) to VMAX (top) in data coords
    cbar.ax.axhline(y=sigma_S_val, color="black",
                    linewidth=1.5, linestyle="--", zorder=5)
    cbar.ax.text(
        1.08, (sigma_S_val - VMIN) / (VMAX - VMIN),
        f"  \u03c3_S={sigma_S_val:.3f}\u00b0C",
        va="center", ha="left", fontsize=7.5,
        transform=cbar.ax.transAxes,
    )

# ── 9. Cluster legend ─────────────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], marker="s", color="w",
           markerfacecolor=CLUSTER_COLORS[k % len(CLUSTER_COLORS)],
           markersize=11,
           label=f"Cluster {k + 1}  ({spans[k][1] - spans[k][0]} models)")
    for k in range(N_CLUST)
]
ax.legend(handles=legend_handles,
          loc="upper right", bbox_to_anchor=(1.01, 1.01),
          frameon=True, framealpha=0.92,
          fontsize=7.5, title="Similarity clusters", title_fontsize=8)

# ── 10. Title and caption ─────────────────────────────────────────────────────
ax.set_title(
    "Model interdependence: pairwise RMSE between ensemble-mean trajectories\n"
    "Models reordered by similarity clustering — boxes show groups of structurally similar models",
    fontsize=10.5, pad=14,
)
fig.text(
    0.12, 0.005,
    "Models in the same box follow similar warming trajectories "
    "\u2192 equal weighting gives their shared characteristics disproportionate influence.",
    fontsize=8.5, color="0.35", style="italic",
)

fig.savefig(os.path.join(OUT_DIR, "fig05_interdependence_heatmap.png"),
            bbox_inches="tight")
plt.close(fig)
print("Saved fig05_interdependence_heatmap.png  (redesigned: clustered, selective annotations)")


# =============================================================================
# 6b. FIGURE 5b – Backup: ranked bar chart of the most similar model pairs
# =============================================================================
# Even more direct than the matrix: immediately shows which pairs are most
# redundant and whether they fall below the sigma_S redundancy threshold.
# Use this as a backup slide or to support a verbal explanation.
# =============================================================================

# Collect every unique off-diagonal pair (upper triangle only)
all_pairs = []
for i in range(N):
    for j in range(i + 1, N):
        all_pairs.append((S_mat[i, j], models_list[i], models_list[j]))
all_pairs.sort()   # ascending RMSE → most similar first

TOP_N      = 12
top_pairs  = all_pairs[:TOP_N]
rmse_vals  = [p[0] for p in top_pairs]
pair_labels = [f"{p[1]}  \u2194  {p[2]}" for p in top_pairs]

# Determine whether each pair falls within the same cluster
same_clust = []
for _, m1, m2 in top_pairs:
    pos1 = leaf_order.index(models_list.index(m1))
    pos2 = leaf_order.index(models_list.index(m2))
    same_clust.append(clust_ord[pos1] == clust_ord[pos2])

bar_colors = ["#E74C3C" if sc else "#95A5A6" for sc in same_clust]

fig_b, ax_b = plt.subplots(figsize=(9, 5.5))
y_pos = list(range(TOP_N - 1, -1, -1))   # reversed so most similar is at top

ax_b.barh(y_pos, rmse_vals,
          color=bar_colors, height=0.65, edgecolor="white")

for y, val in zip(y_pos, rmse_vals):
    ax_b.text(val + 0.001, y, f" {val:.3f}\u00b0C", va="center", fontsize=8.5)

ax_b.set_yticks(y_pos)
ax_b.set_yticklabels(pair_labels, fontsize=8)
ax_b.set_xlabel("Pairwise RMSE (°C)   \u2190   lower = more similar / redundant",
                fontsize=9)
ax_b.axvline(wt["sigma_S"], color="black", linestyle="--", linewidth=1.5)
ax_b.text(wt["sigma_S"] + 0.001, TOP_N - 0.6,
          f"\u03c3_S = {wt['sigma_S']:.3f}\u00b0C\n(redundancy threshold)",
          fontsize=8, va="top", color="black")

legend_handles_b = [
    Rectangle((0, 0), 1, 1, facecolor="#E74C3C", label="Same similarity cluster"),
    Rectangle((0, 0), 1, 1, facecolor="#95A5A6", label="Different clusters"),
]
ax_b.legend(handles=legend_handles_b, frameon=False, fontsize=8.5, loc="lower right")

ax_b.set_xlim(0, max(rmse_vals) * 1.18)
ax_b.set_title(
    f"Top {TOP_N} most similar model pairs\n"
    "Pairs below \u03c3_S are penalized by the independence weight",
    fontsize=10,
)

fig_b.savefig(os.path.join(OUT_DIR, "fig05b_similar_pairs.png"),
              bbox_inches="tight")
plt.close(fig_b)
print("Saved fig05b_similar_pairs.png  (backup: top-N similar pairs)")

# =============================================================================
# 7. FIGURE 6 – Weight decomposition
# =============================================================================
# For each model: performance term, independence term, final weight

fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)

x = np.arange(N_MODELS)
bar_w = 0.6

# Sort by final weight descending for readability
sort_idx2 = np.argsort(wt["weights"])[::-1]
ms = [models_list[i] for i in sort_idx2]

perf_s   = wt["w_perf"][sort_idx2]
indep_s  = wt["w_indep"][sort_idx2]
final_s  = wt["weights"][sort_idx2]

palette = ["#4C9BE8", "#F5A623", "#2ECC71"]

axes[0].barh(x, perf_s,  height=bar_w, color=palette[0])
axes[0].set_title("Performance term\n(normalized exp(-D²/sigma_D²))")
axes[0].axvline(1/N_MODELS, color="gray", linestyle="--",
                linewidth=1, label="Equal weight")
axes[0].legend(frameon=False, fontsize=8)

axes[1].barh(x, indep_s, height=bar_w, color=palette[1])
axes[1].set_title("Independence term\n(normalized 1/(1+redundancy))")
axes[1].axvline(1/N_MODELS, color="gray", linestyle="--", linewidth=1)

axes[2].barh(x, final_s, height=bar_w, color=palette[2])
axes[2].set_title("Final Perf.+Indep. weight\n(product, normalized)")
axes[2].axvline(1/N_MODELS, color="gray", linestyle="--", linewidth=1,
                label=f"Equal (1/{N_MODELS})")
axes[2].legend(frameon=False, fontsize=8)

for ax in axes:
    ax.set_yticks(x)
    ax.set_yticklabels(ms, fontsize=7)
    ax.set_xlabel("Normalized weight")

fig.suptitle("Performance+interdependence weight decomposition\n"
             "(GSAT-only; performance × independence)", y=1.01)

fig.savefig(os.path.join(OUT_DIR, "fig06_weight_decomposition.png"),
            bbox_inches="tight")
plt.close(fig)
print("Saved fig06_weight_decomposition.png")

# =============================================================================
# 8. FIGURE 7 – Performance vs independence scatter
# =============================================================================
# x = RMSE to ERA5 (lower = better)
# y = independence term (higher = more independent)
# bubble size & color = final weight

fig, ax = plt.subplots(figsize=(8, 6))

indep_raw_norm = wt["w_indep"]   # already normalized for display

# Bubble size proportional to final weight
sizes = wt["weights"] / wt["weights"].max() * 600 + 50

norm2  = mcolors.Normalize(vmin=wt["weights"].min(), vmax=wt["weights"].max())
cmap2  = cm.viridis

sc = ax.scatter(wt["D"], indep_raw_norm, s=sizes,
                c=wt["weights"], cmap=cmap2, norm=norm2,
                edgecolors="white", linewidths=0.5, alpha=0.85, zorder=3)

# Label each model
for i, m in enumerate(models_list):
    ax.annotate(m, (wt["D"][i], indep_raw_norm[i]),
                fontsize=6.5, textcoords="offset points", xytext=(4, 3),
                color="0.3")

ax.set_xlabel("RMSE to ERA5 (°C)  ← better performance")
ax.set_ylabel("Independence term (normalized)  ← less redundancy →")
ax.set_title("Performance vs independence\n"
             "(bubble size & color = final Perf.+Indep. weight)")
ax.invert_xaxis()   # lower RMSE = better → right side preferred

plt.colorbar(sc, ax=ax, label="Final Perf.+Indep. weight", shrink=0.7)

# Reference lines at equal-weight level
ax.axhline(1/N_MODELS, color="gray", linestyle=":", linewidth=1,
           label=f"Equal level (1/{N_MODELS})")
ax.legend(frameon=False, fontsize=9)

fig.savefig(os.path.join(OUT_DIR, "fig07_performance_vs_independence.png"),
            bbox_inches="tight")
plt.close(fig)
print("Saved fig07_performance_vs_independence.png")

# =============================================================================
# 9. FIGURE 1 – Historical + future trajectory (1940–2100)
# =============================================================================
# ERA5 historical anomalies + equal-weight mean + Knutti-weighted mean
# Individual model shading in background

fig, ax = plt.subplots(figsize=(11, 5))

# ── Background: individual model ensemble-mean trajectories ─────────────────
for m in models_list:
    ax.plot(df_pivot.index, df_pivot[m], color="lightgray",
            linewidth=0.6, alpha=0.5)

# ── Shaded spread: equal-weight 17–83 percentile range ──────────────────────
years_full = df_pivot.index
lo_band = np.array([weighted_percentile(df_pivot.loc[y].values, equal_weights, 17)
                    for y in years_full])
hi_band = np.array([weighted_percentile(df_pivot.loc[y].values, equal_weights, 83)
                    for y in years_full])
ax.fill_between(years_full, lo_band, hi_band,
                alpha=0.15, color="#4C9BE8",
                label="Equal-weight 17–83% spread\n(not a formal CI)")

# ── Equal-weight multi-model mean ────────────────────────────────────────────
ax.plot(traj_equal.index, traj_equal.values,
        color="#4C9BE8", linewidth=2.0, label="Equal-weight MMM")

# ── Knutti-weighted mean ─────────────────────────────────────────────────────
ax.plot(traj_knutti.index, traj_knutti.values,
        color="#E84C4C", linewidth=2.0, label="Perf.+Indep.-weighted mean")

# ── ERA5 observations ────────────────────────────────────────────────────────
ax.plot(era5_series.index, era5_series.values,
        color="black", linewidth=1.8, label="ERA5 reanalysis")

# Vertical line at end of calibration period
ax.axvline(2014, color="gray", linestyle="--", linewidth=1.0, alpha=0.7,
           label="End of CMIP6 historical (2014)")
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

ax.set_xlabel("Year")
ax.set_ylabel("GSAT anomaly (°C, rel. to 1981–2014)")
ax.set_title("CMIP6 GSAT projections under SSP2-4.5\n"
             "Equal-weight vs performance+interdependence weighting")
ax.set_xlim(1940, 2100)
ax.legend(loc="upper left", frameon=False, fontsize=8.5)

# Annotation: spread is weighted model spread, not probabilistic interval
ax.text(0.99, 0.02,
        "Shading = weighted model spread\n(not a formal probability interval)",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.5, color="0.5", style="italic")

fig.savefig(os.path.join(OUT_DIR, "fig01_historical_future_trajectory.png"),
            bbox_inches="tight")
plt.close(fig)
print("Saved fig01_historical_future_trajectory.png")

# =============================================================================
# 10. FIGURE 2 – Future-only trajectory (2025–2100)
# =============================================================================

fig, ax = plt.subplots(figsize=(9, 5))

future_years = [y for y in df_pivot.index if y >= FUTURE_START]
lo_fut = np.array([weighted_percentile(df_pivot.loc[y].values, equal_weights, 17)
                   for y in future_years])
hi_fut = np.array([weighted_percentile(df_pivot.loc[y].values, equal_weights, 83)
                   for y in future_years])
lo_knut_fut = np.array([weighted_percentile(df_pivot.loc[y].values, wt["weights"], 17)
                        for y in future_years])
hi_knut_fut = np.array([weighted_percentile(df_pivot.loc[y].values, wt["weights"], 83)
                        for y in future_years])

# Spread bands
ax.fill_between(future_years, lo_fut, hi_fut,
                alpha=0.18, color="#4C9BE8",
                label="Equal 17–83% spread")
ax.fill_between(future_years, lo_knut_fut, hi_knut_fut,
                alpha=0.18, color="#E84C4C",
                label="Perf.+Indep. 17–83% spread")

# Individual model lines (faint)
for m in models_list:
    ax.plot(future_years, df_pivot.loc[future_years, m],
            color="lightgray", linewidth=0.5, alpha=0.4)

# Means
eq_fut   = traj_equal.loc[future_years]
kn_fut   = traj_knutti.loc[future_years]
ax.plot(future_years, eq_fut.values,  color="#4C9BE8", linewidth=2.2,
        label=f"Equal-weight mean")
ax.plot(future_years, kn_fut.values,  color="#E84C4C", linewidth=2.2,
        label="Perf.+Indep.-weighted mean")

ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.set_xlabel("Year")
ax.set_ylabel("GSAT anomaly (°C, rel. to 1981–2014)")
ax.set_title(f"Future GSAT projections {FUTURE_START}–2100 under SSP2-4.5\n"
             "Equal-weight vs performance+interdependence weighting")
ax.set_xlim(FUTURE_START, 2100)
ax.legend(loc="upper left", frameon=False, fontsize=9)

ax.text(0.99, 0.02,
        "Shading = weighted model spread\n(not a formal probability interval)",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.5, color="0.5", style="italic")

fig.savefig(os.path.join(OUT_DIR, "fig02_future_trajectory.png"),
            bbox_inches="tight")
plt.close(fig)
print("Saved fig02_future_trajectory.png")

# =============================================================================
# 11. FIGURE 3 – End-of-century summary (2071–2100)
# =============================================================================

fig, ax = plt.subplots(figsize=(7, 5))

# ── Individual model dots ──────────────────────────────────────────────────
x_equal  = 0.85
x_knutti = 2.15
jitter   = 0.06 * np.random.RandomState(42).randn(N_MODELS)

# Bubble size by weight
siz_eq  = equal_weights / equal_weights.max() * 120 + 20
siz_kn  = wt["weights"] / wt["weights"].max() * 120 + 20

ax.scatter(x_equal  + jitter, eoc_arr, s=siz_eq,
           color="#4C9BE8", alpha=0.6, edgecolors="white", linewidths=0.5, zorder=3)
ax.scatter(x_knutti + jitter, eoc_arr, s=siz_kn,
           color="#E84C4C", alpha=0.6, edgecolors="white", linewidths=0.5, zorder=3)

# ── 17–83 % range bars ────────────────────────────────────────────────────
def draw_range_bar(ax, x, lo, hi, mean_val, med_val, color, label_prefix):
    ax.vlines(x, lo, hi, linewidth=3, color=color, alpha=0.7,
              label=f"{label_prefix} 17–83% spread")
    ax.hlines(lo, x-0.06, x+0.06, linewidth=2, color=color, alpha=0.7)
    ax.hlines(hi, x-0.06, x+0.06, linewidth=2, color=color, alpha=0.7)
    ax.scatter(x, mean_val, marker="D", s=80, color=color, zorder=5,
               label=f"{label_prefix} weighted mean ({mean_val:.2f}°C)")
    ax.scatter(x, med_val, marker="_", s=200, color=color, linewidths=3,
               zorder=5, label=f"{label_prefix} weighted median ({med_val:.2f}°C)")

draw_range_bar(ax, x_equal,  eoc_equal_lo,  eoc_equal_hi,
               eoc_equal_mean, eoc_equal_med,  "#2979B8", "Equal-weight")
draw_range_bar(ax, x_knutti, eoc_knutti_lo, eoc_knutti_hi,
               eoc_knutti_mean, eoc_knutti_med, "#C0392B", "Perf.+Indep.")

ax.set_xticks([x_equal, x_knutti])
ax.set_xticklabels(["Equal weighting", "Perf.+Indep. weighting"], fontsize=11)
ax.set_ylabel("GSAT anomaly (°C, rel. to 1981–2014)")
ax.set_title(f"End-of-century warming {EOC_START}–{EOC_END}\n"
             "Model spread and weighted statistics")
ax.set_xlim(0.3, 2.7)

# Deduplicate legend
handles, labels = ax.get_legend_handles_labels()
seen = {}
for h, l in zip(handles, labels):
    if l not in seen:
        seen[l] = h
ax.legend(seen.values(), seen.keys(), frameon=False, fontsize=8,
          loc="upper left", bbox_to_anchor=(0.0, 0.99))

ax.text(0.99, 0.01,
        "Spread = weighted model spread\n(not a formal probability interval)\n"
        "Diamond = weighted mean  |  dash = weighted median",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.5, color="0.5", style="italic")

fig.savefig(os.path.join(OUT_DIR, "fig03_end_of_century_summary.png"),
            bbox_inches="tight")
plt.close(fig)
print("Saved fig03_end_of_century_summary.png")

# =============================================================================
# 12. VALIDATION – Hindcast (split: train 1940–1989, test 1990–2014)
# =============================================================================
# We calibrate weights using model performance over 1940–1989,
# then evaluate both equal and Knutti weights on 1990–2014.
# All periods are within the CMIP6 historical experiment (<= 2014).

print("\n--- Hindcast validation ---")

# Compute Knutti weights calibrated on training period only
wt_hindcast = compute_knutti_weights(
    df_pivot, era5_series,
    HINDCAST_TRAIN_START, HINDCAST_TRAIN_END
)

# Test period ERA5 and model trajectories
test_years = [y for y in df_pivot.index
              if HINDCAST_TEST_START <= y <= HINDCAST_TEST_END
              and y in era5_series.index]
era5_test  = era5_series.loc[test_years].values

# Weighted means over test period
equal_test_traj  = (df_pivot.loc[test_years].values @ equal_weights)
knutti_test_traj = (df_pivot.loc[test_years].values @ wt_hindcast["weights"])

def trend_slope(y_arr, t_arr):
    slope, _, _, _, _ = stats.linregress(t_arr, y_arr)
    return slope

t_arr = np.array(test_years, dtype=float)

metrics_hindcast = {}
for label, traj in [("Equal-weight", equal_test_traj),
                    ("Perf.+Indep.", knutti_test_traj)]:
    rmse = compute_rmse(traj, era5_test)
    mae  = np.mean(np.abs(traj - era5_test))
    r, _ = pearsonr(traj, era5_test)
    trend_mod = trend_slope(traj, t_arr)
    trend_obs = trend_slope(era5_test, t_arr)
    trend_err = abs(trend_mod - trend_obs)
    metrics_hindcast[label] = dict(RMSE=rmse, MAE=mae, Correlation=r,
                                   Trend_error=trend_err)
    print(f"  {label}: RMSE={rmse:.4f}  MAE={mae:.4f}  "
          f"Corr={r:.4f}  TrendErr={trend_err:.5f} °C/yr")

# ── Figure: hindcast validation bar chart ────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(12, 4))

metric_labels = ["RMSE", "MAE", "Correlation", "Trend_error"]
y_labels      = ["RMSE (°C)", "MAE (°C)", "Pearson r",
                 "Trend error (°C/yr)"]
titles        = [f"RMSE to ERA5\n(test: {HINDCAST_TEST_START}–{HINDCAST_TEST_END})",
                 f"MAE to ERA5\n(test: {HINDCAST_TEST_START}–{HINDCAST_TEST_END})",
                 f"Correlation with ERA5\n(test: {HINDCAST_TEST_START}–{HINDCAST_TEST_END})",
                 f"Trend error\n(test: {HINDCAST_TEST_START}–{HINDCAST_TEST_END})"]
colors_v      = ["#4C9BE8", "#E84C4C"]

for ax_i, (metric, ylabel, title) in enumerate(
        zip(metric_labels, y_labels, titles)):
    vals = [metrics_hindcast["Equal-weight"][metric],
            metrics_hindcast["Perf.+Indep."][metric]]
    bars = axes[ax_i].bar(["Equal", "Perf.+Indep."], vals,
                          color=colors_v, width=0.5, edgecolor="white")
    axes[ax_i].set_title(title, fontsize=9)
    axes[ax_i].set_ylabel(ylabel, fontsize=8)
    # Annotate bar values
    for bar, val in zip(bars, vals):
        axes[ax_i].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    if metric == "Correlation":
        axes[ax_i].set_ylim(0.9, 1.02)

fig.suptitle(
    f"Hindcast validation: weights calibrated on {HINDCAST_TRAIN_START}–{HINDCAST_TRAIN_END},\n"
    f"evaluated on {HINDCAST_TEST_START}–{HINDCAST_TEST_END} (within CMIP6 historical)",
    fontsize=11, y=1.02
)

fig.savefig(os.path.join(OUT_DIR, "fig08_hindcast_validation.png"),
            bbox_inches="tight")
plt.close(fig)
print("Saved fig08_hindcast_validation.png")

# ── Figure: hindcast trajectory overlay ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(test_years, era5_test, color="black", linewidth=2.0, label="ERA5")
ax.plot(test_years, equal_test_traj, color="#4C9BE8", linewidth=1.8,
        linestyle="--", label="Equal-weight (hindcast)")
ax.plot(test_years, knutti_test_traj, color="#E84C4C", linewidth=1.8,
        linestyle="-", label="Perf.+Indep. (hindcast)")
ax.set_xlabel("Year")
ax.set_ylabel("GSAT anomaly (°C, rel. to 1981–2014)")
ax.set_title(f"Hindcast: out-of-sample trajectories vs ERA5\n"
             f"(calibrated on {HINDCAST_TRAIN_START}–{HINDCAST_TRAIN_END})")
ax.legend(frameon=False)

fig.savefig(os.path.join(OUT_DIR, "fig08b_hindcast_trajectory.png"),
            bbox_inches="tight")
plt.close(fig)
print("Saved fig08b_hindcast_trajectory.png")

# =============================================================================
# 13. VALIDATION – Perfect-model test (pseudo-observation cross-validation)
# =============================================================================
# For each model m as "pseudo-truth":
#   1. Use the other (N-1) models to compute Knutti weights against pseudo-truth
#      over training period (1940–1989).
#   2. Predict the pseudo-truth over test period (1990–2014) using
#      both equal and Knutti weighted mean of remaining models.
#   3. Compute RMSE(weighted mean, pseudo-truth) for both weightings.
# Summary: distribution of RMSE ratios (Knutti / Equal) across all pseudo-truths.

print("\n--- Perfect-model test ---")

pm_rmse_equal  = []
pm_rmse_knutti = []
pm_labels      = []

for pseudo_model in models_list:
    # other models
    other_models = [m for m in models_list if m != pseudo_model]
    pseudo_series = df_pivot[pseudo_model]   # pseudo-truth time series

    # Sub-pivot without pseudo-truth model
    df_sub = df_pivot[other_models]

    # Equal weights over (N-1) models
    eq_w_sub = np.ones(len(other_models)) / len(other_models)

    # Knutti weights calibrated on training period against pseudo-truth
    wt_pm = compute_knutti_weights(
        df_sub, pseudo_series,
        HINDCAST_TRAIN_START, HINDCAST_TRAIN_END
    )

    # Test period
    test_years_pm = [y for y in df_sub.index
                     if HINDCAST_TEST_START <= y <= HINDCAST_TEST_END]
    pseudo_test   = pseudo_series.loc[test_years_pm].values
    eq_test_traj  = (df_sub.loc[test_years_pm].values @ eq_w_sub)
    kn_test_traj  = (df_sub.loc[test_years_pm].values @ wt_pm["weights"])

    rmse_eq = compute_rmse(eq_test_traj, pseudo_test)
    rmse_kn = compute_rmse(kn_test_traj, pseudo_test)

    pm_rmse_equal.append(rmse_eq)
    pm_rmse_knutti.append(rmse_kn)
    pm_labels.append(pseudo_model)
    print(f"  Pseudo-truth={pseudo_model:<25s}  RMSE_equal={rmse_eq:.4f}  "
          f"RMSE_knutti={rmse_kn:.4f}  ratio={rmse_kn/rmse_eq:.3f}")

pm_ratio = np.array(pm_rmse_knutti) / np.array(pm_rmse_equal)
print(f"\nMedian RMSE ratio (Perf.+Indep./Equal): {np.median(pm_ratio):.3f}  "
      f"(< 1 = Perf.+Indep. better, > 1 = equal better)")

# ── Figure: perfect-model RMSE comparison ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: RMSE per pseudo-truth, grouped bars
x = np.arange(N_MODELS)
w = 0.35
axes[0].bar(x - w/2, pm_rmse_equal,  width=w, color="#4C9BE8",
            label="Equal-weight")
axes[0].bar(x + w/2, pm_rmse_knutti, width=w, color="#E84C4C",
            label="Perf.+Indep.")
axes[0].set_xticks(x)
axes[0].set_xticklabels(pm_labels, rotation=45, ha="right", fontsize=7)
axes[0].set_ylabel("RMSE to pseudo-truth (°C)")
axes[0].set_title(f"Perfect-model RMSE per pseudo-truth\n"
                  f"(test period {HINDCAST_TEST_START}–{HINDCAST_TEST_END})")
axes[0].legend(frameon=False)
axes[0].axhline(np.mean(pm_rmse_equal), color="#4C9BE8", linestyle="--",
                linewidth=1, alpha=0.6)
axes[0].axhline(np.mean(pm_rmse_knutti), color="#E84C4C", linestyle="--",
                linewidth=1, alpha=0.6)

# Right: ratio distribution
colors_ratio = ["#2ECC71" if r < 1 else "#E84C4C" for r in pm_ratio]
axes[1].barh(x, pm_ratio, color=colors_ratio, edgecolor="white")
axes[1].axvline(1.0, color="black", linestyle="--", linewidth=1.2,
                label="Ratio = 1  (equal performance)")
axes[1].set_yticks(x)
axes[1].set_yticklabels(pm_labels, fontsize=7)
axes[1].set_xlabel("RMSE ratio: Perf.+Indep. / Equal\n(green < 1 = Perf.+Indep. better)")
axes[1].set_title("RMSE ratio across pseudo-truths\n"
                  "(< 1 favors Perf.+Indep. weighting)")
axes[1].legend(frameon=False, fontsize=8)
axes[1].text(0.98, 0.02, f"Median ratio = {np.median(pm_ratio):.3f}",
             transform=axes[1].transAxes, ha="right", va="bottom",
             fontsize=9, color="0.3")

fig.suptitle("Perfect-model (pseudo-observation) cross-validation\n"
             "Simplified GSAT-only performance+interdependence weighting", y=1.01)

fig.savefig(os.path.join(OUT_DIR, "fig09_perfect_model_test.png"),
            bbox_inches="tight")
plt.close(fig)
print("Saved fig09_perfect_model_test.png")

# =============================================================================
# 14. SENSITIVITY – Sweep sigma_D and sigma_S
# =============================================================================
# Show how end-of-century weighted mean warming and hindcast RMSE change
# as sigma_D and sigma_S vary around their defaults.

print("\n--- Sensitivity analysis ---")

n_grid = 12
sigma_D_vals = np.linspace(wt["sigma_D"] * 0.2, wt["sigma_D"] * 3.0, n_grid)
sigma_S_vals = np.linspace(wt["sigma_S"] * 0.2, wt["sigma_S"] * 3.0, n_grid)

eoc_grid       = np.zeros((n_grid, n_grid))
hindcast_grid  = np.zeros((n_grid, n_grid))

for i, sD in enumerate(sigma_D_vals):
    for j, sS in enumerate(sigma_S_vals):
        wt_s = compute_knutti_weights(
            df_pivot, era5_series,
            CALIB_START, CALIB_END,
            sigma_D=sD, sigma_S=sS
        )
        # EOC warming
        eoc_arr_s = np.array([df_pivot.loc[EOC_START:EOC_END, m].mean()
                               for m in wt_s["models"]])
        eoc_grid[i, j] = weighted_mean(eoc_arr_s, wt_s["weights"])

        # Hindcast RMSE (calibrate weights on train, evaluate on test)
        wt_hc = compute_knutti_weights(
            df_pivot, era5_series,
            HINDCAST_TRAIN_START, HINDCAST_TRAIN_END,
            sigma_D=sD, sigma_S=sS
        )
        kn_test = (df_pivot.loc[test_years].values @ wt_hc["weights"])
        hindcast_grid[i, j] = compute_rmse(kn_test, era5_test)

# ── Figure: dual sensitivity heatmap ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

def sensitivity_heatmap(ax, grid, sD_vals, sS_vals, title, cbar_label,
                        default_sD, default_sS, fmt=".2f"):
    sD_labels = [f"{v:.3f}" for v in sD_vals]
    sS_labels = [f"{v:.3f}" for v in sS_vals]
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r" if "RMSE" in cbar_label
                   else "coolwarm", origin="lower")
    ax.set_xticks(range(len(sS_vals)))
    ax.set_xticklabels(sS_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(sD_vals)))
    ax.set_yticklabels(sD_labels, fontsize=7)
    ax.set_xlabel("sigma_S (independence scale)")
    ax.set_ylabel("sigma_D (performance scale)")
    ax.set_title(title)

    # Mark default
    def nearest_idx(vals, target):
        return int(np.argmin(np.abs(vals - target)))
    iD = nearest_idx(sD_vals, default_sD)
    iS = nearest_idx(sS_vals, default_sS)
    ax.scatter(iS, iD, marker="*", color="black", s=200, zorder=5,
               label=f"Default (sigma_D={default_sD:.3f}, sigma_S={default_sS:.3f})")
    ax.legend(fontsize=7, frameon=False)
    plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)

sensitivity_heatmap(
    axes[0], eoc_grid, sigma_D_vals, sigma_S_vals,
    f"End-of-century Perf.+Indep.-weighted warming\n({EOC_START}–{EOC_END})",
    "Weighted mean warming (°C)", wt["sigma_D"], wt["sigma_S"]
)
sensitivity_heatmap(
    axes[1], hindcast_grid, sigma_D_vals, sigma_S_vals,
    f"Hindcast RMSE (Perf.+Indep. weights)\n(test: {HINDCAST_TEST_START}–{HINDCAST_TEST_END})",
    "RMSE to ERA5 (°C)", wt["sigma_D"], wt["sigma_S"]
)

fig.suptitle("Sensitivity of results to sigma_D and sigma_S\n"
             "(star = default median-based values)", y=1.01)

fig.savefig(os.path.join(OUT_DIR, "fig10_sensitivity_heatmap.png"),
            bbox_inches="tight")
plt.close(fig)
print("Saved fig10_sensitivity_heatmap.png")

# =============================================================================
# 15. SUMMARY TABLE – Print numerical results
# =============================================================================
print("\n" + "="*70)
print("SUMMARY OF RESULTS")
print("="*70)

print(f"\nCalibration period for final weights:  {CALIB_START}–{CALIB_END}")
print(f"Default sigma_D = {wt['sigma_D']:.4f}degC  |  sigma_S = {wt['sigma_S']:.4f}degC")

print(f"\nEnd-of-century warming ({EOC_START}–{EOC_END}, anomaly rel. 1981–2014):")
print(f"  Equal-weight:   mean={eoc_equal_mean:.2f}degC  "
      f"median={eoc_equal_med:.2f}°C  17–83%=[{eoc_equal_lo:.2f}, {eoc_equal_hi:.2f}]°C")
print(f"  Perf.+Indep.:  mean={eoc_knutti_mean:.2f}degC  "
      f"median={eoc_knutti_med:.2f}°C  17–83%=[{eoc_knutti_lo:.2f}, {eoc_knutti_hi:.2f}]°C")

print(f"\nHindcast validation (calibrated {HINDCAST_TRAIN_START}–{HINDCAST_TRAIN_END}, "
      f"tested {HINDCAST_TEST_START}–{HINDCAST_TEST_END}):")
for lbl, m in metrics_hindcast.items():
    print(f"  {lbl}:  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
          f"Corr={m['Correlation']:.4f}  TrendErr={m['Trend_error']:.5f}")

print(f"\nPerfect-model test  (RMSE ratio Perf.+Indep./Equal, test {HINDCAST_TEST_START}–{HINDCAST_TEST_END}):")
print(f"  Median ratio = {np.median(pm_ratio):.3f}  "
      f"(< 1 favors Perf.+Indep., > 1 favors equal)")
print(f"  Models where Perf.+Indep. is better: "
      f"{sum(pm_ratio < 1)}/{N_MODELS}")

print("\n" + "="*70)
print("CAVEATS (for presentation slides)")
print("="*70)
caveats = [
    "1. GSAT-only metric: performance and interdependence are based solely on global mean",
    "   temperature anomalies. Real Knutti (2017) uses multiple diagnostics.",
    "2. Pairwise RMSE approximates genealogical distance; it does not measure",
    "   structural model independence directly.",
    "3. Weighting reduces effective sample size; the spread shown is weighted model",
    "   spread, not a formal probabilistic confidence interval.",
    "4. sigma_D and sigma_S are set to median values; results are moderately sensitive to",
    "   these (see Fig 10 sensitivity heatmap).",
    "5. Hindcast and perfect-model tests are conducted within the CMIP6 historical",
    "   period (<= 2014) to avoid mixing historical and SSP2-4.5 experiments.",
    "6. ERA5 is a reanalysis, not direct observations; it carries its own",
    "   uncertainties, especially before 1980.",
    "7. The future trajectory uses SSP2-4.5 scenario; scenario uncertainty is not",
    "   quantified here.",
]
for c in caveats:
    print(c)

print(f"\nAll figures saved to: {os.path.abspath(OUT_DIR)}/")


# =============================================================================
# 16. FIG 08 BUG FIX – Regenerate fig08 with correct correlation y-axis
# =============================================================================
# The original fig08 set ylim(0.9, 1.02) for Pearson r, but both values
# (~0.864) are below 0.9, making the correlation bars invisible.
# =============================================================================

fig, axes = plt.subplots(1, 4, figsize=(12, 4))

metric_labels_v2 = ["RMSE", "MAE", "Correlation", "Trend_error"]
y_labels_v2      = ["RMSE (°C)", "MAE (°C)", "Pearson r", "Trend error (°C/yr)"]
titles_v2        = [f"RMSE to ERA5\n(test: {HINDCAST_TEST_START}–{HINDCAST_TEST_END})",
                    f"MAE to ERA5\n(test: {HINDCAST_TEST_START}–{HINDCAST_TEST_END})",
                    f"Correlation with ERA5\n(test: {HINDCAST_TEST_START}–{HINDCAST_TEST_END})",
                    f"Trend error\n(test: {HINDCAST_TEST_START}–{HINDCAST_TEST_END})"]

for ax_i, (metric, ylabel, title) in enumerate(
        zip(metric_labels_v2, y_labels_v2, titles_v2)):
    vals = [metrics_hindcast["Equal-weight"][metric],
            metrics_hindcast["Perf.+Indep."][metric]]
    bars = axes[ax_i].bar(["Equal", "Perf.+Indep."], vals,
                          color=["#4C9BE8", "#E84C4C"], width=0.5, edgecolor="white")
    axes[ax_i].set_title(title, fontsize=9)
    axes[ax_i].set_ylabel(ylabel, fontsize=8)
    for bar, val in zip(bars, vals):
        axes[ax_i].text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    if metric == "Correlation":
        # FIX: set ylim so actual values (~0.864) are visible
        margin = 0.02
        axes[ax_i].set_ylim(min(vals) - margin, max(vals) + 3 * margin)

fig.suptitle(
    f"Hindcast validation: weights calibrated on {HINDCAST_TRAIN_START}–{HINDCAST_TRAIN_END},\n"
    f"evaluated on {HINDCAST_TEST_START}–{HINDCAST_TEST_END} (within CMIP6 historical)",
    fontsize=11, y=1.02
)
fig.savefig(os.path.join(OUT_DIR, "fig08_hindcast_validation.png"),
            bbox_inches="tight")
plt.close(fig)
print("Saved fig08_hindcast_validation.png  (correlation ylim bug fixed)")


# =============================================================================
# 17. FIG 11 – Smoothed RMSE robustness
# =============================================================================
# Year-by-year RMSE can be dominated by ENSO phase alignment, not the
# forced multi-decadal signal.  We additionally compute RMSE on 5-year
# centered rolling means, which are a better proxy for the forced response
# that projections are actually targeting.
#
# Design: same hindcast split (calibrated 1940–1989, evaluated 1990–2014).
# =============================================================================

SMOOTH_WIN = 5  # years

# Build pandas Series from already-computed hindcast trajectories
ts_equal  = pd.Series(equal_test_traj,  index=test_years)
ts_knutti = pd.Series(knutti_test_traj, index=test_years)
ts_era5   = pd.Series(era5_test,        index=test_years)

# 5-year centered rolling mean
sm_equal  = ts_equal.rolling(SMOOTH_WIN, center=True, min_periods=3).mean().dropna()
sm_knutti = ts_knutti.rolling(SMOOTH_WIN, center=True, min_periods=3).mean().dropna()
sm_era5   = ts_era5.rolling(SMOOTH_WIN, center=True, min_periods=3).mean().dropna()

common_yrs = sm_equal.index.intersection(sm_era5.index)
rmse_eq_sm = compute_rmse(sm_equal.loc[common_yrs].values,  sm_era5.loc[common_yrs].values)
rmse_kn_sm = compute_rmse(sm_knutti.loc[common_yrs].values, sm_era5.loc[common_yrs].values)

rmse_eq_yr = metrics_hindcast["Equal-weight"]["RMSE"]
rmse_kn_yr = metrics_hindcast["Perf.+Indep."]["RMSE"]

pct_yr = (1 - rmse_kn_yr / rmse_eq_yr) * 100
pct_sm = (1 - rmse_kn_sm / rmse_eq_sm) * 100

print(f"\nSmoothed RMSE robustness ({SMOOTH_WIN}-yr rolling, test 1990–2014):")
print(f"  Equal  yearly={rmse_eq_yr:.4f}  smoothed={rmse_eq_sm:.4f}")
print(f"  Perf.+Indep. yearly={rmse_kn_yr:.4f}  smoothed={rmse_kn_sm:.4f}")
print(f"  Perf.+Indep. improvement:  yearly={pct_yr:.1f}%   smoothed={pct_sm:.1f}%")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# Left panel: bar chart comparing yearly vs smoothed RMSE
vals_eq = [rmse_eq_yr, rmse_eq_sm]
vals_kn = [rmse_kn_yr, rmse_kn_sm]
x_pos   = np.array([0, 1])
w_bar   = 0.35
bars_eq = axes[0].bar(x_pos - w_bar / 2, vals_eq, width=w_bar,
                      color="#4C9BE8", label="Equal-weight")
bars_kn = axes[0].bar(x_pos + w_bar / 2, vals_kn, width=w_bar,
                      color="#E84C4C", label="Perf.+Indep.")
for bar, val in zip(list(bars_eq) + list(bars_kn), vals_eq + vals_kn):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 1.01,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(["Yearly RMSE", f"{SMOOTH_WIN}-yr rolling RMSE"], fontsize=9)
axes[0].set_ylabel("RMSE (°C)")
axes[0].set_title("Yearly vs smoothed RMSE\n(test period 1990–2014)")
axes[0].legend(frameon=False, fontsize=8)
axes[0].set_ylim(0, max(vals_eq + vals_kn) * 1.25)

# Middle panel: raw yearly trajectories
axes[1].plot(test_years, era5_test,        color="black",   lw=2.0, label="ERA5")
axes[1].plot(test_years, equal_test_traj,  color="#4C9BE8", lw=1.6,
             linestyle="--", label="Equal-weight")
axes[1].plot(test_years, knutti_test_traj, color="#E84C4C", lw=1.6,
             label="Perf.+Indep.")
axes[1].set_title("Yearly trajectories\n(test period 1990–2014)")
axes[1].set_ylabel("GSAT anomaly (°C)")
axes[1].set_xlabel("Year")
axes[1].legend(frameon=False, fontsize=8)

# Right panel: smoothed trajectories
axes[2].plot(sm_era5.index,   sm_era5.values,   color="black",   lw=2.0, label="ERA5")
axes[2].plot(sm_equal.index,  sm_equal.values,  color="#4C9BE8", lw=1.6,
             linestyle="--", label="Equal-weight")
axes[2].plot(sm_knutti.index, sm_knutti.values, color="#E84C4C", lw=1.6,
             label="Perf.+Indep.")
axes[2].set_title(f"{SMOOTH_WIN}-yr rolling mean\n(forced-signal proxy)")
axes[2].set_ylabel("GSAT anomaly (°C)")
axes[2].set_xlabel("Year")
axes[2].legend(frameon=False, fontsize=8)

fig.suptitle(
    "Robustness: does Perf.+Indep. weighting improve the forced decadal signal,\n"
    "not just interannual alignment?  (calibrated 1940–1989; evaluated 1990–2014)",
    fontsize=10
)
fig.savefig(os.path.join(OUT_DIR, "fig11_smoothed_rmse_robustness.png"),
            bbox_inches="tight")
plt.close(fig)
print("Saved fig11_smoothed_rmse_robustness.png")


# =============================================================================
# 18. FIG 12 – Coverage test
# =============================================================================
# Tests whether ERA5 falls within the 17–83% weighted model spread.
# A well-calibrated spread should contain ERA5 ~66% of the time.
# If the Knutti spread is narrower but coverage stays near 66%, the spread
# is more efficient (same information, less uncertainty).
# If coverage drops below 66% under Knutti, the spread is over-tightened.
#
# We evaluate two periods:
#   (a) Full calibration period 1940–2014 (indicative only – training data)
#   (b) Hindcast test period 1990–2014   (out-of-sample – uses hindcast weights)
# =============================================================================

def compute_coverage(df_p, era5_s, weights, years, lo_pct=17, hi_pct=83):
    """
    Fraction of years where era5_s falls within the weighted lo–hi model spread.
    Returns coverage fraction plus per-year lo, hi, obs arrays for plotting.
    """
    lo_vals, hi_vals, obs_vals, yrs_out = [], [], [], []
    inside = 0
    for y in years:
        if y not in era5_s.index or y not in df_p.index:
            continue
        model_vals = df_p.loc[y].values
        lo  = weighted_percentile(model_vals, weights, lo_pct)
        hi  = weighted_percentile(model_vals, weights, hi_pct)
        obs = era5_s.loc[y]
        lo_vals.append(lo)
        hi_vals.append(hi)
        obs_vals.append(obs)
        yrs_out.append(y)
        if lo <= obs <= hi:
            inside += 1
    coverage = inside / len(yrs_out) if yrs_out else np.nan
    return coverage, lo_vals, hi_vals, obs_vals, yrs_out


# (a) Full calibration period – uses full-period weights (indicative only)
calib_yrs_cov = [y for y in df_pivot.index
                 if CALIB_START <= y <= CALIB_END and y in era5_series.index]

cov_eq_full, *_ = compute_coverage(
    df_pivot, era5_series, equal_weights,  calib_yrs_cov)
cov_kn_full, *_ = compute_coverage(
    df_pivot, era5_series, wt["weights"],  calib_yrs_cov)

# (b) Hindcast test period – uses hindcast weights (out-of-sample)
cov_eq_test, lo_eq_t, hi_eq_t, obs_t, yrs_t = compute_coverage(
    df_pivot, era5_series, equal_weights,          test_years)
cov_kn_test, lo_kn_t, hi_kn_t, _,     _     = compute_coverage(
    df_pivot, era5_series, wt_hindcast["weights"], test_years)

print(f"\nCoverage test (ERA5 inside 17-83% model spread):")
print(f"  Calibration period 1940-2014 (indicative): "
      f"Equal={cov_eq_full:.1%}   Perf.+Indep.={cov_kn_full:.1%}   expected~66%")
print(f"  Test period 1990-2014 (out-of-sample):     "
      f"Equal={cov_eq_test:.1%}   Perf.+Indep.={cov_kn_test:.1%}   expected~66%")

# Figure
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

def plot_coverage_band(ax, years, lo, hi, obs, color_band,
                       label_band, coverage_frac, title):
    ax.fill_between(years, lo, hi, alpha=0.25, color=color_band,
                    label=f"{label_band} 17–83% spread")
    ax.plot(years, lo, color=color_band, lw=0.8, linestyle="--")
    ax.plot(years, hi, color=color_band, lw=0.8, linestyle="--")
    ax.plot(years, obs, color="black", lw=1.8, label="ERA5")
    for y, o, l, h in zip(years, obs, lo, hi):
        clr = "#2ECC71" if l <= o <= h else "#E74C3C"
        ax.scatter(y, o, color=clr, s=25, zorder=5)
    ax.set_title(f"{title}\nCoverage = {coverage_frac:.1%}  (expected ≈ 66%)")
    ax.set_xlabel("Year")
    ax.set_ylabel("GSAT anomaly (°C, rel. 1981–2014)")
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.99, 0.02,
            "Green = ERA5 inside spread  |  Red = ERA5 outside spread",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="0.5", style="italic")

plot_coverage_band(
    axes[0], yrs_t, lo_eq_t, hi_eq_t, obs_t,
    "#4C9BE8", "Equal-weight",
    cov_eq_test,
    f"Equal-weight 17–83% spread\n(test period 1990–2014)"
)
plot_coverage_band(
    axes[1], yrs_t, lo_kn_t, hi_kn_t, obs_t,
    "#E84C4C", "Perf.+Indep.",
    cov_kn_test,
    f"Perf.+Indep. 17–83% spread\n(test period 1990–2014; weights calibrated 1940–1989)"
)

fig.suptitle(
    "Coverage test: is ERA5 well-contained within the 17–83% model spread?\n"
    "Coverage ≈ 66% → well-calibrated;  < 66% → spread is over-tightened",
    fontsize=10
)
fig.savefig(os.path.join(OUT_DIR, "fig12_coverage_test.png"),
            bbox_inches="tight")
plt.close(fig)
print("Saved fig12_coverage_test.png")

