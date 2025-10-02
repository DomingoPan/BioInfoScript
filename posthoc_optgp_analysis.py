#使用optGH取樣後的.zarr，進行後續的分析與出圖
# > conda activate gem-py39 (自創獨立分析環境，防止版本衝突引發的bug)
# > python "D:\cch_data\posthoc_optgp_analysis_mix.py" 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from scipy.stats import ttest_ind, ks_2samp
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import zarr
import xarray as xr
import multiprocessing, dask          # ← 放在原有 import 之前或之後都可
# ──────────── CPU 核心數自訂 ────────────
N_CORES = 12                                # ★改成你想用的核心數
os.environ["OMP_NUM_THREADS"]       = str(N_CORES)
os.environ["OPENBLAS_NUM_THREADS"]  = str(N_CORES)
os.environ["MKL_NUM_THREADS"]       = str(N_CORES)
os.environ["NUMEXPR_MAX_THREADS"]   = str(N_CORES)
dask.config.set(scheduler="threads", num_workers=N_CORES)
# --- NEW: effect-size helpers  --------------------
from itertools import combinations
from math import sqrt
from collections import Counter

def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:           # 任一組空
        return np.nan
    s1, s2 = a.var(ddof=1), b.var(ddof=1)
    s_pooled = sqrt(((len(a)-1)*s1 + (len(b)-1)*s2) / (len(a)+len(b)-2))
    return (np.mean(a) - np.mean(b)) / s_pooled if s_pooled else np.nan

def cliffs_delta(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = sum(x >  y for x in a for y in b)
    lt = sum(x <  y for x in a for y in b)
    return (gt - lt) / (len(a)*len(b))


# ================= 全域參數 =================
VAR_TOL       =0    # 跨樣本變異門檻
MIN_NZR       = 0.20      # 非零樣本比例門檻
USE_LIMMA     = True
USE_TREAT     = True
TREAT_LFC     = 1      # log2 ≈ 1.5-fold
LOG2_EPS      = 1e-9
LIMMA_ROBUST  = True
LIMMA_TREND   = True
# ===========================================
# 設定參數與目錄
DATA_DIRS = [
    Path(r"D:/cch_data/mice_RNAdata_for_AST/eflux_FVA_sampling/optGP_N_20000_FVA08_0815/eflux_FVA_sampling"),
    Path(r"D:/cch_data/mice_RNAdata_for_AST/eflux_FVA_sampling/optGP_N_20000_0728/eflux_FVA_sampling"),
    Path(r"D:/cch_data/mice_RNAdata_for_AST/eflux_FVA_sampling/optGP_N_20000_0826/eflux_FVA_sampling"),
]
OUT_DIR = Path("D:/cch_data/mice_RNAdata_for_AST/eflux_FVA_sampling/optGP_0815_0728_0826_mix/posthoc_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)
GROUP_CSV = "D:/cch_data/mice_RNAdata_for_AST/GSEA_R_result/sample_groups.csv"
BURN = 10000
THIN = 100
# 新增：反應 → KEGG pathway 對應表（自行準備成 csv）
BASE_DIR = Path(r"D:\cch_data\mice_RNAdata_for_AST\eflux_FVA_sampling")
PATH_MAP  = BASE_DIR / "rxn2kegg.csv"   # columns: reaction,pathway
N_PERM   = 10000                        # permutation 次數
# 匯入 group 資訊
rxn_id_map = {}          # <--- 新增：每個 sample 對應的 rxn_ids 清單
group_df = pd.read_csv(GROUP_CSV)
group_df["sample"] = group_df["sample"].astype(str)
sample_ids = group_df["sample"].tolist()
def open_flux_8chains(sid, burn=BURN, thin=THIN):
    """
    回傳 dims=('chain','draw','reaction') 的 DataArray，
    把兩個 run 的 4+4 鏈在 chain 維度串接（只對齊 reaction）。
    """
    das = []
    chain_offset = 0
    for root in DATA_DIRS:
        z = root / sid / "optgp.zarr"
        if not z.exists():
            print(f"⚠️ {sid} 缺少 {z}，略過該 run")
            continue
        ds = xr.open_zarr(z, chunks="auto")
        da = (ds["flux"]
              .isel(draw=slice(min(burn, ds.sizes.get("draw", 0)), None))
              .isel(draw=slice(None, None, thin))
              .transpose("chain","draw","reaction"))
        # 讓 chain 座標唯一（避免 0..3 重複）
        new_chain = np.arange(chain_offset, chain_offset + da.sizes["chain"])
        da = da.assign_coords(chain=("chain", new_chain))
        das.append(da)
        chain_offset += da.sizes["chain"]
        ds.close()

    if not das:
        return None

    # 只在 reaction 維度做交集，避免 chain 被 inner 掉成空
    rxn_inter = None
    for da in das:
        r = da.coords["reaction"].values
        rxn_inter = r if rxn_inter is None else np.intersect1d(rxn_inter, r)

    if rxn_inter is None or rxn_inter.size == 0:
        print(f"⚠️ {sid} 兩個 run 的 reaction 沒有交集")
        return None

    das2 = [da.sel(reaction=rxn_inter) for da in das]
    da_all = xr.concat(das2, dim="chain")   # 直接沿 chain 串接
    return da_all
# —— 工具函式：四/八鏈合併後取中位數（含等量抽樣避免某鏈過重） ——
def pooled_median_from_flux_da(flux_da, equalize=True):
    # 這裡假設傳進來的 flux_da 已經 burn/thin 並有 dims=('chain','draw','reaction')
    flux_da = flux_da.transpose("chain","draw","reaction")
    arr = np.asarray(flux_da.values)    # (C, D, R)
    C, D, R = arr.shape
    if D == 0 or R == 0:
        rxn_ids = flux_da.coords["reaction"].values.astype(str)
        return pd.Series(np.full(R, np.nan), index=rxn_ids, name="median")
    if equalize:
        nmin = min(np.count_nonzero(~np.isnan(arr[c, :, 0])) for c in range(C))
        idx  = np.linspace(0, D-1, nmin, dtype=int)
        arr  = arr[:, idx, :]
    pooled = arr.reshape(-1, R)         # (C*nmin, R)
    med = np.nanmedian(pooled, axis=0)
    rxn_ids = flux_da.coords["reaction"].values.astype(str)
    return pd.Series(med, index=rxn_ids, name="median")

# --- 一次迴圈：每個樣本打開 4n 鏈，產生 pooled median；（可選）保留 flatten draws 供 KS/trace ---
samp_df, rxn_id_map, med_cols = {}, {}, {}
for sid in sample_ids:
    da = open_flux_8chains(sid, burn=BURN, thin=THIN)   # ← 這裡已經做過 burn/thin
    if da is None:
        continue
    print(f"[DEBUG] {sid} dims={da.dims} sizes={k:int(v) for k,v in da.sizes.items()}")

    #4n 鏈合併後中位數（點估計）
    med_cols[str(sid)] = pooled_median_from_flux_da(da, equalize=True)

    #flatten draws 供 KS/trace/ACF
    arr = da.transpose("reaction","chain","draw").values  # (R, C, D)
    R, C, D = arr.shape
    data = arr.reshape(R, C*D)                            # (R, C*D)
    rxn_ids = da.coords["reaction"].values.astype(str).tolist()
    samp_df[str(sid)] = pd.DataFrame(data, index=rxn_ids)
    rxn_id_map[str(sid)] = rxn_ids

# 組 pooled median 矩陣（反應 × 樣本）
med_df = pd.DataFrame(med_cols).sort_index()
med_df.to_csv(OUT_DIR / "flux_median_matrix.csv")
print("✔ pooled median matrix:", med_df.shape)

#分組欄位（確保型別與對齊）
group_df["sample"] = group_df["sample"].astype(str)
med_df.columns     = med_df.columns.astype(str)
ctrl_cols = group_df.loc[group_df["group"] == "Control", "sample"].tolist()
sca3_cols = group_df.loc[group_df["group"] == "SCA3",     "sample"].tolist()

#若有遺漏，印警告並從列表中移除
missing_ctrl = [c for c in ctrl_cols if c not in med_df.columns]
missing_sca3 = [c for c in sca3_cols if c not in med_df.columns]
if missing_ctrl or missing_sca3:
    print("⚠️  下列 sample 在 med_df 找不到、將被忽略：",
          ", ".join(missing_ctrl + missing_sca3))
ctrl_cols = [c for c in ctrl_cols if c in med_df.columns]
sca3_cols = [c for c in sca3_cols if c in med_df.columns]
#最後組 all_cols，供 permutation 洗牌使用
all_cols  = ctrl_cols + sca3_cols

# ===================== Reaction-level differential analysis =====================
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
# ---- QC 指標（以每樣本彙整值 med_df 為單位）----
vals = med_df.loc[:, all_cols].astype(float)
var_samples = vals.var(axis=1, ddof=1)
finite_counts  = np.isfinite(vals).sum(axis=1)
nonzero_counts = ((vals != 0) & np.isfinite(vals)).sum(axis=1)
nonzero_ratio  = (nonzero_counts / np.where(finite_counts>0, finite_counts, np.nan)).fillna(0.0)

# ---- 逐反應：Welch t + 效應量 ----
res_rows = []
for rxn in med_df.index:
    a = med_df.loc[rxn, ctrl_cols].astype(float).to_numpy()
    b = med_df.loc[rxn, sca3_cols].astype(float).to_numpy()
    a = a[np.isfinite(a)];  b = b[np.isfinite(b)]

    ctrl_mean = np.nanmean(a) if a.size else np.nan
    sca3_mean = np.nanmean(b) if b.size else np.nan

    # 位移後 log2FC（處理 0/負）
    log2fc = np.nan
    if np.isfinite(ctrl_mean) and np.isfinite(sca3_mean):
        mins = []
        if a.size: mins.append(np.nanmin(a))
        if b.size: mins.append(np.nanmin(b))
        if not mins: mins = [min(ctrl_mean, sca3_mean)]
        shift = -float(np.nanmin(mins)) + LOG2_EPS if np.nanmin(mins) <= 0 else 0.0
        num = sca3_mean + shift; den = ctrl_mean + shift
        if (num > 0) and (den > 0):
            log2fc = float(np.log2(num / den))

    if (a.size == 0) or (b.size == 0):
        t_stat, p_t = np.nan, np.nan
    elif (np.nanstd(a, ddof=1) == 0 and np.nanstd(b, ddof=1) == 0):
        t_stat, p_t = 0.0, 1.0
    else:
        t_stat, p_t = ttest_ind(a, b, equal_var=False, nan_policy="omit")
    # NEW: effect sizes
    d_val   = cohens_d(a, b)
    cliff_d = cliffs_delta(a, b)

    res_rows.append({"reaction": rxn,
                     "ctrl_mean": ctrl_mean, "sca3_mean": sca3_mean,
                     "log2FC": log2fc, "t": t_stat, "p": p_t,
                     "d_cohen": d_val, "delta_cliff": cliff_d})
res_rxn = pd.DataFrame(res_rows).set_index("reaction")
# 加入 QC 指標
res_rxn["var_samples"]        = var_samples.reindex(res_rxn.index).to_numpy()
res_rxn["nonzero_ratio_samp"] = nonzero_ratio.reindex(res_rxn.index).to_numpy()

# ---- 獨立過濾 → BH-FDR ----
mask_valid = np.isfinite(res_rxn["p"])
mask_var   = res_rxn["var_samples"] > VAR_TOL
mask_nzr   = res_rxn["nonzero_ratio_samp"] >= MIN_NZR
print("[check] valid p:", int(mask_valid.sum()),
      " var>", VAR_TOL, "→", int(mask_var.sum()),
      " nzr≥", MIN_NZR, "→", int(mask_nzr.sum()))
mask_use = mask_valid & mask_var & mask_nzr
res_rxn["pass_filter"] = mask_use
if mask_use.any():
    res_rxn.loc[mask_use, "fdr_bh"] = multipletests(res_rxn.loc[mask_use, "p"], method="fdr_bh")[1]

print(f"[t-test] tested {int(mask_use.sum())}/{int(mask_valid.sum())} after filtering; "
      f"BH q<0.05: {int((res_rxn['fdr_bh']<0.05).sum())}")
# ================ limma/eBayes + TREAT（反應層級；在過濾之後） =============
USE_LIMMA    = True
USE_TREAT    = True
TREAT_LFC    = 0.58   # log2 ≈ 1.5-fold
LOG2_EPS     = 1e-9
LIMMA_ROBUST = True
LIMMA_TREND  = True

# 這裡以「反應層級主表 res_rxn」為基準
res_path = res_rxn.copy()

# 取「通過過濾且 p 有效」的反應清單
rxn_keep = res_path.index[(res_path["pass_filter"]) & np.isfinite(res_path["p"])].tolist()

if USE_LIMMA and len(rxn_keep) >= 2:
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import default_converter
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects import pandas2ri

        all_cols = group_df.set_index("sample").loc[ctrl_cols + sca3_cols].index.tolist()
        ann = group_df.set_index("sample").loc[all_cols, ["group"]].copy()
        Y = med_df.loc[rxn_keep, all_cols].astype(float)

        # global shift + log2（確保>0；TREAT 的門檻在 log2 尺度）
        Y_vals = Y.values
        mins   = np.nanmin(Y_vals, axis=1)
        shifts = np.where(mins <= 0, -mins + LOG2_EPS, 0.0)
        Y_log2 = np.log2(Y_vals + shifts[:, None])
        Y_log2 = pd.DataFrame(Y_log2, index=Y.index, columns=Y.columns)

        # 傳進 R
        with localconverter(default_converter + pandas2ri.converter):
            ro.globalenv["Y"]   = Y_log2
            ro.globalenv["ann"] = ann.reset_index(drop=False)[["sample","group"]]

        # limma + TREAT
        ro.r(f"""
            suppressMessages(library(limma))
            ann$group <- factor(ann$group, levels=c("Control","SCA3"))
            design <- model.matrix(~ 0 + group, data=ann)
            colnames(design) <- sub("^group", "", colnames(design))
            fit   <- lmFit(Y, design)
            contr <- makeContrasts(SCA3 - Control, levels=design)
            fit2  <- contrasts.fit(fit, contr)
            fit2  <- eBayes(fit2, robust={'TRUE' if LIMMA_ROBUST else 'FALSE'}, trend={'TRUE' if LIMMA_TREND else 'FALSE'})
            tab   <- topTable(fit2, number=Inf, sort.by='none')
            fit3  <- treat(fit2, lfc={float(TREAT_LFC)})
            tab_t <- topTable(fit3, number=Inf, sort.by='none')
        """)

        # 從 R 拉回 pandas
        with localconverter(default_converter + pandas2ri.converter):
            tab   = ro.conversion.rpy2py(ro.r("as.data.frame(tab)"))
            tab_t = ro.conversion.rpy2py(ro.r("as.data.frame(tab_t)"))
        tab.index   = Y_log2.index
        tab_t.index = Y_log2.index

        # 合併回 res_path
        res_path.loc[tab.index,   ["logFC_limma_log2","t_limma","p_limma","fdr_limma"]] = \
            tab[["logFC","t","P.Value","adj.P.Val"]].to_numpy()
        res_path.loc[tab_t.index, ["logFC_treat_log2","t_treat","p_treat","fdr_treat"]] = \
            tab_t[["logFC","t","P.Value","adj.P.Val"]].to_numpy()

        n_q05_limma = int((res_path.get("fdr_limma") < 0.05).sum()) if "fdr_limma" in res_path else 0
        n_q05_treat = int((res_path.get("fdr_treat") < 0.05).sum()) if "fdr_treat" in res_path else 0
        s_min, s_med, s_max = float(np.nanmin(shifts)), float(np.nanmedian(shifts)), float(np.nanmax(shifts))
        print(f"[limma@reaction] shift(min/median/max)={s_min:.3g}/{s_med:.3g}/{s_max:.3g}, "
      f"FDR<0.05: limma={n_q05_limma}, TREAT={n_q05_treat}")


        # 回寫到主表 res_rxn（只覆蓋/新增統計欄）
        res_rxn.loc[res_path.index, res_path.columns.intersection(
            ["logFC_limma_log2","t_limma","p_limma","fdr_limma",
             "logFC_treat_log2","t_treat","p_treat","fdr_treat"])] = \
            res_path[["logFC_limma_log2","t_limma","p_limma","fdr_limma",
                      "logFC_treat_log2","t_treat","p_treat","fdr_treat"]]
    except Exception as e:
        print("[limma@reaction] skipped:", e)

# ---- Storey q-value（可選）----
try:
    from qvalue import qvalue
    msk = res_rxn["pass_filter"] & np.isfinite(res_rxn["p"])
    res_rxn.loc[msk, "q_storey"] = qvalue(res_rxn.loc[msk, "p"].values)
except Exception as e:
    print("[qvalue] skipped:", e)

# ---- 輸出反應層級主表 ----
rxn_csv = OUT_DIR / "diff_flux_optGP_median.csv"
res_rxn.reset_index().to_csv(rxn_csv, index=False)
print("✔ reaction-level file:", rxn_csv, "rows=", len(res_rxn))


# ===================== Pathway permutation test (median) =====================
import numpy as np
from statsmodels.stats.multitest import multipletests
#讀取反應 → pathway 對應
rxn2path = (pd.read_csv(PATH_MAP)
              .dropna(subset=["reaction", "pathway"])
              .set_index("reaction")["pathway"])
#把 median matrix (med_df) 依 group 分兩個子矩陣
ctrl_vals = med_df.loc[:, ctrl_cols].astype(float)
sca3_vals = med_df.loc[:, sca3_cols].astype(float)

#計算「實際」統計量：各 pathway 的 median log2FC
log2fc_rxn = np.log2((sca3_vals.mean(1) + 1e-9) /
                     (ctrl_vals.mean(1) + 1e-9))
# ---------- 1. 把 mapping 變成 DataFrame，展開多對映 ----------
# 假設 rxn2path 原本是 dict {rxn: [pwy1, pwy2]} 或 Series
mapping_df = (pd
              .Series(rxn2path, name="pathway")       # -> Series
              .explode()                              # 多對映展開成多列
              .dropna()
              .to_frame())                            # => DataFrame: index=reaction
# ---------- 2. 把 log2FC 對應進來 ----------
mapping_df["log2fc"] = log2fc_rxn.reindex(mapping_df.index)
# ---------- 3. 計算每 pathway 統計量 ----------
path_stat = mapping_df.groupby("pathway")["log2fc"].median()
# mapping_df 已在前面建立： index = reaction , col = pathway
# -------------------------------------------------------------
# ---------- (1) permutation 迴圈內：safe join -------------------------
all_vals = med_df.loc[:, all_cols].astype(float).to_numpy()
null_dist = {p: [] for p in path_stat.index}
label     = np.array([0]*len(ctrl_cols) + [1]*len(sca3_cols))
rng       = np.random.default_rng(42)

for _ in range(N_PERM):                       # ← 迴圈開始
    rng.shuffle(label)
    ctrl_idx, sca3_idx = label == 0, label == 1

    with np.errstate(divide="ignore", invalid="ignore"):
        perm_fc = np.log2((all_vals[:, sca3_idx].mean(1)+1e-9) /
                          (all_vals[:, ctrl_idx].mean(1)+1e-9))

    # -------★ 把下列 4 行放在 **這裡** ----------
    perm_ser = pd.Series(perm_fc, index=med_df.index, name="log2fc")

    tmp = (mapping_df
           .drop(columns="log2fc", errors="ignore")
           .join(perm_ser, how="left"))

    perm_stat = tmp.groupby("pathway")["log2fc"].median()
    # --------------------------------------------

    for p, v in perm_stat.items():            # 蒐集 null 分布
        null_dist[p].append(v)
# 迴圈結束

# ---------- (2) 產生 p 值  & FDR --------------------------------------
pvals = {}
for p in path_stat.index:
    dist = np.asarray(null_dist[p], dtype=float)
    if dist.size == 0:               # 全空 → 填 NaN 佔位
        dist = np.array([np.nan])
    pvals[p] = (np.nansum(np.abs(dist) >= abs(path_stat[p])) + 1) / (N_PERM + 1)

from statsmodels.stats.multitest import multipletests
fdr = multipletests(list(pvals.values()), method="fdr_bh")[1]
# -----------------------------------------------------------------------

# ❷ 建立 res：統計 + p + fdr
res = (pd.DataFrame({
          "pathway" : list(pvals.keys()),
          "stat"    : [path_stat[p] for p in pvals],
          "p"       : list(pvals.values()),
          "fdr"     : fdr
      })
      .set_index("pathway"))
# 1) 先把同一路徑的 p 聚合（這裡示範「取最小 p」；也可改 Fisher 合併等方法）
agg_p = res["p"].groupby(level=0).min()   # level=0 = index 分組，避免 FutureWarning

# 2) 對有限值做 BH 校正
pvals = agg_p.values
mask  = np.isfinite(pvals)
qvals = np.full_like(pvals, np.nan, dtype=float)
qvals[mask] = multipletests(pvals[mask], method="fdr_bh")[1]

# 3) 以 index 對齊回併，不動原本 res 的欄
agg_df = pd.DataFrame({"fdr_pathway": qvals}, index=agg_p.index)
res = res.join(agg_df, how="left")

# 若你要用 pathway-level FDR 來設門檻，可在這裡或 explode 之後再設：
res["pass_filter"] = (res["fdr_pathway"] < 0.10) & np.isfinite(res["p"])

# 4) 輸出
res.to_csv(OUT_DIR / "diff_flux_pathway_FDR_recalc.csv", index=False)
print("✓ pathway-level FDR（聚合後 BH）已寫出 diff_flux_pathway_FDR_recalc.csv")
# ❸ 把「pathway ↔ reaction（BiGG）」對映加進來（一對多 → explode）
mapping_df = mapping_df.rename_axis("reaction")  # 確保 index 名稱固定為 'reaction'
# 若 mapping_df["pathway"] 是 'map' 前綴，轉成 'rn' 以對上 res 裏的 'rn'
mapping_df["pathway"] = mapping_df["pathway"].str.replace("map", "rn", regex=False)
rxn_list = (mapping_df
            .reset_index()                       # reaction | pathway
            .groupby("pathway")["reaction"]
            .apply(list)
            .rename("reaction_bigg_id"))

res = (res
       .join(rxn_list)            # 每條 pathway 對到一串 BiGG 反應
       .explode("reaction_bigg_id")
       .reset_index())            # pathway 變回一般欄

# ★ 加上 pass_filter 欄位（之後會被用到）
res["pass_filter"] = (res["fdr"] < 0.10) & np.isfinite(res["p"])

# ❹ 寫檔（full & sig）
full_csv = OUT_DIR /"diff_flux_pathway_permutation_full.csv"
sig_csv  = OUT_DIR /"diff_flux_pathway_permutation_sig.csv"

res.to_csv(full_csv, index=False)
res.query("pass_filter").to_csv(sig_csv, index=False)

print(f"✔  full  file：{full_csv}  (rows={len(res)})")
print(f"✔  signif file：{sig_csv}  (rows={len(res.query('pass_filter'))})")

# 下游若需要「顯著 pathway 的反應清單」→ 用 reaction_bigg_id 欄
rxn_keep = res.loc[res["pass_filter"], "reaction_bigg_id"].dropna().unique().tolist()

# --- ATPM trace & ACF (4n chains, pretty style) -------------------
ATPM_ID = "ATPM"

def _acf_1d(x, max_lag=None):
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    n = x.size
    if n < 2:
        return np.arange(1), np.array([np.nan])
    if max_lag is None or max_lag >= n:
        max_lag = n - 1
    c = np.correlate(x, x, mode="full")
    mid = c.size // 2
    ac = c[mid:mid+max_lag+1] / c[mid]
    lags = np.arange(0, max_lag+1)
    return lags, ac

def plot_acf_pretty(chains_2d, title, out_png):
    """
    chains_2d: (n_chain, n_draw) 已去暖機
    風格：多色曲線 + 0 基準線 + 灰色 95% CI 帶
    """
    C, D = chains_2d.shape
    ci = 1.96 / np.sqrt(D)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    for i in range(C):
        lags, ac = _acf_1d(chains_2d[i], max_lag=D-1)
        ax.plot(lags, ac, linewidth=2, label=f"chain {i}")

    ax.fill_between([0, D-1], -ci, ci, color="0.85")  # 95% CI 帶
    ax.axhline(0, color="0.25", linewidth=1)          # 0 線
    ax.set_xlim(0, D-1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

for sid in sample_ids:
    da = open_flux_8chains(sid, burn=BURN, thin=THIN)  # ← 兩個 run 串成 4n 鏈
    if da is None:
        continue
    if ATPM_ID not in da.coords["reaction"].values:
        print(f"⚠️ {sid} 不含 {ATPM_ID}，略過")
        continue

    vals = da.sel(reaction=ATPM_ID).values  # shape: (chain, draw)

    # trace
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    for i in range(vals.shape[0]):
        ax.plot(vals[i], linewidth=1.2, label=f"chain {i}")
    ax.set_title(f"{ATPM_ID} trace – {sid}")
    ax.set_xlabel("Draw")
    ax.set_ylabel("Flux")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{sid}_{ATPM_ID}_trace.png")
    plt.close(fig)

    # ACF（你要的樣式）
    plot_acf_pretty(vals, f"{ATPM_ID} ACF – {sid}", OUT_DIR / f"{sid}_{ATPM_ID}_acf.png")

# --- PCA & UMAP ----------------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(med_df.T)
pca = PCA(n_components=2).fit_transform(X)
umap_result = UMAP(n_components=2).fit_transform(X)
pca_df = pd.DataFrame(pca, columns=["PC1", "PC2"], index=med_df.columns)
umap_df = pd.DataFrame(umap_result, columns=["UMAP1", "UMAP2"], index=med_df.columns)
pca_df["group"] = group_df.set_index("sample").loc[pca_df.index]["group"]
umap_df["group"] = pca_df["group"]
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="group")
plt.title("PCA of flux medians")
plt.savefig(OUT_DIR / "flux_PCA.png")
plt.close()
sns.scatterplot(data=umap_df, x="UMAP1", y="UMAP2", hue="group")
plt.title("UMAP of flux medians")
plt.savefig(OUT_DIR / "flux_UMAP.png")
plt.close()
