import matplotlib
matplotlib.use('Agg')
#!/usr/bin/env python3

# ──────────────────────────────────────────────────────────────────────────────
#                            ORIGINAL CODE ZONE
# ──────────────────────────────────────────────────────────────────────────────

### ORIGINAL CODE START #######################################################

# ── 把你現有的 gem_by_libsbml_eflux_fva_sampling.py 原文完整貼在這裡 ──

### ORIGINAL CODE END #########################################################
#GEM_by_libsbml_Eflux_FVA_sampling.py
#-----------------------------------
#* 完整重構：每一個 context‑specific GEM 皆內嵌 FVA、OptGP 取樣與區間視覺化
#* 引入連續 **E‑flux**：以基因 TPM 相對 95th percentile 縮放反應 upper_bound
#* 保留 rFASTCORMICS switch‑based 剪枝 (−1→block)，在此基礎上加入連續 scaling 以提高解析度
#保留 rFASTCORMICS 離散化、Boolean GPR。。
#每 sample：pFBA → FVA → OptGP sampling。
# **統計 & 視覺化**
#  1. pFBA flux Welch t‑test (Control vs SCA3) + 顯著反應箱型。
#  2. **Active reactions / expressed genes count boxplot**（分組）。
#  3. **PCA** of pFBA flux（前 2 PCs）。
#  4. FVA min‑max interval plot per sample。
#  5. UMAP of sampling mean flux。
#############################################################################
#注意:Gurobi Free Academic licenses需索取。Free Trial金鑰只有2K扣打，不足以進行。沒有Gurobi依舊可以進行E-flux，只是會 用GLPK跑超級慢，N_SAMPLES=500下，單一樣本約需3到4小時。
#############################################################################
#主要步驟
#1. 讀入 iMM1865、RNA‑seq TPM、樣本分組
#2. rFASTCORMICS：兩段式 GMM 切點 → gene status (−1/0/+1)
#3. **E‑flux scaling**：若基因為 expressed/unknown，計算 scaling = TPM / P95(TPM)
#   ‑ 單基因反應：直接乘 scaling
#   ‑ 多基因 (OR)：取 max(scaling)
#   ‑ 多基因 (AND)：取 min(scaling)
#4. 建立 sample 專屬 GEM
#   ‑ 封死 inactive 反應 (lower=upper=0)
#   ‑ 調整 upper/lower_bound 以反映 scaling
#5. **pFBA + FVA(90 % optimum) + OptGP sampling (N=5000)**
#6. 將 pFBA、FVA(min/max) 與 sampling 平均值匯出 → downstream 繪圖
#7. 自動產生 interval plot (min‑max) 與 sampling 熱圖
#
#★ 依賴：cobra>=0.29、libSBML、OptGPSampler (cobra.sampling)、pandas、numpy、matplotlib、seaborn、scikit‑learn、umap‑learn
# 建議於 conda 環境 gem-py39 下執行本程式
# > conda activate gem-py39 (自創獨立分析環境，防止版本衝突引發的bug)
# > python "D:\cch_data\GEM_by_OPTGP.py" --parallel --expr D:/cch_data/mice_RNAdata_for_AST/GSEA_R_result/expr_matrix_filtered.csv
###############################################################################
import os, re, multiprocessing as mp, time, warnings, json
os.environ['OMP_NUM_THREADS'] = '1'           # 每子行程單執行緒
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from scipy.stats import norm, ttest_ind
from scipy.optimize import brentq
from statsmodels.stats.multitest import multipletests
from cobra.io import read_sbml_model, write_sbml_model
from cobra.flux_analysis import pfba, flux_variability_analysis
from cobra.flux_analysis import find_essential_reactions
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import gurobipy as gp
import cobra
import arviz as az
import xarray as xr
import sys
from pathlib import Path
import dask.array as da, zarr
from numcodecs import Blosc
from cobra.sampling import OptGPSampler
from scipy.stats import ks_2samp
#-----------Monkey patch------------
from cobra.sampling.hr_sampler import HRSampler
# 全域關閉冗餘檢查：直接回傳全 False
def _no_redundant(self, mat):
    # mat.shape[0] = warm-up 向量數（行）
    return np.zeros(mat.shape[0], dtype=bool)
HRSampler._is_redundant = _no_redundant
# ───────────────────────────────
# Global hyper-parameters 
# ───────────────────────────────
# 需要改門檻時只改這裡；子行程會再次 import 同檔，自然拿到同值
FVA_FRAC    = 0.8   # fraction-of-optimum for FVA
RUN_SAMPLING = True
N_BURN      = 10000
N_KEEP      = 20000
THINNING    = 100
N_CHAINS    = 4
N_SAMPLES   = N_BURN + N_KEEP
N_PROCESSES = 4
# ── Global hyper-parameters ──────────────────────────────────────────
DTYPE        = np.float32           
Z_CHUNK_DRAW = 2_500               # 每塊 draw 數，上下可調
COMPRESSOR   = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
# ── quick pool-test helper ─────────────────────────────────
def ping(i):
    import os, time
    time.sleep(0.5)          # 模擬負載
    return os.getpid()
# ── Global solver/verbosity ────────────────────────────────────────────────
os.environ['GRB_PARAM_LOGTOCONSOLE'] = '0'
gp.setParam('OutputFlag', 0)                 # silence solver
try:
    import gurobipy as gp
    cobra.Configuration().solver = 'gurobi'
    try:
        gp.setParam('OutputFlag', 0)
    except Exception:
        pass
except Exception:
    warnings.warn('Gurobi 不可用，改用 GLPK；速度會慢很多')
    cobra.Configuration().solver = 'glpk'

# ── Parameters  ────────────────────────────────────────────────
expr_file   = "D:/cch_data/mice_RNAdata_for_AST/GSEA_R_result/expr_matrix_filtered.csv"
group_file  = "D:/cch_data/mice_RNAdata_for_AST/GSEA_R_result/sample_groups.csv"
model_file  = "D:/cch_data/mice_RNAdata_for_AST/GSEA_R_result/iMM1865.xml"
core_rxn_txt= "D:/cch_data/mice_RNAdata_for_AST/GSEA_R_result/core_reactions.txt"
out_dir     = "D:/cch_data/mice_RNAdata_for_AST/eflux_FVA_sampling/optGP_N_20000_0826/eflux_FVA_sampling"
OUTDIR: Path = Path(out_dir)
model_base = read_sbml_model(model_file)
auto_core   = True
SAVE_EFLUX  = True
LOAD_EFLUX  = True
PLOT_ALL    = True
PERCENTILE  = 95

np.random.seed(62)

pre_scale_dir = os.path.join(out_dir, "eflux_models"); os.makedirs(pre_scale_dir, exist_ok=True)
regex_gene = re.compile(r"(?:[Gg]_)?(\d{4,7})")

# ── Helper ────────────────────────────────────────────────────────────────
# ── ESS helper ────────────────────────────────────────────
def compute_ess_zarr(zarr_path, n_chains):
    ds = xr.open_zarr(zarr_path, chunks={"draw": Z_CHUNK_DRAW})
    flux = ds["flux"]                                       # dims: (chain, draw, reaction)
    # 先算各反應的方差；Dask 會保留 chunk
    var = flux.var(dim=("chain", "draw"))
    # 建布林遮罩（是否變動夠大），並確保「不是 Dask 布林」才拿來索引
    mask = (var > 1e-10)
    if getattr(mask, "chunks", None) is not None:
        mask = mask.compute()                               # 變成 NumPy 布林陣列
    # 用 isel(reaction=mask) 篩掉幾乎不變的反應（不要用 where(..., drop=True)）
    flux = flux.isel(reaction=mask)
    # 交給 ArviZ；DataArray 直接給 from_dict 沒問題
    idata = az.from_dict(posterior={"flux": flux})
    # 算 bulk ESS；如用 Dask，允許平行化
    ess = az.ess(idata, var_names=["flux"], method="bulk",
                 dask_kwargs={"dask": "parallelized"})["flux"]
    return ess.to_series()
def ess_dask(ds, **kw):
    return az.ess(
        ds, dask_kwargs={"dask": "parallelized"}, **kw
    )["flux"]
def rhat_dask(ds, **kw):
    return az.rhat(
        ds, dask_kwargs={"dask": "parallelized"}, **kw
    )["flux"]
# --- rFASTCORMICS‑style cutpoints (strict) ---------------------------------

def find_cutpoints(logv):
    lv = logv[logv > 0].reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42).fit(lv)
    m, s, w = gmm.means_.flatten(), np.sqrt(gmm.covariances_.flatten()), gmm.weights_
    idx = np.argsort(m)                  # m0 < m1
    m0, m1 = m[idx[0]], m[idx[1]]
    s0, s1 = s[idx[0]], s[idx[1]]
    w0, w1 = w[idx[0]], w[idx[1]]
    try:
        cut1 = brentq(lambda x: w0*norm.pdf(x, m0, s0) - w1*norm.pdf(x, m1, s1), m0, m1)
    except ValueError:                   # fallback: midpoint
        cut1 = (m0 + m1) / 2
    cut2 = m1 - s1                       # μ_expr − σ_expr
    return cut1, cut2, (m0, s0, m1, s1, w0, w1)

# -------- 新增：把 cut-point + 密度圖封裝成函式 ----------
def plot_density(tpm_series, sample_id, out_dir):
    """畫 log2(TPM) 雙峰 + cut-point；回傳 cut1, cut2"""
    logv = np.log2(tpm_series + 1).values
    cut1, cut2, (m0, s0, m1, s1, w0, w1) = find_cutpoints(logv)
    matplotlib.use("Agg")                 # 保險，防止並行衝突
    import matplotlib.pyplot as plt

    x = np.linspace(0, logv.max() + 1, 300)
    plt.figure(figsize=(5, 3))
    sns.histplot(logv, bins=80, stat="density",
                 color="steelblue", edgecolor=None, alpha=.4)
    plt.plot(x, w0 * norm.pdf(x, m0, s0), "--", c="orange")
    plt.plot(x, w1 * norm.pdf(x, m1, s1), "-.", c="green")
    plt.axvline(cut1, c="red");  plt.axvline(cut2, c="green")
    plt.title(f"{sample_id} log₂(TPM)")
    plt.tight_layout()

    den_dir = os.path.join(out_dir, "density")
    os.makedirs(den_dir, exist_ok=True)
    plt.savefig(f"{den_dir}/{sample_id}_density.png", dpi=300)
    plt.close()
    return cut1, cut2
# ----------------------------------------------------------
def rxn_inactive(rxn, gs):
    if not rxn.genes:
        return False
    gstatus = [gs.get(g.id, gs.get(_norm(g.id), "unknown")) for g in rxn.genes]
    return all(s != "expressed" for s in gstatus)
# ── Core list auto build (same as rev‑10) ─────────────────────────────────

def build_core_if_missing(model):
    if os.path.exists(core_rxn_txt) or not auto_core:
        return
    print("[Core] missing → run Richelle tasks …")
    core = {r.id for r in find_essential_reactions(model, processes=N_PROCESSES)}
    core |= {r.id for r in model.reactions if 'biomass' in r.id.lower()}
    with open(core_rxn_txt, 'w') as f:
        f.write("\n".join(sorted(core)))
regex_gene = re.compile(r"(?:[Gg]_)?(\d{4,7})")
_norm = lambda gid: re.sub(r"^[G_]+", "", str(gid).upper())
# ── Main ──────────────────────────────────────────────────────────────────

def main():
    t0=time.time(); os.makedirs(out_dir,exist_ok=True)
    expr = pd.read_csv(expr_file,index_col=0); expr.index = expr.index.map(_norm)
    group_df = (
        pd.read_csv(group_file, index_col=0)  # 原本 sample 在 index
          .reset_index()                     # ★ 這行把 sample 升為欄位
    )
    build_core_if_missing(model_base)
    with open(core_rxn_txt) as f: core_rxns={l.strip() for l in f if l.strip()}
    core_rxns |= {r.id for r in model_base.reactions if 'biomass' in r.id.lower()}
    all_genes={g.id for g in model_base.genes}
    rec_pFBA, rec_FVA, rec_samp = [], [], []
    reaction_status_records = []
    cut_summary=[]

    for i,smp in enumerate(expr.columns,1):
        print(f"[Start] {smp} ({i}/{expr.shape[1]})"); t1=time.time()
        pre_path=os.path.join(pre_scale_dir,f"{smp}_eflux.xml")
        if LOAD_EFLUX and os.path.exists(pre_path):
            m=read_sbml_model(pre_path)
            cut1, cut2 = plot_density(expr[smp], smp, out_dir)
            cut_summary.append({'sample': smp, 'cut1': cut1, 'cut2': cut2})   # ← 新增
            status_json = f"{pre_scale_dir}/{smp}_gene_status.json"
            if os.path.exists(status_json):
                # ▶ 正常路徑：讀回 gene_status
                with open(status_json) as fp:
                    gs = json.load(fp)
            else:
                # ▶ 第一次跑被中斷 → JSON 不在 → 立即補算並存檔
                logv = np.log2(expr[smp] + 1).values
                cut1, cut2, _ = find_cutpoints(logv)
                gs = {
                    g: ('unexpressed' if (lv := np.log2(v + 1)) < cut1
                                 else ('unknown' if lv < cut2 else 'expressed'))
                    for g, v in expr[smp].items()
                }
                with open(status_json, "w") as fp:
                    json.dump(gs, fp)

        else:
            tpm=expr[smp]
            cut1, cut2, _ = find_cutpoints(np.log2(tpm + 1).values)
            cut_summary.append({'sample':smp,'cut1':cut1,'cut2':cut2})
            # density plot
            # -----------------------------------------------------------------
            p95=np.percentile(tpm,PERCENTILE) or 1.0
            gs={g:('unexpressed' if (lv:=np.log2(v+1))<cut1 else ('unknown' if lv<cut2 else 'expressed')) for g,v in tpm.items()}
            sc={g:min(v/p95,1.0) for g,v in tpm.items()}
            gs.update({g:'unexpressed' for g in all_genes-gs.keys()}); sc.update({g:0.0 for g in all_genes-sc.keys()})
            m=model_base.copy()
            for r in m.reactions:
                if r.id not in core_rxns and rxn_inactive(r,gs):
                    r.lower_bound=r.upper_bound=0; continue
                if r.id in core_rxns: continue
                if r.genes:
                    gids=[g.id for g in r.genes]
                    scale=min(sc[g] for g in gids) if 'and' in (r.gene_reaction_rule or '').lower() else max(sc[g] for g in gids)
                else: scale=1.0
                scale=max(scale, SCALE_FLOOR)
                if r.upper_bound>0: r.upper_bound*=scale
                if r.lower_bound<0: r.lower_bound*=scale
            if SAVE_EFLUX:
                os.makedirs(pre_scale_dir,exist_ok=True)
                for rxn in m.reactions:
                    rxn.lower_bound=float(rxn.lower_bound or 0.0)
                    rxn.upper_bound=float(rxn.upper_bound or 0.0)
                    rxn.reversible=(rxn.lower_bound<0)
                write_sbml_model(m,pre_path)
                with open(f"{pre_scale_dir}/{smp}_gene_status.json", "w") as fp:
                    json.dump(gs, fp)
        try: m.solver.configuration.threads=1
        except AttributeError: pass
        active_rxn=[r.id for r in m.reactions if abs(r.lower_bound)+abs(r.upper_bound)>1e-6]
        m.objective=next(r for r in m.reactions if 'biomass' in r.id.lower())
        fva=flux_variability_analysis(m,reaction_list=active_rxn,fraction_of_optimum=FVA_FRAC,processes=N_PROCESSES)
        fva['sample']=smp; rec_FVA.append(fva.reset_index())
        if RUN_SAMPLING:
            for chain in range(N_CHAINS):                 # 兩條獨立鏈
                sampler = OptGPSampler(m,
                    processes=1,
                    thinning=THINNING,
                    seed=62,
                    remove_redundant=False,   # ★ 關鍵：禁用相關矩陣
                    fraction_of_optimum=0.8   # ★ 可選：再加速 FVA warm-up
                )
                part = sampler.sample(N_BURN + N_KEEP)    # DataFrame
                part = part.astype(DTYPE)
                part["chain"] = chain
                rec_samp.append(part)
        # --- build reaction_status (long) ----------------------------------
        for rxn in m.reactions:
            glist   = [g.id for g in rxn.genes]
            gstatus = [gs.get(g.id, gs.get(_norm(g.id), "unknown")) for g in rxn.genes] 
            # ── ① 依基因狀態決定 call ──────────────────────────────
            if not glist:                                # 沒有任何 GPR
                call = "noncore"                         # 或改成 "core" 皆可
            elif any(s == "expressed" for s in gstatus):
                call = "core"
            elif all(s == "unexpressed" for s in gstatus):
                call = "inactive"
            else:                                        # 全 unknown 或混合
                call = "noncore"
            # ── ② 硬保護：只在「將被關閉」時救回 ─────────────────
            if call == "inactive" and rxn.id in core_rxns:
                call = "core"
            # ── 記錄 ────────────────────────────────────────────
            reaction_status_records.append({
                "sample"   : smp,
                "reaction" : rxn.id,
                "gene_list": ";".join(glist),
                "statuses" : ";".join(gstatus),
                "call"     : call,
            })
        # -------------------------------------------------------------------
        print(f"[Done] {smp} | {time.time()-t1:.1f}s")

    # -------- save tables --------
    pd.DataFrame(reaction_status_records).to_csv(f"{out_dir}/reaction_status.csv", index=False)
    pd.DataFrame(rec_pFBA).to_csv(f"{out_dir}/pFBA_flux.csv",index=False)
    pd.concat(rec_FVA,ignore_index=True).rename(columns={'minimum':'min_flux','maximum':'max_flux'}).to_csv(f"{out_dir}/FVA_ranges.csv",index=False)
    if False:  # rec_samp undefined; disabled by fix
        pd.concat(rec_samp,ignore_index=True).to_csv(f"{out_dir}/flux_samples.csv.gz",index=False,compression='gzip')
    pd.DataFrame(cut_summary).to_csv(f"{out_dir}/cutoff_summary.csv",index=False)

    # -------- visuals --------
    if PLOT_ALL:
        sns.set(style='whitegrid',font_scale=0.8)
        # counts boxplot
        stat_df = pd.DataFrame(
            [{'sample':d['sample'],'count':d['flux'],'type':'active_rxn'}
                for d in rec_pFBA if d['reaction']=='__STAT__active_rxn'] +
            [{'sample':d['sample'],'count':d['flux'],'type':'active_met'}
                for d in rec_pFBA if d['reaction']=='__STAT__active_met'] +
            [{'sample':d['sample'],'count':d['flux'],'type':'expr_gene'}
                for d in rec_pFBA if d['reaction']=='__STAT__expr_gene']
        )

        stat_df=stat_df.merge(group_df,on='sample')
        # --- 三張獨立箱鬍 -------------------------------------------------
        for tp in ["active_rxn", "active_met", "expr_gene"]:
            sub = stat_df[stat_df["type"] == tp]
            plt.figure(figsize=(3,3))
            sns.boxplot(
                data=sub, x='group', y='count', hue='group',
                showfliers=False, legend=False,
                palette={'Control': 'skyblue', 'SCA3': 'salmon'}
            )
            plt.ylabel("count"); plt.title(tp.replace("_", " "))
            plt.tight_layout()
            plt.savefig(f"{out_dir}/{tp}_box.png", dpi=300)
            plt.close()

        # pFBA flux matrix
        flux_df=pd.DataFrame([d for d in rec_pFBA if '__STAT__' not in d['reaction']]).merge(group_df,on='sample',how='left')
        # ── Heatmap: remove any NaN/±Inf *rows與列* ─────────────
        flux_mat = (
            flux_df[~flux_df.reaction.str.startswith("__STAT__")]
                    .pivot(index="reaction", columns="sample", values="flux")
                    .astype(float)
                    .replace([np.inf, -np.inf], np.nan)
        )
        flux_mat.dropna(axis=0, how="any", inplace=True)   # 刪含 NaN 的反應
        flux_mat.dropna(axis=1, how="any", inplace=True)   # 刪含 NaN 的樣本
        sel = flux_mat.var(axis=1).nlargest(500).index
        print("flux_mat shape →", flux_mat.shape)
        print("flux_mat columns →", flux_mat.columns.tolist())   
        # 安全檢查
        
        if flux_mat.shape[1] < 2:
            print("⚠️  只剩 1 個樣本，跳過 clustermap")
        else:
            Z = flux_mat.loc[sel].apply(lambda r: (r - r.mean())/r.std(ddof=0), axis=1).fillna(0)
            assert np.isfinite(Z.values).all(), "non‑finite left!"
            sns.clustermap(Z, cmap="vlag", robust=True, figsize=(8,6))
            plt.savefig(out_dir/"flux_clustermap.png", dpi=300)
            plt.close()

        # PCA w/ explained variance
        pca=PCA(2).fit(flux_mat.T); var=pca.explained_variance_ratio_*100; scores=pca.transform(flux_mat.T)
        p_df=pd.DataFrame(scores,index=flux_mat.columns,columns=['PC1','PC2']).merge(group_df.set_index('sample'),left_index=True,right_index=True)
        p_df.to_csv(f"{out_dir}/PCA_pFBA_scores.csv")
        plt.figure(figsize=(5,4)); sns.scatterplot(data=p_df,x='PC1',y='PC2',hue='group',s=70)
        plt.xlabel(f"PC1 ({var[0]:.1f}%)"); plt.ylabel(f"PC2 ({var[1]:.1f}%)"); plt.tight_layout(); plt.savefig(f"{out_dir}/PCA_pFBA.png",dpi=300); plt.close()
        # UMAP
        try:
            import umap
            um=umap.UMAP(n_neighbors=5,min_dist=0.1,random_state=42).fit_transform(flux_mat.T)
            u_df=pd.DataFrame(um,index=flux_mat.columns,columns=['U1','U2']).merge(group_df.set_index('sample'),left_index=True,right_index=True)
            u_df.to_csv(f"{out_dir}/UMAP_pFBA_coords.csv")
            plt.figure(figsize=(5,4)); sns.scatterplot(data=u_df,x='U1',y='U2',hue='group',s=70); plt.tight_layout(); plt.savefig(f"{out_dir}/UMAP_pFBA.png",dpi=300); plt.close()
        except ImportError: pass
        # Welch test & significant box
        rows = []
        for rid in flux_mat.index:                     # 每個反應
            ctrl = flux_df[(flux_df["reaction"] == rid) & (flux_df["group"] == 'Control')]["flux"]
            sca  = flux_df[(flux_df["reaction"] == rid) & (flux_df["group"] == 'SCA3')]["flux"]
            if len(ctrl) < 1 or len(sca) < 1:
                continue
            stat, p = ttest_ind(ctrl, sca, equal_var=False, nan_policy='omit')
            rows.append({"reaction": rid,
                         "ctrl_mean": ctrl.mean(),
                         "sca3_mean": sca.mean(),
                         "pval": p})
        diff_tbl = pd.DataFrame(rows).set_index('reaction')
        if not diff_tbl.empty:
            diff_tbl["FDR"] = multipletests(diff_tbl.pval, method='fdr_bh')[1]
            diff_tbl["log2FC"] = np.log2(diff_tbl.sca3_mean + 1e-9) - \
                                 np.log2(diff_tbl.ctrl_mean + 1e-9)
            diff_tbl.to_csv(f"{out_dir}/diff_flux_pFBA.csv")
            sig = diff_tbl[diff_tbl.FDR < 0.10].index
            print(f"[Diff] {(diff_tbl.FDR<0.10).sum()} reactions FDR<0.10")
            if len(sig):
                plt.figure(figsize=(max(10,len(sig)*1.2),4)); sns.boxplot(data=flux_df[flux_df['reaction'].isin(sig)],x='reaction',y='flux',hue='group',showfliers=False); plt.xticks(rotation=45,ha='right'); plt.tight_layout(); plt.savefig(f"{out_dir}/sig_rxn_box.png",dpi=300); plt.close()
        # OptGP mean PCA/UMAP
        if False:  # rec_samp undefined; disabled by fix
            samp_df=pd.concat(rec_samp,ignore_index=True); mean_samp=samp_df.groupby('sample').mean(numeric_only=True)
            samp_df = pd.concat(rec_samp, ignore_index=True)   # sample × reaction
           # ① 重新 pivot → wide table，並保留 chain
            samp_df = samp_df.copy()
            samp_df["draw"] = samp_df.groupby("chain").cumcount()
            wide = (samp_df
                    .set_index(["chain", "draw"])
                    .sort_index()) 
            wide = wide.select_dtypes(include=["number"])                    
            wide = (wide.replace([np.inf, -np.inf], np.nan)
                         .fillna(0))          # 確保純 float
            
            n_chain = wide.index.levels[0].size
            n_draw  = wide.index.levels[1].size
            
            xa = xr.DataArray(
                    wide.values.reshape(n_chain, n_draw, -1),  # ★
                    dims   = ("chain", "draw", "reaction"),
                    coords = {"reaction": wide.columns}
            )        
            idata = az.from_dict(posterior={"flux": xa})
            rhat    = az.rhat(idata)
            try:
                geweke_z = az.stats.diagnostics.geweke(idata)
            except AttributeError:
                try:
                    from arviz.stats.diagnostics import geweke as geweke_fn
                    geweke_z = geweke_fn(idata)
                except ImportError:
                    print("⚠️  此版本 ArviZ 無 Geweke，跳過")
                    geweke_z = None
            if geweke_z is not None:
                geweke_z.to_dataframe(name="geweke_z").to_csv(out_dir/"geweke.csv") 
            rhat.to_dataframe(name="rhat").to_csv(f"{out_dir}/rhat.csv")
            pca2=PCA(2).fit(mean_samp); v2=pca2.explained_variance_ratio_*100; sc2=pca2.transform(mean_samp)
            p2=pd.DataFrame(sc2,index=mean_samp.index,columns=['PC1','PC2']).merge(group_df.set_index('sample'),left_index=True,right_index=True)
            plt.figure(figsize=(5,4)); sns.scatterplot(data=p2,x='PC1',y='PC2',hue='group',s=70); plt.xlabel(f"PC1 ({v2[0]:.1f}%)"); plt.ylabel(f"PC2 ({v2[1]:.1f}%)"); plt.tight_layout(); plt.savefig(f"{out_dir}/PCA_optgp.png",dpi=300); plt.close()
            try:
                um2=umap.UMAP(n_neighbors=5,min_dist=0.1,random_state=42).fit_transform(mean_samp)
                u2=pd.DataFrame(um2,index=mean_samp.index,columns=['U1','U2']).merge(group_df.set_index('sample'),left_index=True,right_index=True)
                plt.figure(figsize=(5,4)); sns.scatterplot(data=u2, x='U1', y='U2', hue='group', s=70); plt.tight_layout(); plt.savefig(f"{out_dir}/UMAP_optgp.png",dpi=300); plt.close()
            except Exception: pass
    print(f"✅ All finished in {(time.time()-t0)/60:.1f} min | results in {out_dir}")

# ──────────────────────────────────────────────────────────────────────────────
#                        PARALLEL DRIVER (Append‑only)
# ──────────────────────────────────────────────────────────────────────────────

import argparse, os
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import warnings

# Global lower bound floor for e-flux scaling
SCALE_FLOOR = 0.01

from joblib import Parallel, delayed

try:
    from cobra.flux_analysis import pfba, flux_variability_analysis
    from cobra.sampling import OptGPSampler
except ImportError:
    pfba = flux_variability_analysis = sample = None  # fallback for type‑checker
GROUP_CSV = Path("D:/cch_data/mice_RNAdata_for_AST/GSEA_R_result/sample_groups.csv")
N_CORES: int   = 4      # physical cores
EXPR_TSV: Optional[Path] = None
# ── Model builder stub ─────────────────────────────────────────────────
def build_context_model(sample_id: str, expr_df: pd.DataFrame):
    """
    回傳 (model, gene_status_dict)。範例以你的舊 for-loop 為藍本；
    若已有 make_model_for_sample()，改成 `return make_model_for_sample(...)`。
    """
    m = model_base.copy() 
    with open(core_rxn_txt) as f:
        core_rxns = {l.strip() for l in f if l.strip()}
    core_rxns |= {r.id for r in model_base.reactions if 'biomass' in r.id.lower()}
    # ---------------------------------------------------------------
    tpm = expr_df[sample_id]
    cut1, cut2, _ = find_cutpoints(np.log2(tpm + 1).values)
    p95 = np.percentile(tpm, PERCENTILE) or 1.0

    gs = {}
    for gid, v in tpm.items():
        lv = np.log2(v + 1)
        status = ('unexpressed' if lv < cut1
                  else ('unknown' if lv < cut2 else 'expressed'))
        gs[gid] = status          # 原始 key，例如 "G_12345"
        gs[_norm(gid)] = status   # 正規化 key，例如 "12345"
    missing = {g.id for g in m.genes} - gs.keys()
    gs.update({gid: "unexpressed" for gid in missing})
    sc = {}
    for gid, v in tpm.items():
        val = min(v / p95, 1.0)
        sc[gid]          = val            # 原始 key，如 '268860', 'G_12345'
        sc[_norm(gid)]   = val            # 正規化 key，如 '268860'
    def _sc(gid: str) -> float:
        return sc.get(gid, sc.get(_norm(gid), 1.0))
    for r in m.reactions:
        if r.id in core_rxns:
            continue
        if rxn_inactive(r, gs):
            r.lower_bound = r.upper_bound = 0
            continue
        if r.genes:
            vals = [_sc(g.id) for g in r.genes]   # ← 只用 _sc()
            rule = r.gene_reaction_rule or ""
            scale = min(vals) if "and" in rule.lower() else max(vals)
        else:
            scale = 1.0
        scale = max(scale, SCALE_FLOOR)
        if r.upper_bound > 0:
            r.upper_bound *= scale
        if r.lower_bound < 0:
            r.lower_bound *= scale
    return m, gs, core_rxns 
# ── Worker function ───────────────────────────────────────────────────────────
def analyse_one(sample_id: str, expr_df: pd.DataFrame) -> Tuple[str, Path]:
    # ── ① 在子行程內關掉冗餘檢查 ─────────────────────
    from cobra.sampling.hr_sampler import HRSampler
    import numpy as np
    def _no_redundant(self, mat):
        # mat.shape[0] = warm-up 向量數（行）
        return np.zeros(mat.shape[0], dtype=bool)
    HRSampler._is_redundant = _no_redundant        # 標記一次即可
    # ── 先印參數，馬上回饋 ───────────────────────────
    steps = (N_BURN + N_KEEP) * THINNING
    print(f"[{sample_id}]  N_BURN={N_BURN}  N_KEEP={N_KEEP}  "
          f"THINNING={THINNING}  ⇒  steps={steps}",
          flush=True)  
              # 先建立輸出資料夾，供 ESS 圖存檔
    out_dir = OUTDIR / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model, gs, core_rxns = build_context_model(sample_id, expr_df)
    m = model
    reaction_status = [
        {
            "sample": sample_id,
            "reaction": rxn.id,
            "gene_list": ";".join(g.id for g in rxn.genes),
            "statuses": ";".join(gs.get(g.id, gs.get(_norm(g.id), "unknown")) for g in rxn.genes),
            "call": "core" if rxn.id in core_rxns else ("inactive" if rxn_inactive(rxn, gs) else "noncore")
        } for rxn in model.reactions
    ]
    pd.DataFrame(reaction_status).to_csv(out_dir / "reaction_status.csv", index=False)
    cut1, cut2, _ = find_cutpoints(np.log2(expr_df[sample_id] + 1).values)
    pd.DataFrame([{"sample": sample_id, "cut1": cut1, "cut2": cut2}]).to_csv(out_dir / "cutoff_summary.csv", index=False)

    try:
        model.solver.configuration.threads = 1   # 單執行緒求解器
    except AttributeError:
        pass
    # 先檢查 biomass 是否有被完全鎖死 ------------------------
    bio_rxn = next(r for r in model.reactions if 'biomass' in r.id.lower())
    if bio_rxn.upper_bound == 0:
        bio_rxn.upper_bound = 1000       # 給一個寬鬆上界
    if bio_rxn.lower_bound < 0:
        bio_rxn.lower_bound = 0          # 禁止負生長
    
    # pFBA 容忍 1% optimum（若最優是 0 也能找近似可行解）
    pfba_sol = pfba(model, fraction_of_optimum=0.01)  
    flux_series = pfba_sol.fluxes.replace([np.inf, -np.inf], 0)
    flux_series.rename_axis("reaction")            \
           .to_csv(out_dir / "pfba_flux.csv", header=["flux"])        
    fva_df   = flux_variability_analysis(model, fraction_of_optimum=FVA_FRAC, processes=1)
    # ── OptGP sampling ───────────────────────
    rec = []
    for c in range(N_CHAINS):
        sampler = OptGPSampler(m,
            processes=1,
            thinning=THINNING,
            seed=62+c,
            remove_redundant=False,   # ★ 關鍵：禁用相關矩陣
            fraction_of_optimum=0.8   # ★ 可選：再加速 FVA warm-up
        )
        part = sampler.sample(N_BURN + N_KEEP)    # DataFrame
        part = part.astype(DTYPE)
        part["chain"] = c
        rec.append(part.iloc[N_BURN:])
    optgp_df = pd.concat(rec, ignore_index=True)
    # --- 轉 xarray → Zarr（含 chain 維度） ------------------------------
    zarr_path = out_dir / "optgp.zarr"
    # 1) 反應欄位（排除 chain）
    rxn_cols = [c for c in optgp_df.columns if c != "chain"]
    # 2) 疊成 (chain, draw, reaction)
    samples = np.stack(
        [df[rxn_cols].to_numpy(dtype=DTYPE)      # shape = (draw, reaction)
         for df in rec]                          # len(rec) = N_CHAINS
    )                                            # → (chain, draw, reaction)
    # 3) DataArray 取名 xa（或 x，名稱一致就好）
    xa = xr.DataArray(
            da.from_array(samples, chunks=(1, Z_CHUNK_DRAW, -1)),
            dims=("chain", "draw", "reaction"),
            coords={"chain": np.arange(samples.shape[0]),
                    "reaction": rxn_cols},
            name="flux",
    )
    # 4) 真正寫 Zarr；用同一變數名 xa
    xr.Dataset({"flux": xa}).to_zarr(
        zarr_path,
        mode="w",
        encoding={"flux": {"compressor": COMPRESSOR,
                           "dtype": DTYPE,
                           "chunks": (1, Z_CHUNK_DRAW, -1)}},
    )
    print(f"[✓ ZARR] {sample_id} → {zarr_path}", flush=True)
    # ── ESS & 收斂圖（僅在有取樣時執行） ──────────────────
    if N_KEEP > 0:
        ess = compute_ess_zarr(zarr_path, n_chains=N_CHAINS)
        ess_df = ess.reset_index().rename(columns={0: "ESS"})
        ess_df.to_csv(out_dir / "ess.csv", index=False)

        if ess.size >= 2:
            plt.figure(figsize=(4, 3))
            sns.histplot(ess, bins=40, color="steelblue")
            plt.axvline(200, c="red", ls="--", lw=.8)
            plt.xlabel("Effective sample size"); plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(out_dir / "ess_hist.png", dpi=300)
            plt.close()

        target_rxns = ["ATPM", "EX_glc__D_e","GLUt4","GLUt2m", "HKt", "PFK", "PYK","PDHm", "ACCOAC","PEPCKm"]
        xa = xr.open_zarr(zarr_path).flux   # dims: chain, draw, reaction
        for rid in target_rxns:
            if rid in xa.coords["reaction"].values:
                # trace（每條鏈一條線）
                plt.figure(figsize=(4, 2))
                sel = xa.sel(reaction=rid)
                for c in sel.coords["chain"].values:
                    plt.plot(sel.sel(chain=c).values, lw=.5, alpha=.9, label=f"chain {int(c)}")
                plt.title(f"{rid} trace – {sample_id}")
                plt.legend(loc="upper right", fontsize=7, frameon=False)
                plt.tight_layout(); plt.savefig(out_dir / f"{rid}_trace.png", dpi=300); plt.close()        
                # ACF（每條鏈一條線）
                plt.figure(figsize=(3, 2))
                for c in sel.coords["chain"].values:
                    s = pd.Series(sel.sel(chain=c).values)
                    pd.plotting.autocorrelation_plot(s)
                plt.title(f"{rid} ACF")
                plt.tight_layout(); plt.savefig(out_dir / f"{rid}_acf.png", dpi=300); plt.close()
    # ── Active counts ‑‑ based on pFBA non‑zero flux ─────────────
    nz_mask   = flux_series.abs() > 1e-9        # 1 e‑9 作為非零門檻
    num_rxn   = int(nz_mask.sum())
    
    active_rxn_ids = flux_series.index[nz_mask]
    active_rxns    = [model.reactions.get_by_id(r) for r in active_rxn_ids]
    active_mets    = {m.id for rxn in active_rxns for m in rxn.metabolites}
    num_met        = len(active_mets)
    
    num_gene = int(expr_df.loc[:, sample_id].gt(0).sum())
    pd.DataFrame([
        {"reaction": "__STAT__active_rxn", "flux": num_rxn},
        {"reaction": "__STAT__active_met", "flux": num_met},
        {"reaction": "__STAT__expr_gene",  "flux": num_gene},
    ]).to_csv(out_dir / "pfba_flux.csv", mode="a", header=False, index=False)
    fva_df.to_csv(out_dir / "fva.csv")
    return sample_id, out_dir
def diff_optgp(out_dir: Path, group_df: pd.DataFrame):
    # 1) 蒐集所有 Zarr
    files = list(out_dir.glob("*/optgp.zarr"))
    if not files:
        print("❗ 沒有找到任何 optgp.zarr，跳過差異分析"); return

    # 2) 取反應欄位
    rxn_cols = xr.open_zarr(files[0]).flux.reaction.values.tolist()
    # ③ 讀入所有 sample → samp_df dict
    samp_df: dict[str, pd.DataFrame] = {}
    for f in files:
        sid = f.parent.name
        try:
            ds = xr.open_zarr(f, chunks={"draw": Z_CHUNK_DRAW})
            df = (
                ds.flux                                  # (chain, draw, reaction)
                  .stack(sample_dim=("chain", "draw"))   # → (sample_dim, reaction)
                  .to_pandas()
                  .rename_axis("draw")
            )
            samp_df[sid] = df
        except Exception as e:
            print(f"✗ 讀檔失敗 {sid}: {e}")

    if not samp_df:         
        print("❗ 所有 Zarr 都讀失敗，跳過差異分析"); return
    # === 建立 med_df：每個 sample 的反應中位數 + group 資訊 ==================
    # samp_df: {sample_id: DataFrame(draw × reaction)}
    # 1) 明確命名 concat 外層 key = 'sample'
    med_stack = pd.concat(samp_df, names=["sample"])   # index: (sample, draw) 或 (sample, chain_draw)
    # 2) 以 sample 做中位數（只對數值欄）
    med_mid = med_stack.groupby(level="sample").median(numeric_only=True)   
    # 3) 扁平化欄位（若是 MultiIndex，取最後一層當作反應名）
    if isinstance(med_mid.columns, pd.MultiIndex):
        med_mid.columns = med_mid.columns.get_level_values(-1)    
    # 4) 把 sample 變成欄位，接回 group
    med_mid = med_mid.reset_index()                    # 有一欄 'sample'
    if "sample" in group_df.columns:
        grp_map = group_df.drop_duplicates("sample").set_index("sample")["group"]
    else:
        grp_map = group_df["group"] if "group" in group_df.columns else group_df
    med_mid["sample"] = med_mid["sample"].astype(str)
    grp_map.index      = grp_map.index.astype(str)
    med_mid["group"]   = med_mid["sample"].map(grp_map)    
    # 5) 這就是最終要用的 med_df
    med_df = med_mid    
    # 6) 反應欄位清單（排除非反應欄）
    rxn_cols = [c for c in med_df.columns if c not in ("sample", "group")]  
    # --- median Welch t-test ------------------------------------------------
    rows = []
    for rxn in rxn_cols:
        # 取兩組數值、清 NaN/Inf
        ctrl = pd.to_numeric(med_df.loc[med_df["group"] == "Control", rxn], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        dise = pd.to_numeric(med_df.loc[med_df["group"] == "SCA3",    rxn], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    
        if ctrl.size < 2 or dise.size < 2:          # 至少各 2 個樣本才做 t-test
            continue    
        stat, p = ttest_ind(ctrl, dise, equal_var=False, nan_policy="omit")
        rows.append({
            "reaction": rxn,
            "ctrl_mean": float(ctrl.mean()),
            "sca3_mean": float(dise.mean()),
            "pval": float(p),
        })
    res = pd.DataFrame(rows).set_index("reaction")
    # ── median Welch t-test 結果寫回 ───────────────────
    if res.empty:
        print("[OptGP diff] median 表空，樣本數不足或全部變異為 0")
    else:
        mask = res["pval"].notna()          # ← 只有這裡才建立 mask
        if mask.any():                      # 至少有一個可用 p‑value
            res.loc[mask, "fdr"] = multipletests(
                res.loc[mask, "pval"], method="fdr_bh"
            )[1]
        else:
            res["fdr"] = np.nan
    
        res["log2FC"] = np.log2(res.sca3_mean + 1e-9) - \
                        np.log2(res.ctrl_mean + 1e-9)
        res.to_csv(out_dir / "diff_flux_optGP_median.csv")
        sig = (res["fdr"] < 0.10).sum()
        print(f"[OptGP diff] median+t‑test → {sig} rxns FDR<0.10")

    # --- 2. KS-test on full distributions ------------------------------------
    ks_rows = []
    for rxn in samp_df[next(iter(samp_df))].columns:     # iterate reactions
        ctrl_ids = group_df[group_df["group"] == "Control"]["sample"]
        ctrl_all = np.concatenate([samp_df[s][rxn].dropna().values for s in ctrl_ids])

        dise_ids = group_df[group_df["group"] == "SCA3"]["sample"]
        dise_all = np.concatenate([samp_df[s][rxn].dropna().values for s in dise_ids])

        if len(ctrl_all) == 0 or len(dise_all) == 0:
            continue
        stat, p = ks_2samp(ctrl_all, dise_all, alternative="two-sided", mode="asymp")
        ks_rows.append({"reaction": rxn, "D": stat, "p": p})
    ks = pd.DataFrame(ks_rows)
    ks["fdr"] = multipletests(ks.p, method="fdr_bh")[1]
    ks.to_csv(out_dir / "diff_flux_optGP_KS.csv", index=False)

    print(f"[OptGP diff] median+t-test → {res.fdr.lt(0.10).sum()} rxns FDR<0.10")
    print(f"[OptGP diff] KS-test      → {ks.fdr.lt(0.10).sum()} rxns FDR<0.10")
# ── Parallel runner ───────────────────────────────────────────────────────────
def aggregate_results(out_dir: Path, group_df: pd.DataFrame):
    group_df = group_df.rename(columns=lambda c: c.lower().strip())
    # 讀回所有 pFBA
    flux_df = (
        pd.concat([pd.read_csv(fn).assign(sample=fn.parent.name)
                   for fn in out_dir.glob("*/pfba_flux.csv")],
                  ignore_index=True)
          .merge(group_df, on="sample", how="left")        # ← 只 merge 一次
    )
    for col in flux_df.columns:
        if col.startswith("group") and col != "group":
            flux_df = flux_df.rename(columns={col: "group"})
    flux_df["flux"] = pd.to_numeric(flux_df["flux"], errors="coerce")  
    rec_pFBA_list = flux_df.to_dict("records")             # ← 改名，避免覆蓋
    # 讀回所有 OptGP (改 Zarr)
    samp_stats = []
    for f in out_dir.glob("*/optgp.zarr"):
        sid = f.parent.name
        ds  = xr.open_zarr(f, chunks={"draw": Z_CHUNK_DRAW})
        mean = ds.flux.mean(dim="draw").to_pandas()
        mean["sample"] = sid
        samp_stats.append(mean)
    
    if samp_stats:
        samp_df = pd.concat(samp_stats, ignore_index=True)
    else:
        samp_df = pd.DataFrame()        # 保留與下游一致
    # 3. 把「原 PLOT_ALL」段落整段貼進來，
    #    並把懸掛的全域名改用剛剛建立的 rec_pFBA / rec_samp
    # -------- visuals --------
    if PLOT_ALL:
        sns.set(style='whitegrid',font_scale=0.8)
        # counts boxplot
        stat_df = pd.DataFrame(
            [{'sample':d['sample'],'count':d['flux'],'type':'active_rxn'}
                for d in rec_pFBA_list if d['reaction']=='__STAT__active_rxn'] +
            [{'sample':d['sample'],'count':d['flux'],'type':'active_met'}
                for d in rec_pFBA_list if d['reaction']=='__STAT__active_met'] +
            [{'sample':d['sample'],'count':d['flux'],'type':'expr_gene'}
                for d in rec_pFBA_list if d['reaction']=='__STAT__expr_gene']
        )
        stat_df=stat_df.merge(group_df,on='sample')
        # --- 三張獨立箱鬍 -------------------------------------------------
        for tp in ["active_rxn", "active_met", "expr_gene"]:
            sub = stat_df[stat_df["type"] == tp]
            plt.figure(figsize=(3,3))
            sns.boxplot(
                data=sub, x='group', y='count', hue='group',
                showfliers=False, legend=False,
                palette={'Control': 'skyblue', 'SCA3': 'salmon'}
            )
            plt.ylabel("count"); plt.title(tp.replace("_", " "))
            plt.tight_layout()
            plt.savefig(f"{out_dir}/{tp}_box.png", dpi=300)
            plt.close()
        # pFBA flux matrix
        flux_df2 = (
            pd.DataFrame([d for d in rec_pFBA_list
                          if '__STAT__' not in d['reaction']])
            )
        print("flux_df2 columns →", flux_df2.columns.tolist())
        print("group unique →", flux_df2.get("group", pd.Series()).unique()[:5])
        # ── Heatmap: remove any NaN/±Inf *rows與列* ─────────────
        flux_mat = (
            flux_df2[~flux_df2.reaction.str.startswith("__STAT__")]
                    .pivot(index="reaction", columns="sample", values="flux")
                    .astype(float)
                    .replace([np.inf, -np.inf], np.nan)
        )
        flux_mat.dropna(axis=0, how="any", inplace=True)   # 刪含 NaN 的反應
        flux_mat.dropna(axis=1, how="any", inplace=True)   # 刪含 NaN 的樣本
        sel = flux_mat.var(axis=1).nlargest(500).index
        print("flux_mat shape →", flux_mat.shape)
        print("flux_mat columns →", flux_mat.columns.tolist())   

        if flux_mat.shape[1] < 2:
            print("⚠️  只剩 1 個樣本，跳過 clustermap")
        else:
            Z = flux_mat.loc[sel].apply(lambda r: (r - r.mean())/r.std(ddof=0), axis=1).fillna(0)
                    # 安全檢查
            assert np.isfinite(Z.values).all(), "non‑finite left!"
            sns.clustermap(Z, cmap="vlag", robust=True, figsize=(8,6))
            plt.savefig(out_dir/"flux_clustermap.png", dpi=300)
            plt.close()

        # PCA w/ explained variance
        pca=PCA(2).fit(flux_mat.T); var=pca.explained_variance_ratio_*100; scores=pca.transform(flux_mat.T)
        p_df=pd.DataFrame(scores,index=flux_mat.columns,columns=['PC1','PC2']).merge(group_df.set_index('sample'),left_index=True,right_index=True)
        p_df.to_csv(f"{out_dir}/PCA_pFBA_scores.csv")
        plt.figure(figsize=(5,4)); sns.scatterplot(data=p_df,x='PC1',y='PC2',hue='group',s=70)
        plt.xlabel(f"PC1 ({var[0]:.1f}%)"); plt.ylabel(f"PC2 ({var[1]:.1f}%)"); plt.tight_layout(); plt.savefig(f"{out_dir}/PCA_pFBA.png",dpi=300); plt.close()
        # UMAP
        try:
            import umap
            um=umap.UMAP(n_neighbors=5,min_dist=0.1,random_state=42).fit_transform(flux_mat.T)
            u_df=pd.DataFrame(um,index=flux_mat.columns,columns=['U1','U2']).merge(group_df.set_index('sample'),left_index=True,right_index=True)
            u_df.to_csv(f"{out_dir}/UMAP_pFBA_coords.csv")
            plt.figure(figsize=(5,4)); sns.scatterplot(data=u_df,x='U1',y='U2',hue='group',s=70); plt.tight_layout(); plt.savefig(f"{out_dir}/UMAP_pFBA.png",dpi=300); plt.close()
        except ImportError: pass
        # Welch test & significant box
        rows = []
        for rid in flux_mat.index:                     # 每個反應
            ctrl = flux_df2[(flux_df2.reaction == rid) & (flux_df2["group"] == 'Control')]['flux']
            sca  = flux_df2[(flux_df2.reaction == rid) & (flux_df2["group"] == 'SCA3')]['flux']
            if len(ctrl) < 1 or len(sca) < 1:
                continue
            stat, p = ttest_ind(ctrl, sca, equal_var=False, nan_policy='omit')
            rows.append({"reaction": rid,
                         "ctrl_mean": ctrl.mean(),
                         "sca3_mean": sca.mean(),
                         "pval": p})
        diff_tbl = pd.DataFrame(rows).set_index('reaction')
        if not diff_tbl.empty:
            diff_tbl["FDR"] = multipletests(diff_tbl.pval, method='fdr_bh')[1]
            slog2 = lambda x: np.sign(x) * np.log2(np.abs(x) + 1e-9)
            diff_tbl["log2FC"] = slog2(diff_tbl.sca3_mean) - slog2(diff_tbl.ctrl_mean)
            diff_tbl.to_csv(f"{out_dir}/diff_flux_pFBA.csv")
            sig = diff_tbl[diff_tbl.FDR < 0.10].index
            print(f"[Diff] {(diff_tbl.FDR<0.10).sum()} reactions FDR<0.10")
            if len(sig):
                plt.figure(figsize=(max(10,len(sig)*1.2),4)); sns.boxplot(data=flux_df2[flux_df2['reaction'].isin(sig)],x='reaction',y='flux',hue='group',showfliers=False); plt.xticks(rotation=45,ha='right'); plt.tight_layout(); plt.savefig(f"{out_dir}/sig_rxn_box.png",dpi=300); plt.close()
        # OptGP mean PCA/UMAP
        if False:  # rec_samp undefined; disabled by fix
            samp_df=pd.concat(rec_samp,ignore_index=True); mean_samp=samp_df.groupby('sample').mean(numeric_only=True)
            samp_df = pd.concat(rec_samp, ignore_index=True)   # sample × reaction
           # ① 重新 pivot → wide table，並保留 chain
            samp_df = samp_df.copy()
            samp_df["draw"] = samp_df.groupby("chain").cumcount()
            wide = (samp_df
                    .set_index(["chain", "draw"])
                    .sort_index())    
            wide = wide.select_dtypes(include=["number"])                    
            wide = (wide.replace([np.inf, -np.inf], np.nan)
                         .fillna(0))          # 確保純 float
            
            n_chain = wide.index.levels[0].size
            n_draw  = wide.index.levels[1].size
            
            xa = xr.DataArray(
                    wide.values.reshape(n_chain, n_draw, -1),  # ★
                    dims   = ("chain", "draw", "reaction"),
                    coords = {"reaction": wide.columns}
            )        
            idata = az.from_dict(posterior={"flux": xa})
            ess    = az.ess(idata, method="bulk")
            print(ess.quantile([0.05, 0.25, 0.5, 0.75, 0.95]))
            rhat   = az.rhat(idata)
            try:
                geweke_z = az.stats.diagnostics.geweke(idata)
            except AttributeError:
                try:
                    from arviz.stats.diagnostics import geweke as geweke_fn
                    geweke_z = geweke_fn(idata)
                except ImportError:
                    print("⚠️  此版本 ArviZ 無 Geweke，跳過")
                    geweke_z = None
            if geweke_z is not None:
                geweke_z.to_dataframe(name="geweke_z").to_csv(out_dir/"geweke.csv")         
            # ess 可能是 xarray.DataArray
            ess_df = ess.to_dataframe().reset_index()
            ess_df.columns = list(ess_df.columns[:-1]) + ["ess"]   # 最後一欄改名 ess
            ess_df.to_csv(out_dir / "ess.csv", index=False)
            rhat_df = rhat.to_dataframe().reset_index()
            rhat_df.columns = list(rhat_df.columns[:-1]) + ["rhat"]
            rhat_df.to_csv(out_dir / "rhat.csv", index=False)
            if geweke_z is not None:
                g_df = geweke_z.to_dataframe().reset_index()
                g_df.columns = list(g_df.columns[:-1]) + ["geweke_z"]
                g_df.to_csv(out_dir/"geweke.csv", index=False)
            pca2=PCA(2).fit(mean_samp); v2=pca2.explained_variance_ratio_*100; sc2=pca2.transform(mean_samp)
            p2=pd.DataFrame(sc2,index=mean_samp.index,columns=['PC1','PC2']).merge(group_df.set_index('sample'),left_index=True,right_index=True)
            plt.figure(figsize=(5,4)); sns.scatterplot(data=p2,x='PC1',y='PC2',hue='group',s=70); plt.xlabel(f"PC1 ({v2[0]:.1f}%)"); plt.ylabel(f"PC2 ({v2[1]:.1f}%)"); plt.tight_layout(); plt.savefig(f"{out_dir}/PCA_optgp.png",dpi=300); plt.close()
            try:
                um2=umap.UMAP(n_neighbors=5,min_dist=0.1,random_state=42).fit_transform(mean_samp)
                u2=pd.DataFrame(um2,index=mean_samp.index,columns=['U1','U2']).merge(group_df.set_index('sample'),left_index=True,right_index=True)
                plt.figure(figsize=(5,4)); sns.scatterplot(data=u2, x='U1', y='U2', hue='group', s=70); plt.tight_layout(); plt.savefig(f"{out_dir}/UMAP_optgp.png",dpi=300); plt.close()
            except Exception: pass

def run_parallel(sample_ids: List[str]):
    if not (pfba and flux_variability_analysis and OptGPSampler):
        raise RuntimeError("cobra 功能未匯入，請確認安裝 cobra≥0.29.1")
    if EXPR_TSV is None:
        raise ValueError("EXPR_TSV 未指定！")

    expr_df = pd.read_csv(EXPR_TSV, sep=None, engine="python", index_col=0)
    OUTDIR.mkdir(exist_ok=True)
    steps = (N_BURN + N_KEEP) * THINNING
    print(f"\n=== 參數確認 ===  BURN={N_BURN}  KEEP={N_KEEP}  "
          f"THINNING={THINNING}  ⇒  每樣本步數={steps}\n")
    for sid in sample_ids:
        print(f"  • {sid}")
    print("───────────────────────────────────\n")
    finished = Parallel(n_jobs=N_CORES, backend="loky", verbose=10)(
        delayed(analyse_one)(sid, expr_df) for sid in sample_ids
    )
    finished = [res for res in finished if res is not None]
    print("\n✓ 平行分析完成：")
    for sid, folder in finished:
        print(f"  {sid}: {folder}")
# ── CLI ───────────────────────────────────────────────────────────────────────
def _detect_samples(expr_path: Path) -> List[str]:
    if not expr_path or not expr_path.exists():
        return []
    cols = pd.read_csv(expr_path, sep=None, engine="python", nrows=1, index_col=0).columns
    return cols.tolist()[:12]
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="12‑core parallel driver")
    p.add_argument("--expr", help="expression matrix TSV 路徑")
    p.add_argument("--parallel", action="store_true", help="啟用平行外層")
    p.add_argument("--samples", nargs="+", help="樣本 ID 列表，不指定則取 expr 檔前 12 欄")
    args = p.parse_args()
    print("*args.samples=",args.samples)

    if args.expr:
        EXPR_TSV = Path(args.expr)
    if EXPR_TSV is None or not EXPR_TSV.exists():
        raise SystemExit(f"❌ EXPR_TSV 未找到：{EXPR_TSV}")
    if args.parallel:
        targets = args.samples or _detect_samples(Path(args.expr))
        if not targets:
            raise SystemExit("❌ 無法取得 sample IDs，請用 --samples 指定")
        run_parallel(targets)
        group_df = (
            pd.read_csv(GROUP_CSV, index_col=0)  # 原本 sample 在 index
              .reset_index()                     # 把 sample 升為欄位
        )
        diff_optgp(OUTDIR, group_df)
        aggregate_results(OUTDIR, group_df)
        sys.exit(0)
    else:
        # fallback to original main()
        if "main" in globals():
            globals()["main"]()
        else:
            raise SystemExit("找不到 original main()，請確認已貼入舊腳本全文！")