#!/usr/bin/env python3
# > conda activate gem-py39 (自創獨立分析環境，防止版本衝突引發的bug)
# > python "D:\cch_data\rxn2kegg.py"
"""
Generate rxn2kegg.csv from an SBML GEM (e.g. iMM1865.xml).

輸出格式：
reaction,pathway
R_r0321,map00650
R_r0321,map01212
...
"""
import re, time, json, sys
from pathlib import Path
import pandas as pd
import cobra
from bioservices import KEGG
try:
    # 舊版 bioservices (<1.10) 內建的錯誤類別
    from bioservices.kegg import KEGGParserError
except ImportError:
    # ➟ 新版已移除；自建 fallback，讓下方程式仍可捕捉例外
    class KEGGParserError(Exception):
        """Fallback for bioservices>=1.10 where KEGGParserError was removed."""
        pass

MODEL    = "D:\cch_data\mice_RNAdata_for_AST\GSEA_R_result\iMM1865.xml"  
BASE_DIR = Path(r"D:\cch_data\mice_RNAdata_for_AST\eflux_FVA_sampling")
OUT_CSV  = BASE_DIR / "rxn2kegg_mmu.csv"
CACHE    = "kegg_rxn2mmu.json"    # 避免重複查詢
ORG      = "mmu"

# ---------- 1. 從 SBML 撈出 Rxxxxx ----------
model = cobra.io.read_sbml_model(MODEL)
rxn2rid, pat = {}, re.compile(r"(?:/kegg\.reaction/)?(R\d{5})", re.I)
for rxn in model.reactions:
    hit = {m.group(1) for s in ([*rxn.annotation.keys(), *rxn.annotation.values()])
           for seg in (s if isinstance(s, (list, tuple, set)) else [s])
           for m in [pat.search(str(seg))] if m}
    if hit:
        rxn2rid[rxn.id] = sorted(hit)
print(f"Found KEGG reaction IDs for {len(rxn2rid)} / {len(model.reactions)} reactions")

# ---------- 2. Rxxxxx → mmuXXXXX ----------
try:
    cache = json.loads(Path(CACHE).read_text())
except FileNotFoundError:
    cache = {}

kegg, rows = KEGG(), []
for rxn_id, krxns in rxn2rid.items():
    for kr in krxns:
        if kr not in cache:                         # 先查快取
            try:
                # ▲ (a) 直接問 KEGG link 有沒有 mmu-pathway
                mmu_maps = [ln.split('\t')[1].split(':')[1]          # => mmu00620
                            for ln in kegg.link("pathway", kr).splitlines()
                            if ln.startswith(f"path:{ORG}")]
                if not mmu_maps:                                    # ▲ (b) fallback：map → mmu
                    entry = kegg.get(kr)
                    generic = [ln.split()[1]                        # map00620
                               for ln in entry.splitlines()
                               if ln.startswith("PATHWAY")]
                    mmu_maps = [ORG + g[3:] for g in generic]       # → mmu00620
                cache[kr] = mmu_maps
                time.sleep(0.2)
            except (KEGGParserError, Exception) as e:
                print("KEGG fetch error", kr, e, file=sys.stderr)
                cache[kr] = []
        for mp in cache[kr]:
            rows.append((rxn_id, mp))

# ---------- 3. 輸出 ----------
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
(pd.DataFrame(rows, columns=["reaction", "mmu_pathway"])
   .drop_duplicates()
   .to_csv(OUT_CSV, index=False))
Path(CACHE).write_text(json.dumps(cache))
print(f"✅ Saved → {OUT_CSV}  (rows={len(rows)})")
