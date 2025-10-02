#conda activate gem-py39 (自創獨立分析環境，防止版本衝突引發的bug)
# 基因層級（推薦用於 gene-level TPM）
#python D:\compute_lengths_and_tpm.py D:\ref_BDGP6_32\Drosophila_melanogaster.BDGP6.32.53.gtf -o D:\ref_BDGP6_32\gene_lengths.csv --mode gene
# 轉錄本層級（若你做 transcript-level TPM）
#python D:\compute_lengths_and_tpm.py D:\ref_BDGP6_32\Drosophila_melanogaster.BDGP6.32.53.gtf -o D:\ref_BDGP6_32\tx_lengths.csv --mode transcript
#!/usr/bin/env python3
# save as compute_lengths_and_tpm.py
import argparse, csv, sys
from collections import defaultdict

def parse_attrs(attr_field: str) -> dict:
    out = {}
    for kv in attr_field.strip().split(";"):
        kv = kv.strip()
        if not kv: 
            continue
        if " " not in kv:
            continue
        k, v = kv.split(" ", 1)
        out[k] = v.strip().strip('"')
    return out

def merge_len(intervals):
    if not intervals:
        return 0
    intervals.sort()
    merged = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e + 1:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return sum(e - s + 1 for s, e in merged)

def compute_lengths_from_gtf(gtf_path, mode="gene", protein_coding_only=False):
    # data structures
    if mode == "gene":
        exons = defaultdict(list)          # gene_id -> list[(start,end)]
        ginfo = {}                         # gene_id -> dict
    else:
        exons = defaultdict(list)          # transcript_id -> list[(start,end)]
        tinfo = {}                         # transcript_id -> dict with gene_id/name/biotype

    with open(gtf_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            chrom, _, feature, start, end, _, strand, _, attrs = fields
            if feature != "exon":
                continue
            a = parse_attrs(attrs)
            gene_id = a.get("gene_id")
            gene_name = a.get("gene_name", "")
            biotype = a.get("gene_biotype", a.get("gene_type", ""))
            if protein_coding_only and biotype != "protein_coding":
                continue
            s, e = int(start), int(end)

            if mode == "gene":
                if gene_id is None:
                    continue
                exons[gene_id].append((s, e))
                if gene_id not in ginfo:
                    ginfo[gene_id] = {"gene_name": gene_name, "biotype": biotype}
            else:
                tid = a.get("transcript_id")
                if gene_id is None or tid is None:
                    continue
                exons[tid].append((s, e))
                if tid not in tinfo:
                    tinfo[tid] = {"gene_id": gene_id, "gene_name": gene_name, "biotype": biotype}

    rows = []
    if mode == "gene":
        for gid, ivals in exons.items():
            length = merge_len(ivals)
            rows.append({"gene_id": gid,
                        "gene_name": ginfo[gid]["gene_name"],
                        "biotype": ginfo[gid]["biotype"],
                        "length_bp": length})
    else:
        for tid, ivals in exons.items():
            info = tinfo[tid]
            length = merge_len(ivals)
            rows.append({"transcript_id": tid,
                        "gene_id": info["gene_id"],
                        "gene_name": info["gene_name"],
                        "biotype": info["biotype"],
                        "length_bp": length})
    return rows

def write_csv(rows, out_csv):
    if not rows:
        raise SystemExit("No rows to write; check filters/inputs.")
    keys = rows[0].keys()
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser(description="Compute gene/transcript effective lengths from GTF (BDGP6.* compatible).")
    ap.add_argument("gtf", help="BDGP6.32 GTF path (e.g., Drosophila_melanogaster.BDGP6.32.gtf)")
    ap.add_argument("-o", "--out", default="lengths.csv", help="Output CSV")
    ap.add_argument("--mode", choices=["gene", "transcript"], default="gene", help="Length type")
    ap.add_argument("--protein_coding", action="store_true", help="Only include protein_coding")
    args = ap.parse_args()

    rows = compute_lengths_from_gtf(args.gtf, args.mode, args.protein_coding)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows):,} rows to {args.out}")

if __name__ == "__main__":
    main()

