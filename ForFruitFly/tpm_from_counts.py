#!/usr/bin/env python3
#conda activate gem-py39 (自創獨立分析環境，防止版本衝突引發的bug)
# 基因層級（推薦用於 gene-level TPM）
#python D:\tpm_from_counts.py "D:\dds_combat_sel.csv" --gtf D:\fly_RNAseq\ref_BDGP6_32\Drosophila_melanogaster.BDGP6.32.53.gtf --id-type gene_id -o D:\dds_combat_sel_tpm.tsv  --keep_metadata
# 轉錄本層級（若你做 transcript-level TPM）
#python D:\compute_lengths_and_tpm.py D:\fly_RNAseq\ref_BDGP6_32\Drosophila_melanogaster.BDGP6.32.53.gtf -o D:\fly_RNAseq\ref_BDGP6_32\tx_lengths.csv --mode transcript
#!/usr/bin/env python3
"""
tpm_from_counts.py

Compute TPM from a counts matrix for Drosophila (BDGP6.*) or other genomes.
- Accepts a counts matrix (rows = genes or transcripts, columns = samples; first column = ID)
- Computes effective lengths from a GTF (union of exons) OR uses a precomputed lengths CSV
- Supports IDs as gene_id (FBgn...) or gene_name (e.g., Atxn3) when a GTF is provided (for mapping)
- Optionally restrict to protein_coding entries
- Outputs a TPM matrix (same shape & column order as counts)

Usage examples:
  # 1) Using a BDGP6.32 GTF to compute gene lengths (recommended for gene-level TPM)
  python tpm_from_counts.py counts.tsv --gtf Drosophila_melanogaster.BDGP6.32.gtf --id-type gene_id -o fly_gene_tpm.tsv

  # 2) If your counts use gene_name as the first column, map via the same GTF
  python tpm_from_counts.py counts_by_name.tsv --gtf Drosophila_melanogaster.BDGP6.32.gtf --id-type gene_name -o fly_gene_tpm.tsv

  # 3) Using precomputed lengths
  python tpm_from_counts.py counts.tsv --lengths gene_lengths.csv --id-type gene_id -o fly_gene_tpm.tsv
"""
import argparse
import gzip
import io
import sys
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def _open_text_maybe_gzip(path: str):
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def parse_gtf_attrs(attr_field: str) -> Dict[str, str]:
    """
    Parse the 9th GTF column into a dict. Keys are unescaped; values are stripped of quotes.
    """
    out = {}
    # Split by semicolons; tolerate extra whitespace
    for kv in attr_field.strip().split(";"):
        kv = kv.strip()
        if not kv:
            continue
        if " " not in kv:
            continue
        k, v = kv.split(" ", 1)
        out[k] = v.strip().strip('"')
    return out


def merge_intervals_length(intervals: List[Tuple[int, int]]) -> int:
    """
    Merge overlapping/adjacent [start,end] (1-based, inclusive) intervals and return total length in bp.
    """
    if not intervals:
        return 0
    # sort by start then end
    intervals = sorted(intervals)
    merged = []
    s0, e0 = intervals[0]
    for s, e in intervals[1:]:
        if s <= e0 + 1:
            e0 = e if e > e0 else e0
        else:
            merged.append((s0, e0))
            s0, e0 = s, e
    merged.append((s0, e0))
    return sum(e - s + 1 for s, e in merged)


def compute_lengths_from_gtf(
    gtf_path: str,
    mode: str = "gene",
    protein_coding_only: bool = False,
) -> pd.DataFrame:
    """
    Read a GTF and compute effective lengths:
      - mode == 'gene': union of all exons per gene_id
      - mode == 'transcript': union of all exons per transcript_id

    Returns a DataFrame with columns:
      - for mode='gene': ['gene_id','gene_name','biotype','length_bp']
      - for mode='transcript': ['transcript_id','gene_id','gene_name','biotype','length_bp']
    """
    assert mode in ("gene", "transcript")
    # Store exon intervals
    buckets: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    # Metadata
    g_meta: Dict[str, Dict[str, str]] = {}
    t_meta: Dict[str, Dict[str, str]] = {}

    with _open_text_maybe_gzip(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            _chrom, _source, feature, start, end, _score, _strand, _frame, attrs = parts
            if feature != "exon":
                continue
            try:
                s = int(start); e = int(end)
            except ValueError:
                continue
            a = parse_gtf_attrs(attrs)
            gene_id = a.get("gene_id")
            gene_name = a.get("gene_name", "")
            biotype = a.get("gene_biotype", a.get("gene_type", ""))

            if protein_coding_only and biotype != "protein_coding":
                continue

            if mode == "gene":
                if gene_id is None:
                    continue
                buckets[gene_id].append((s, e))
                if gene_id not in g_meta:
                    g_meta[gene_id] = {"gene_name": gene_name, "biotype": biotype}
            else:
                tid = a.get("transcript_id")
                if gene_id is None or tid is None:
                    continue
                buckets[tid].append((s, e))
                if tid not in t_meta:
                    t_meta[tid] = {"gene_id": gene_id, "gene_name": gene_name, "biotype": biotype}

    rows = []
    if mode == "gene":
        for gid, ivals in buckets.items():
            rows.append({
                "gene_id": gid,
                "gene_name": g_meta.get(gid, {}).get("gene_name", ""),
                "biotype": g_meta.get(gid, {}).get("biotype", ""),
                "length_bp": merge_intervals_length(ivals)
            })
        return pd.DataFrame(rows, columns=["gene_id","gene_name","biotype","length_bp"])
    else:
        for tid, ivals in buckets.items():
            info = t_meta.get(tid, {})
            rows.append({
                "transcript_id": tid,
                "gene_id": info.get("gene_id", ""),
                "gene_name": info.get("gene_name", ""),
                "biotype": info.get("biotype", ""),
                "length_bp": merge_intervals_length(ivals)
            })
        return pd.DataFrame(rows, columns=["transcript_id","gene_id","gene_name","biotype","length_bp"])


def infer_sep(path: str, user_sep: Optional[str]) -> str:
    """
    Determine the separator for counts file: tab or comma.
    """
    if user_sep in ("tab", "csv"):
        return "\t" if user_sep == "tab" else ","
    p = str(path).lower()
    if p.endswith(".tsv") or p.endswith(".txt"):
        return "\t"
    return ","  # default to csv


def load_counts(path: str, sep_choice: Optional[str]) -> pd.DataFrame:
    sep = infer_sep(path, sep_choice)
    df = pd.read_csv(path, sep=sep, header=0)
    if df.shape[1] < 2:
        raise SystemExit("Counts file must have at least 2 columns: ID + ≥1 sample columns.")
    return df


def validate_counts(counts: pd.DataFrame) -> None:
    # First col is ID
    if counts.iloc[:, 1:].isnull().any().any():
        print("Warning: NaNs in counts matrix (non-ID columns). They will be treated as zeros.", file=sys.stderr)


def normalize_tpm(counts: pd.DataFrame, lengths_bp: pd.Series, min_length_bp: int = 1) -> pd.DataFrame:
    """
    Compute TPM given raw counts and a length series aligned to counts.index (in bp).
    """
    # Convert lengths to kb; avoid zero
    lengths_kb = lengths_bp.clip(lower=min_length_bp) / 1000.0

    # Replace NaN counts with 0
    mat = counts.fillna(0.0).astype(float).to_numpy()  # (n_genes, n_samples)

    # Compute RPK
    rpk = mat / lengths_kb.to_numpy()[:, None]

    # Per-sample scaling to 1e6
    scaling = rpk.sum(axis=0)
    # To avoid /0, set zero sums to 1 (will produce zeros in TPM for that column)
    scaling[scaling == 0] = 1.0
    tpm = (rpk / scaling[None, :]) * 1e6

    out = pd.DataFrame(tpm, index=counts.index, columns=counts.columns)
    return out


def main():
    ap = argparse.ArgumentParser(description="Convert a counts matrix to TPM using gene/transcript lengths from GTF or a precomputed CSV.")
    ap.add_argument("counts", help="Counts matrix file (TSV/CSV). First column = ID (gene_id/gene_name/transcript_id).")
    ap.add_argument("--gtf", help="Path to GTF (e.g., BDGP6.32). Used to compute lengths and/or map gene_name->gene_id.")
    ap.add_argument("--lengths", help="Precomputed lengths CSV with columns including one of: gene_id,length_bp OR transcript_id,length_bp.")
    ap.add_argument("--mode", choices=["gene","transcript"], default="gene", help="Length type to compute/use (default: gene).")
    ap.add_argument("--id-type", choices=["gene_id","gene_name","transcript_id"], default="gene_id",
                    help="Type of IDs found in the first column of the counts file.")
    ap.add_argument("--protein_coding", action="store_true", help="Restrict to protein_coding (requires --gtf or lengths file containing biotype).")
    ap.add_argument("--sep", choices=["tab","csv"], help="Separator of counts file (auto by extension if omitted).")
    ap.add_argument("--min_length", type=int, default=1, help="Minimum effective length in bp to avoid divide-by-zero (default: 1).")
    ap.add_argument("-o","--out", default="tpm.tsv", help="Output TPM matrix path (TSV).")
    ap.add_argument("--write_lengths", help="Optional path to write the (filtered) lengths used here as a TSV.")
    ap.add_argument("--keep_metadata", action="store_true", help="If available, add gene_name/biotype columns to output (as leftmost columns).")
    args = ap.parse_args()

    # Load counts
    counts_df = load_counts(args.counts, args.sep)
    id_col = counts_df.columns[0]
    counts_df = counts_df.rename(columns={id_col: "ID"})
    # Separate ID column
    ids = counts_df["ID"].astype(str)
    expr = counts_df.drop(columns=["ID"])

    # Load/compute lengths
    lengths_df: Optional[pd.DataFrame] = None
    if args.lengths:
        lengths_df = pd.read_csv(args.lengths)
    elif args.gtf:
        lengths_df = compute_lengths_from_gtf(args.gtf, mode=args.mode, protein_coding_only=False)
    else:
        raise SystemExit("Please provide either --gtf or --lengths.")

    # Determine join key according to mode & id-type
    if args.mode == "gene":
        # Expect gene_id OR gene_name in counts
        if "gene_id" not in lengths_df.columns:
            raise SystemExit("Lengths table missing 'gene_id' column for mode=gene.")
        join_key = "gene_id" if args.id_type == "gene_id" else "gene_name"
        if args.id_type == "gene_name" and "gene_name" not in lengths_df.columns:
            raise SystemExit("Lengths table missing 'gene_name' needed to map counts by gene_name.")
        # Build a small lengths table with desired join key and metadata
        cols = ["gene_id","gene_name","biotype","length_bp"]
        present = [c for c in cols if c in lengths_df.columns]
        L = lengths_df[present].copy()
        # If user wants protein_coding only, filter here (requires biotype column)
        if args.protein_coding:
            if "biotype" not in L.columns:
                print("Warning: --protein_coding requested but 'biotype' column not found in lengths; skipping filter.", file=sys.stderr)
            else:
                L = L[L["biotype"] == "protein_coding"]
        # Merge lengths onto counts IDs
        if join_key not in L.columns:
            raise SystemExit(f"Join key '{join_key}' not found in lengths. Available: {L.columns.tolist()}")
        L = L.drop_duplicates(subset=[join_key])
        merge_df = pd.DataFrame({"ID": ids})
        merge_df = merge_df.merge(L, left_on="ID", right_on=join_key, how="left")
        if merge_df["length_bp"].isna().any():
            n_miss = int(merge_df["length_bp"].isna().sum())
            print(f"Warning: {n_miss} rows in counts had no matching length; they will be dropped.", file=sys.stderr)
        # Build the aligned series & optional metadata
        keep_mask = merge_df["length_bp"].notna()
        aligned_lengths = merge_df.loc[keep_mask, "length_bp"].astype(float)
        meta_cols = []
        if args.keep_metadata:
            for c in ("gene_id","gene_name","biotype"):
                if c in merge_df.columns:
                    meta_cols.append(c)
        meta_df = merge_df.loc[keep_mask, meta_cols] if meta_cols else None
        # Subset expression accordingly
        expr = expr.loc[keep_mask.values, :]
        ids_final = merge_df.loc[keep_mask, "ID"].astype(str)

    else:
        # mode == transcript
        if "transcript_id" not in lengths_df.columns:
            raise SystemExit("Lengths table missing 'transcript_id' column for mode=transcript.")
        join_key = "transcript_id" if args.id_type == "transcript_id" else None
        if args.id_type != "transcript_id":
            raise SystemExit("For mode=transcript, counts first column must be transcript_id (set --id-type transcript_id).")
        cols = ["transcript_id","gene_id","gene_name","biotype","length_bp"]
        present = [c for c in cols if c in lengths_df.columns]
        L = lengths_df[present].copy()
        if args.protein_coding:
            if "biotype" not in L.columns:
                print("Warning: --protein_coding requested but 'biotype' column not found in lengths; skipping filter.", file=sys.stderr)
            else:
                L = L[L["biotype"] == "protein_coding"]
        L = L.drop_duplicates(subset=["transcript_id"])
        merge_df = pd.DataFrame({"ID": ids}).merge(L, left_on="ID", right_on="transcript_id", how="left")
        if merge_df["length_bp"].isna().any():
            n_miss = int(merge_df["length_bp"].isna().sum())
            print(f"Warning: {n_miss} rows in counts had no matching transcript length; they will be dropped.", file=sys.stderr)
        keep_mask = merge_df["length_bp"].notna()
        aligned_lengths = merge_df.loc[keep_mask, "length_bp"].astype(float)
        meta_cols = []
        if args.keep_metadata:
            for c in ("transcript_id","gene_id","gene_name","biotype"):
                if c in merge_df.columns:
                    meta_cols.append(c)
        meta_df = merge_df.loc[keep_mask, meta_cols] if meta_cols else None
        expr = expr.loc[keep_mask.values, :]
        ids_final = merge_df.loc[keep_mask, "ID"].astype(str)

    # Compute TPM
    tpm = normalize_tpm(expr, aligned_lengths, min_length_bp=args.min_length)
    # Prepare output
    out_df = tpm.copy()
    out_df.insert(0, "ID", ids_final.values)

    # If metadata requested, prepend (deduplicated) columns
    if args.keep_metadata and meta_df is not None and not meta_df.empty:
        # Ensure the order matches ids_final
        meta_df = meta_df.reset_index(drop=True)
        out_df = pd.concat([meta_df, out_df], axis=1)

    # Write TPM as TSV (explicitly use tab to avoid precision issues in Excel)
    out_path = args.out
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"[✓] Wrote TPM matrix to: {out_path}  (n={out_df.shape[0]} features, m={out_df.shape[1]-1} samples)")

    # Optionally write the lengths used (aligned to output order)
    if args.write_lengths:
        # Put ID + length_bp (+ optional metadata) for transparency
        to_write = pd.DataFrame({"ID": ids_final.values, "length_bp": aligned_lengths.values})
        if args.keep_metadata and meta_df is not None and not meta_df.empty:
            # Avoid duplicate columns
            meta_add = meta_df.copy()
            for c in ("ID","length_bp"):
                if c in meta_add.columns:
                    meta_add = meta_add.drop(columns=[c])
            to_write = pd.concat([meta_add, to_write], axis=1)
        to_write.to_csv(args.write_lengths, sep="\t", index=False)
        print(f"[✓] Wrote lengths used to: {args.write_lengths}")


if __name__ == "__main__":
    main()
