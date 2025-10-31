#!/usr/bin/env bash
# ── 保證 RSeQC 指令在 $PATH ───────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rseqc_qc
# 此script在Ubuntu的conda下運行，RSeQC不支持conda in windows運作。
# =========================================================================
#  run_rseqc.sh — Batch QC for RNA-seq BAM files with RSeQC + MultiQC
# -------------------------------------------------------------------------
#  USAGE
#    bash run_rseqc.sh <BAM_DIR> <ANNOTATION.{gtf|bed}> [THREADS]
#  EXAMPLE (WSL/Linux)
#    # activate your conda env first (see README below)
#    bash run_rseqc.sh /mnt/e/cch_data/mice_RNAseq_mix_data/BAM \
#                     /home/domingo/Mus_musculus.GRCm39.109.gtf 16
# -------------------------------------------------------------------------
#  Dependencies (install once via conda – see README section):
#    samtools  rseqc  bedops  multiqc  ( + their run-time deps )
# -------------------------------------------------------------------------
#  Output
#    QC_results/
#      ├─ <sampleID>/               # per-sample reports / plots
#      └─ multiqc/                  # aggregated HTML report
# -------------------------------------------------------------------------
#  建立執行環境:
#  conda create -n rseqc_qc -c bioconda -c conda-forge \
#     python=3.10 rseqc samtools bedops multiqc
#     conda activate rseqc_qc
#  轉換生成BED12參考基因組
#  cd /home/domingo
#  gffread Mus_musculus.GRCm39.109.gtf -B -o Mus_musculus.GRCm39.109.bed
#  執行腳本:
# 進入腳本所在資料夾
#  cd /mnt/e/cch_data/mice_RNAseq_mix_data/RSeQC
#  chmod +x run_rseqc.sh          # 第一次需要加可執行權限
#  sed -i 's/\r$//' run_rseqc.sh  
#  bash run_rseqc.sh /mnt/e/cch_data/mice_RNAseq_mix_data/BAM /home/domingo/mm39.bed 32
# -----------------------------------------------------------------------------
#  Output tree
#    QC_results/
#      ├─ <sampleID>/               # per‑sample RSeQC reports / plots
#      └─ multiqc/                  # aggregated HTML report
# =========================================================================
set -euo pipefail

# ── 0. Load conda env & make sure conda tools > system tools ────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rseqc_qc
export PATH="$CONDA_PREFIX/bin:$PATH"

# ── 1. Parse arguments ──────────────────────────────────────────────────────
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <BAM_DIR> <GTF|BED> [THREADS]" >&2
  exit 1
fi

BAM_DIR=$(realpath "$1")
ANN_PATH=$(realpath "$2")
THREADS=${3:-8}

QC_ROOT="QC_results"; mkdir -p "$QC_ROOT"

# ── 2. Make / fetch BED12 annotation ─────────────────────────────────────────
case "$ANN_PATH" in
  *.bed)   BED12="$ANN_PATH" ;;
  *.gtf)
    BED12="$QC_ROOT/$(basename "${ANN_PATH}" .gtf).bed"
    if [[ ! -s "$BED12" ]]; then
      echo "[INFO] Converting GTF → BED12 via gffread …"
      gffread "$ANN_PATH" -B -o - | grep -v '^#' > "$BED12"
    fi ;;
  *) echo "[ERROR] Annotation must be .gtf or .bed" >&2; exit 1 ;;
esac

# ── 3. Collect BAMs (case‑insensitive) ───────────────────────────────────────
shopt -s nullglob nocaseglob
BAM_LIST=("${BAM_DIR}"/*.bam "${BAM_DIR}"/*.BAM)
if [[ ${#BAM_LIST[@]} -eq 0 ]]; then
  echo "[ERROR] No BAM files found in $BAM_DIR" >&2; exit 1
fi

# ── 4. Iterate over BAM files ───────────────────────────────────────────────-
for BAM in "${BAM_LIST[@]}"; do
  SID=$(basename "${BAM%.*}")          # strip .bam/.BAM
  OUTDIR="$QC_ROOT/$SID"; mkdir -p "$OUTDIR"
  echo -e "\n==> Processing $SID"

  # 4‑A. ensure BAM index exists
  [[ -f "${BAM}.bai" ]] || samtools index "$BAM"

  # 4‑B. RSeQC modules (no multi‑thread flags except RPKM_saturation)
  infer_experiment.py  -i "$BAM" -r "$BED12"                          > "$OUTDIR/${SID}.strandness.txt"
  read_distribution.py -i "$BAM" -r "$BED12"                          > "$OUTDIR/${SID}.read_distribution.txt"
  read_duplication.py  -i "$BAM"                                       > "$OUTDIR/${SID}.duplication.txt"
  geneBody_coverage.py -i "$BAM" -r "$BED12" -o "$OUTDIR/${SID}.geneBody"
  junction_saturation.py -i "$BAM" -r "$BED12" -o "$OUTDIR/${SID}.junction_sat"
  RPKM_saturation.py    -i "$BAM" -r "$BED12" --thread "$THREADS" -o "$OUTDIR/${SID}.rpkm_sat"

done

# ── 5. Aggregate with MultiQC ───────────────────────────────────────────────
if command -v multiqc &> /dev/null; then
  echo -e "\n[INFO] Generating MultiQC report …"
  multiqc "$QC_ROOT" -o "$QC_ROOT/multiqc"
else
  echo "[WARN] multiqc not found – skipping aggregate report" >&2
fi

echo -e "\n[Done] All QC reports saved in: $(realpath "$QC_ROOT")"