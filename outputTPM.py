import pandas as pd

def extract_gene_lengths(gtf_file, output_file="gene_lengths.csv"):
    gene_lengths = {}
    with open(gtf_file, "r") as gtf:
        for line in gtf:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if fields[2] == "transcript":
                attributes = fields[8]
                gene_id = [x.split('"')[1] for x in attributes.split(";") if "gene_id" in x][0]
                length = int(fields[4]) - int(fields[3]) + 1
                if gene_id not in gene_lengths:
                    gene_lengths[gene_id] = length
    pd.DataFrame.from_dict(gene_lengths, orient="index", columns=["Length"]).to_csv(output_file)

# 生成基因長度文件
extract_gene_lengths("merged.gtf", "gene_lengths.csv")
