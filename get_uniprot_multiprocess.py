import argparse
import pandas as pd
import numpy as np
import os
import time
import math
import re
from pathlib import Path
from bioservices import UniProt
from multiprocessing import Pool, cpu_count

def fetch_uniprot_data(gene_block):
    """
    Fetches UniProt data for a block of gene names.
    """
    service = UniProt()
    query = " OR ".join(gene_block)
    
    for attempt in range(5):  # Retry mechanism for robustness
        try:
            df = service.get_df(query, organism="Homo sapiens")
            df = df[df['Gene Names (primary)'].isin(gene_block)]
            return df
        except Exception as e:
            print(f"⚠️ Error fetching UniProt data (Attempt {attempt+1}/5): {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    return pd.DataFrame()

def get_uniprot_data_parallel(gene_path, out_csv, num_workers=None):
    """
    Fetches UniProt data in parallel using multiprocessing.
    """
    with open(gene_path, "r") as f:
        gene_names = [line.strip() for line in f if line.strip()]
    
    if not gene_names:
        raise ValueError("No gene names found!")
    
    num_workers = num_workers or cpu_count()  # Use all available cores if not specified
    n_blocks = math.ceil(len(gene_names) / 5)
    gene_blocks = [gene_names[i*5:(i+1)*5] for i in range(n_blocks)]
    
    print(f"🔍 Querying UniProt with {num_workers} workers...")
    
    with Pool(num_workers) as pool:
        results = pool.map(fetch_uniprot_data, gene_blocks)
    
    df = pd.concat(results, ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ UniProt data saved to {out_csv}")
    return df

def check_genes(gene_path, in_csv):
    """
    Checks which genes are found in the output CSV and logs missing ones.
    """
    with open(gene_path, 'r') as file:
        words = [line.strip() for line in file]
    
    with open(in_csv, 'r') as file:
        contents = file.read()
    
    matches = [word for word in words if re.search(rf'\b{word}\b', contents)]
    not_found = [word for word in words if word not in matches]
    
    output_path = os.path.splitext(in_csv)[0] + '_not_found.txt'
    with open(output_path, 'w') as file:
        file.write('\n'.join(not_found))
    
    print(f"✅ {len(matches)} genes found, {len(not_found)} not found. Check: {output_path}")
    return matches, not_found

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel UniProt Gene Query')
    parser.add_argument('-gene_file', type=str, required=True, help='Path to gene list file')
    parser.add_argument('-out_csv', type=str, required=True, help='Output CSV file')
    parser.add_argument('-workers', type=int, default=None, help='Number of worker processes (default: all CPUs)')
    args = parser.parse_args()
    
    df = get_uniprot_data_parallel(args.gene_file, args.out_csv, args.workers)
    check_genes(args.gene_file, args.out_csv)
