import os
import argparse
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from Bio.PDB import PDBList, MMCIFParser, PDBParser
import glob
from prody import parseMMCIF, writePDB
import warnings
warnings.filterwarnings("ignore")

def download_single_pdb(row, out_path):
    """Downloads a single PDB file."""
    pdb_value = row['PDB']
    if isinstance(pdb_value, str):
        pdb_list = pdb_value.split(';')
        pdbl = PDBList()
        gene = row['Entry'].split(';')[0]
        gene_dir = os.path.join(out_path, gene)
        os.makedirs(gene_dir, exist_ok=True)
        
        for pdb in pdb_list:
            pdbl.retrieve_pdb_file(pdb, pdir=gene_dir, file_format='mmCif')
            pdb_file = os.path.join(gene_dir, pdb + ".cif")
            if os.path.exists(pdb_file):
                parser = MMCIFParser()
                try:
                    data = parser.get_structure(pdb, pdb_file)
                    return {
                        "UniProt ID": row['Entry'],
                        "Gene Name": row['Entry Name'],
                        "PDB": pdb,
                        "Resolution": data.header.get("resolution"),
                        "Has missing residues": data.header.get("has_missing_residues"),
                        "Structure method": data.header.get("structure_method"),
                    }
                except Exception as e:
                    print(f"Error parsing {pdb_file}: {e}")
    return None

def download_pdb(df_path, out_path):
    """Parallelized function to download PDB files."""
    df = pd.read_csv(df_path)
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(download_single_pdb, [(row, out_path) for _, row in df.iterrows()])
    
    results = [res for res in results if res is not None]
    merged_df = pd.DataFrame(results).drop_duplicates().fillna('-')
    merged_df.to_csv(os.path.join(out_path, 'Protein_info.csv'), index=False)
    return merged_df

def download_alphafold(entry, out_path):
    """Downloads AlphaFold structure for a single UniProt ID."""
    alphafold_ID = f'AF-{entry}-F1'
    database_version = 'v2'
    model_url = f'https://alphafold.ebi.ac.uk/files/{alphafold_ID}-model_{database_version}.pdb'
    error_url = f'https://alphafold.ebi.ac.uk/files/{alphafold_ID}-predicted_aligned_error_{database_version}.json'
    
    gene_folder = os.path.join(out_path, entry)
    os.makedirs(gene_folder, exist_ok=True)
    model_path = os.path.join(gene_folder, f'{alphafold_ID}.pdb')
    error_path = os.path.join(gene_folder, f'{alphafold_ID}.json')
    
    os.system(f'curl -s {model_url} -o {model_path}')
    os.system(f'curl -s {error_url} -o {error_path}')
    
    return {
        "UniProt ID": entry,
        "PDB": alphafold_ID,
        "AlphaFold": True,
        "Resolution": '-',
        "Has missing residues": '-',
        "Structure method": 'AlphaFold'
    }

def get_alphafold(df_path, merged_df, out_path):
    """Parallelized function to download AlphaFold structures."""
    df = pd.read_csv(df_path)
    entries = df['Entry'].unique()
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(download_alphafold, [(entry, out_path) for entry in entries])
    
    results_df = pd.DataFrame(results).drop_duplicates().fillna('-')
    combined_df = pd.concat([merged_df, results_df], ignore_index=True)
    combined_df.to_csv(os.path.join(out_path, 'Protein_info.csv'), index=False)
    return combined_df

def extract_required_chains(final_df, out_path):
    ent_files = glob.glob(out_path+'/**/*.cif', recursive=True)
    
    for ent_file in ent_files:
        try:
            pdb_code = os.path.splitext(os.path.basename(ent_file))[0]
            syst = parseMMCIF(ent_file)
            final_df_pdb = final_df[final_df['PDB'] == pdb_code.upper()]
            
            if not final_df_pdb.empty:
                chain = final_df_pdb['Chain ID'].values[0]
                uniprot_id = final_df_pdb['UniProt ID'].values[0]
                syst_chain = syst.select(f'chain {chain}')
                writePDB(os.path.join(out_path, uniprot_id, f'{pdb_code}_{chain}.pdb'), syst_chain)
        except Exception as e:
            print(f"Error processing {ent_file}: {e}")
            os.remove(ent_file)
            with open('deleted_files.txt', 'a') as f:
                f.write(os.path.basename(ent_file) + '\n')
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Query PDB and AF2 structure database for proteins')
    parser.add_argument('-df_uniprot', type=str, help='Path to CSV containing gene names')
    parser.add_argument('-out_path', type=str, help='Path to output folder')
    args = parser.parse_args()
    
    df_uniprot = Path(args.df_uniprot).resolve()
    if not df_uniprot.is_file():
        raise ValueError(f"Invalid file: {df_uniprot}")
    out_path = args.out_path
    
    merged_df = download_pdb(df_uniprot, out_path)
    print('PDB structures downloaded.')
    
    combined_df = get_alphafold(df_uniprot, merged_df, out_path)
    print('AlphaFold structures downloaded.')
    
    extract_required_chains(combined_df, out_path)
    print('Extracted required chains.')
    print('Done.')
