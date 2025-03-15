import pubchempy as pcp
import requests
import os
import sys
import multiprocessing
from rdkit.Chem import SDMolSupplier

def download_sdf(drug_name, output_folder="sdf_files"):
    """
    Downloads the SDF file for a given drug name from PubChem.
    """
    try:
        compounds = pcp.get_compounds(drug_name, 'name')
        if not compounds:
            print(f"❌ No compound found for {drug_name}.")
            return None

        cid = compounds[0].cid
        print(f"✅ Found CID: {cid} for {drug_name}.")

        # Construct the URL for downloading the SDF file
        sdf_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/{cid}/SDF?record_type=3d"
        sdf_data = requests.get(sdf_url).text

        # Create the output folder if it does not exist
        os.makedirs(output_folder, exist_ok=True)

        # Define the output file path
        output_file = os.path.join(output_folder, f"{drug_name.replace(' ', '_')}.sdf")

        # Save the SDF file
        with open(output_file, "w") as f:
            f.write(sdf_data)

        print(f"💾 SDF saved for {drug_name}: {output_file}")

        # Load the molecule using RDKit
        suppl = SDMolSupplier(output_file)
        mols = [mol for mol in suppl if mol is not None]
        return drug_name, mols

    except Exception as e:
        print(f"⚠️ Error processing {drug_name}: {e}")
        return None

def process_drug_list(file_path, num_workers=None):
    """
    Reads a TXT file with drug names and downloads SDF files in parallel.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found.")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as file:
        drug_names = [line.strip() for line in file.readlines() if line.strip()]
    
    if not drug_names:
        print("⚠️ Warning: The file is empty!")
        sys.exit(1)

    # Set the number of workers (default: all CPU cores)
    num_workers = num_workers or multiprocessing.cpu_count()
    print(f"🚀 Using {num_workers} parallel workers...")

    all_molecules = {}
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(download_sdf, drug_names)
    
    for result in results:
        if result:
            drug_name, mols = result
            all_molecules[drug_name] = mols

    print(f"📦 Total {len(all_molecules)} drugs processed.")
    return all_molecules

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Error: Please provide the TXT file containing drug names!")
        print("📌 Usage: python drug_fetcher.py drugs.txt [num_workers]")
        sys.exit(1)

    file_path = sys.argv[1]
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else None

    process_drug_list(file_path, num_workers)