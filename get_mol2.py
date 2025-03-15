from rdkit import Chem
from rdkit.Chem import AllChem
import os

# Input file name and the folder where molecules will be saved
sdf_file = "Maybridge_HitCreator_V2.sdf"
output_folder = "Maybridge_HitCreator_test"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read the SDF file
supplier = Chem.SDMolSupplier(sdf_file)

# Function to manually generate a .mol2 format file
def mol_to_mol2(mol, mol_name="molecule"):
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    
    # Create the header lines
    mol2_data = f"@<TRIPOS>MOLECULE\n{mol_name}\n"
    mol2_data += f"{mol.GetNumAtoms()} {mol.GetNumBonds()} 0 0 0\nSMALL\nUSER_CHARGES\n\n"
    
    # Add atom information
    mol2_data += "@<TRIPOS>ATOM\n"
    for atom in atoms:
        idx = atom.GetIdx() + 1  # Atom indices must start from 1
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        atom_type = atom.GetSymbol()
        mol2_data += f"{idx} {atom_type}{idx} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f} {atom_type} {1} <0>\n"
    
    # Add bond information
    mol2_data += "@<TRIPOS>BOND\n"
    for i, bond in enumerate(bonds):
        bond_type = bond.GetBondType()
        if bond_type == Chem.rdchem.BondType.SINGLE:
            bond_type = "1"
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            bond_type = "2"
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            bond_type = "3"
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            bond_type = "ar"
        
        start = bond.GetBeginAtomIdx() + 1
        end = bond.GetEndAtomIdx() + 1
        mol2_data += f"{i+1} {start} {end} {bond_type}\n"
    
    return mol2_data

# Process each molecule and save it in .mol2 format
for i, mol in enumerate(supplier):
    if mol is None:
        continue  # Skip invalid molecules

    # Set the molecule name and file name
    product_description = mol.GetProp("Product_Description") if mol.HasProp("Product_Description") else "unknown"
    code = mol.GetProp("Code") if mol.HasProp("Code") else "unknown"
    mol2_filename = os.path.join(output_folder, f"{product_description}_{code}.mol2")

    # Optimize the molecule (optional)
    AllChem.Compute2DCoords(mol)
    
    # Convert to .mol2 format and save
    mol2_data = mol_to_mol2(mol, mol_name=f"{product_description}_{code}")
    with open(mol2_filename, "w") as f:
        f.write(mol2_data)
    
    print(f"{product_description}_{code} saved in .mol2 format: {mol2_filename}")

print("All molecules have been successfully separated and saved in .mol2 format.")
