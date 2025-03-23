from pathlib import Path
import subprocess, os, glob, multiprocessing, argparse, logging, warnings, uuid, sys, shutil
from datetime import datetime
import numpy as np
import pandas as pd
from logging.handlers import RotatingFileHandler

#import nglview as nv
from openbabel import pybel

from opencadd.structure.core import Structure
from opencadd.io.dataframe import DataFrame

from rdkit import Chem
from rdkit.Chem import AllChem

# filter warnings
warnings.filterwarnings("ignore")
ob_log_handler = pybel.ob.OBMessageHandler()
pybel.ob.obErrorLog.SetOutputLevel(0)

def setup_logging(log_file):
    """
    Configure and return a logger that writes to both console and file.
    Implements rotating file handler and timestamped log file names.
    """
    logger = logging.getLogger("consensus_docker")
    logger.setLevel(logging.DEBUG)

    # Configure file handler with rotating capabilities
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)

    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Format handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def pdb_to_pdbqt(pdb_path, pdbqt_path, logger, pH=7.4):
    """
    Convert a PDB file to a PDBQT file needed by docking programs of the AutoDock family.

    Parameters
    ----------
    pdb_path: str or pathlib.Path
        Path to input PDB file.
    pdbqt_path: str or pathlib.path
        Path to output PDBQT file.
    pH: float
        Protonation at given pH.
    """
    logger.info('Converting receptor pdb to pdbqt format...')
    molecule = list(pybel.readfile("pdb", str(pdb_path)))[0]
    # add hydrogens at given pH
    #molecule.OBMol.CorrectForPH(pH)
    #molecule.addh()
    # add partial charges to each atom
    for atom in molecule.atoms:
        atom.OBAtom.GetPartialCharge()
    molecule.write("pdbqt", str(pdbqt_path), overwrite=True)
    logger.info('Converting receptor pdb to pdbqt format... Done.')
    return

def lepro(args, logger):
    logger.info('Preparing input pdb for LeDock by lepro exe...')
    subprocess.run([args.lepro_path, args.receptor_pdb])
    logger.info('Preparing input pdb for LeDock by lepro exe... Done.')

    # Move pro.pdb found in working directory to os.path.join(args.outfolder, 'ledock')
    pro_path = os.path.join(os.getcwd(), 'pro.pdb')
    shutil.move(pro_path, args.outfolder_ledock)

    # Return final path of pro.pdb
    return os.path.join(args.outfolder_ledock, 'pro.pdb')

def sdf_to_mol2(sdf_file, mol_name, mol2_filepath):

    # Read the SDF file
    supplier = Chem.SDMolSupplier(sdf_file)

    mol = supplier[0]

    # Optimize the molecule (optional)
    AllChem.Compute2DCoords(mol)
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

    with open(mol2_filepath, "w") as f:
        f.write(mol2_data)

def get_pocket_coords(args, logger):
    logger.info('Getting pocket coordinates...')
    structure_df = DataFrame.from_file(args.pocket_pdb)
    positions = np.array([structure_df["atom.x"].values,structure_df["atom.y"].values,structure_df["atom.z"].values])
    min_coords = np.min(positions,axis=1)
    max_coords = np.max(positions,axis=1)
    pocket_center = list(map(str,(max_coords + min_coords) / 2))
    pocket_size = list(map(str,(max_coords - min_coords) + 5))
    #logger.info(f'Pocket center: {pocket_center}, Pocket size: {pocket_size}')
    logger.info('Getting pocket coordinates... Done.')
    return pocket_center, pocket_size, min_coords, max_coords

def run_smina(args, logger):
    logger.info('Running smina...')
    subprocess.call(f"{args.smina_path} -r {args.smina_pdbqt} -l {args.ligand_sdf} \
    --center_x {args.pocket_center[0]} --center_y {args.pocket_center[1]} --center_z {args.pocket_center[2]} \
    --size_x {args.pocket_size[0]} --size_y {args.pocket_size[1]} --size_z {args.pocket_size[2]} --out {os.path.join(args.outfolder_smina, 'out.sdf')} \
    --num_modes {args.num_modes} --exhaustiveness {args.exhaustiveness} --cpu {args.num_threads} --log {os.path.join(args.outfolder_smina, 'out.sdf')}",shell=True)
    logger.info('Running smina... Done.')

    split_mol(args, logger, tool="smina")

def split_mol(args, logger, tool="smina"):
    logger.info('Splitting docked poses into individual files...')
    if tool == "smina":
        docked_poses_path = Path(os.path.join(args.outfolder_smina, 'out.sdf'))
        molecules = pybel.readfile("sdf", str(docked_poses_path))
        for i, molecule in enumerate(molecules, 1):
            molecule.write("sdf", os.path.join(args.outfolder_smina, f"out_{i}.sdf"), overwrite=True)

    elif tool == "ledock":
        docked_poses_path = os.path.join(args.outfolder_ledock, 'out.dok')
        
        with open(docked_poses_path, 'r') as infile:
            file_counter = 1
            current_chunk = []
            for line in infile:
                if line.startswith('REMARK Docking time:'):
                    continue #Skip this line.
                elif line.startswith("REMARK Cluster"):
                    # Start a new chunk
                    if current_chunk:  # If we have content in the current chunk, write it
                        output_filepath = os.path.join(args.outfolder_ledock, f"out_{file_counter}.dok")
                        with open(output_filepath, 'w') as outfile:
                            outfile.writelines(current_chunk)
                        file_counter += 1
                    current_chunk = [line]  # Start a new chunk with the "REMARK Cluster" line
                else:
                    current_chunk.append(line)

            # Write the last chunk (if any)
            if current_chunk:
                output_filepath = os.path.join(args.outfolder_ledock, f"out_{file_counter}.dok")
                with open(output_filepath, 'w') as outfile:
                    outfile.writelines(current_chunk)

    logger.info('Splitting docked poses into individual files... Done.')
    
def run_ledock(args, logger):
    logger.info('Running LeDock...')
    
    # Get stem from file name of args.ligand_sdf
    sdf_path = Path(args.ligand_sdf)
    sdf_stem = sdf_path.stem
    args.ligand_mol2 = os.path.join(args.outfolder_ledock, sdf_stem + '.mol2')
    sdf_to_mol2(args.ligand_sdf, 'LIG', args.ligand_mol2)

    with open(os.path.join(args.outfolder_ledock,'ligand.txt'), "w") as f:
        f.write(args.ligand_mol2)

    dock_in = f"""
    Receptor
    {args.lepro_pdb}

    RMSD
    1.0

    Binding pocket
    {args.min_coords[0]} {args.max_coords[0]} 
    {args.min_coords[1]} {args.max_coords[1]}
    {args.min_coords[2]} {args.max_coords[2]}

    Number of binding poses
    20

    Ligands list
    {os.path.join(args.outfolder_ledock, 'ligand.txt')}

    END
    """

    with open(os.path.join(args.outfolder_ledock, 'dock.in'), 'w') as dock_in_f:
        dock_in_f.write(dock_in.strip() + "\n")

    subprocess.call(f"{args.ledock_path} {os.path.join(args.outfolder_ledock, 'dock.in')}", shell=True)

    # Find the file with ".dok" extension in ledock output folder.
    dock_files = glob.glob(os.path.join(args.outfolder_ledock, '*.dok'))
    dock_file = dock_files[0]

    # Rename that file to out.dok
    if dock_file is not None:
        shutil.move(dock_file, os.path.join(args.outfolder_ledock, 'out.dok'))

    logger.info('Running LeDock... Done.')

    split_mol(args, logger, tool="ledock")

def consensus_dock(args, logger):

    # Convert receptor pdb to pdbqt format
    args.smina_pdbqt = os.path.join(args.outfolder_smina, 'receptor.pdbqt')
    pdb_to_pdbqt(args.receptor_pdb, args.smina_pdbqt, logger, pH=args.pH)

    # Get pocket coordinates
    pocket_center, pocket_size, min_coords, max_coords = get_pocket_coords(args, logger)

    # Add pocket coordinates and ligands chunks to args
    args.pocket_center = pocket_center
    args.pocket_size = pocket_size
    args.min_coords = min_coords
    args.max_coords = max_coords

    # Run smina docking
    run_smina(args, logger)

    # Run LeDock docking
    args.lepro_pdb = lepro(args, logger)
    run_ledock(args, logger)

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="CLI tool to perform consensus docking simulations using Smina and LeDock."
    )

    parser.add_argument('--outfolder', type=str, help='Base output directory (default: current directory)')
    parser.add_argument('--smina_path', type=str, default='smina', help='Path to Smina executable (default: smina)')
    parser.add_argument('--ledock_path', type=str, default='ledock', help='Path to LeDock executable (default: ledock)')
    parser.add_argument('--lepro_path', type=str, default='lepro', help='Path to lepro executable (default: lepro)')
    parser.add_argument('--pH', type=float, default=7.4, help='pH value for adding missing hydrogens (default: 7.4)')
    parser.add_argument('--receptor_pdb', type=str, help='Path to receptor PDB file')
    parser.add_argument('--ligand_sdf', type=str, help='Path to ligand SDF file')
    parser.add_argument('--pocket_pdb', type=str, help='Path to pocket PDB file')
    parser.add_argument('--exhaustiveness', type=int, default=12, help='Exhaustiveness value for Smina (default: 12)')
    parser.add_argument('--num_modes', type=int, default=20, help='Number of modes for Smina (default: 20)')
    parser.add_argument('--num_threads', type=int, default=1, help='Number of threads for Smina (default: 1)')

    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.outfolder, exist_ok=False)

    # Create output directory for smina within outfolder
    os.makedirs(os.path.join(args.outfolder, 'smina'), exist_ok=False)
    args.outfolder_smina = os.path.join(args.outfolder, 'smina')

    # Create output directory for ledock within outfolder
    os.makedirs(os.path.join(args.outfolder, 'ledock'), exist_ok=False)
    args.outfolder_ledock = os.path.join(args.outfolder, 'ledock')

    # Timestamp for job ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
    job_id = f"{timestamp}_{str(uuid.uuid4())[:4]}"

    logger = setup_logging(os.path.join(args.outfolder,f"log_{job_id}.log"))
    logger.info('########## Starting consensus_docker.py #########')
    logger.info('consensus_docker.py was called with the following arguments: ')
    logger.info(' '.join(sys.argv))
    logger.info('########## Starting consensus_docker.py #########')

    # Execute commands
    consensus_dock(args, logger)

if __name__ == "__main__":
    main()