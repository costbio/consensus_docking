from pathlib import Path
import subprocess, os, glob, multiprocessing, argparse, logging, warnings, uuid, sys, shutil
from datetime import datetime
import numpy as np
import pandas as pd
from logging.handlers import RotatingFileHandler

#import nglview as nv
from openbabel import pybel, openbabel

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

def make_complex(args, logger, tool="smina"):
    logger.info('Making complex...')

    if tool == "smina":
        receptor = Chem.MolFromPDBFile(args.receptor_pdb, removeHs=False, sanitize=False)
        if receptor is None:
            raise RuntimeError(f"Failed to load receptor from PDB: {args.receptor_pdb}")

        # Find out how many ligands we have
        ligand_sdf = os.path.join(args.outfolder_smina, 'out.sdf')
        ligand_supplier = Chem.SDMolSupplier(ligand_sdf, removeHs=False)
        ligands = [m for m in ligand_supplier if m is not None]
        if not ligands:
            raise RuntimeError(f"No valid molecules found in SDF file: {ligand_file}")
        
        num_ligands = len(list(ligands))

        for i in range(1, num_ligands+1):
            ligand = ligands[i-1]
            
            docked_complex = Chem.CombineMols(receptor, ligand)
            Chem.MolToPDBFile(docked_complex, os.path.join(args.outfolder_smina, f"complex_{i}.pdb"))

    elif tool == "ledock":
        receptor = Chem.MolFromPDBFile(args.lepro_pdb, removeHs=False, sanitize=False)
        if receptor is None:
            raise RuntimeError(f"Failed to load receptor from PDB: {args.lepro_pdb}")

        # Find out how many ligands we have by finding out how many .dok files we have
        dok_files = glob.glob(os.path.join(args.outfolder_ledock, '*.dok'))
        num_ligands = len(dok_files)-1 # out.dok contains all ligands so we don't count it

        for i in range(1, num_ligands+1):
            ligand_dok = os.path.join(args.outfolder_ledock, f"out_{i}.dok")
            ligand = Chem.MolFromPDBFile(ligand_dok, removeHs=False, sanitize=True)
            
            docked_complex = Chem.CombineMols(receptor, ligand)
            Chem.MolToPDBFile(docked_complex, os.path.join(args.outfolder_ledock, f"complex_{i}.pdb"))
    
    logger.info('Making complex... Done.')
            
def run_smina(args, logger):
    logger.info('Running smina...')
    subprocess.call(f"{args.smina_path} -r {args.receptor_pdbqt} -l {args.ligand_sdf} \
    --center_x {args.pocket_center[0]} --center_y {args.pocket_center[1]} --center_z {args.pocket_center[2]} \
    --size_x {args.pocket_size[0]} --size_y {args.pocket_size[1]} --size_z {args.pocket_size[2]} --out {os.path.join(args.outfolder_smina, 'out.sdf')} \
    --num_modes {args.num_modes} --exhaustiveness {args.exhaustiveness} --cpu {args.num_threads} --log {os.path.join(args.outfolder_smina, 'out.sdf')}",shell=True)
    logger.info('Running smina... Done.')

    # Split docked poses
    split_mol(args, logger, tool="smina")

    # Make complex
    make_complex(args, logger, tool="smina")
    
def run_ledock(args, logger):
    logger.info('Running LeDock...')

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
    dock_files = glob.glob(os.path.join(args.outfolder_input, '*.dok'))
    dock_file = dock_files[0]

    # Rename that file to out.dok
    if dock_file is not None:
        shutil.move(dock_file, os.path.join(args.outfolder_ledock, 'out.dok'))

    logger.info('Running LeDock... Done.')

    # Split docked poses
    split_mol(args, logger, tool="ledock")

    # Make complex
    make_complex(args, logger, tool="ledock")

def run_gd3(args, logger):
    logger.info('Running gd3...')
    
    dock_in = f"""
    data_directory   {args.gd3_path}/data/
    infile_pdb       {args.receptor_pdb}
    infile_ligand    {args.ligand_mol2}
    top_type         polarh
    fix_type         all
    ligdock_prefix   {args.outfolder_gd3}/
    grid_box_cntr    {args.pocket_center[0]} {args.pocket_center[1]} {args.pocket_center[2]}
    grid_n_elem      61 61 61
    grid_width       0.375
    weight_type      GalaxyDock3
    first_bank       rand
    max_trial        50000
    e0max            1000.0
    e1max            1000000.0
    n_proc           10
    """

    with open(os.path.join(args.outfolder_gd3, 'galaxydock.in'), 'w') as dock_in_f:
        dock_in_f.write(dock_in.strip() + "\n")

    subprocess.call(
        f"{args.gd3_path}/bin/GalaxyDock3 {os.path.join(args.outfolder_gd3, 'galaxydock.in')} > {os.path.join(args.outfolder_gd3, 'galaxydock.log')}", 
        shell=True)

    logger.info('Running gd3... Done.')

    # Split docked poses
    #split_mol(args, logger, tool="gd3")

    # Make complex
    #make_complex(args, logger, tool="gd3")

def consensus_dock(args, logger):
    # Convert receptor pdb to pdbqt format
    args.receptor_pdbqt = os.path.join(args.outfolder_input, 'receptor.pdbqt')
    pdb_to_pdbqt(args.receptor_pdb, args.receptor_pdbqt, logger, pH=args.pH)

    # Get stem from file name of args.ligand_sdf
    sdf_path = Path(args.ligand_sdf)
    sdf_stem = sdf_path.stem
    args.ligand_mol2 = os.path.join(args.outfolder_input, sdf_stem + '.mol2')

    # Convert ligand sdf to mol2
    sdf_to_mol2(args.ligand_sdf, 'LIG', args.ligand_mol2)

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

    # Run GalaxyDock3 docking
    run_gd3(args, logger)

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="CLI tool to perform consensus docking simulations using Smina and LeDock."
    )

    parser.add_argument('--outfolder', type=str, help='Base output directory (default: current directory)')
    parser.add_argument('--smina_path', type=str, default=False, help='Path to Smina executable (default: smina)')
    parser.add_argument('--ledock_path', type=str, default=False, help='Path to LeDock executable (default: ledock)')
    parser.add_argument('--lepro_path', type=str, default='lepro', help='Path to lepro executable (default: lepro)')
    parser.add_argument('--gd3_path', type=str, help='Path to GalaxyDock3 directory', default=False)
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

    # Create an input directory within outfolder
    os.makedirs(os.path.join(args.outfolder, 'input'), exist_ok=False)
    args.outfolder_input = os.path.join(args.outfolder, 'input')

    # Create output directory for smina within outfolder
    os.makedirs(os.path.join(args.outfolder, 'smina'), exist_ok=False)
    args.outfolder_smina = os.path.join(args.outfolder, 'smina')

    # Create output directory for ledock within outfolder
    os.makedirs(os.path.join(args.outfolder, 'ledock'), exist_ok=False)
    args.outfolder_ledock = os.path.join(args.outfolder, 'ledock')

    # Create output directory for gd3 within outfolder
    os.makedirs(os.path.join(args.outfolder, 'gd3'), exist_ok=False)
    args.outfolder_gd3 = os.path.join(args.outfolder, 'gd3')

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