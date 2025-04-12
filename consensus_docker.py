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

from Bio.PDB import PDBParser

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

def to_mol2(infile, mol_name, mol2_filepath, logger):

    logger.info('Converting protein/ligand to mol2 format...')
    if infile.endswith('.sdf'):
        mol = Chem.SDMolSupplier(infile)[0]
    elif infile.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(infile)

    # Optimize the molecule (optional)
    #AllChem.Compute2DCoords(mol)
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

    logger.info('Converting protein/ligand to mol2 format... Done.')

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

def write_gold_res_file(args, logger):
    
    logger.info('Writing gold res file...')
    # create a PDBParser object
    parser = PDBParser()

    # parse the PDB file
    structure = parser.get_structure("protein", args.pocket_pdb)

    # get the first model of the structure
    model = structure[0]

    # get the chains
    chains = list(model.get_chains())

    # loop over the residues in the chain
    res_list = []
    for chain in chains:
        for residue in chain:
            # get the residue number and name
            res_num = residue.get_id()[1]
            res_name = residue.get_resname()
            # print the residue number and name
            res_list.append("{}{}".format(res_name, res_num))

    res_list = ' '.join(res_list)

    res_list_path = os.path.join(args.outfolder_gold,"res_list.txt")

    with open(res_list_path, "w") as f:
        f.write(res_list)

    res_file_path = os.path.join(args.outfolder_gold,"res_file.txt")

    with open(res_list_path, 'r') as input_file, open(res_file_path, 'w') as output_file:
        for line in input_file:
            while len(line) > 200:
                split_index = line.rfind(' ', 0, 200)
                output_file.write(line[:split_index] + '\n')
                line = line[split_index+1:]
            output_file.write(line)

    with open(res_file_path, 'r+') as file:
        original_content = file.read()
        file.seek(0, 0)  # move the file pointer to the beginning of the file
        file.write('> <Gold.Protein.ActiveResidues>\n' + original_content)  

    logger.info('Writing gold res file... Done.')

def write_gold_conf_file(args, logger):
    logger.info('Writing gold conf file...')

    conf_in = f"""
  GOLD CONFIGURATION FILE

  AUTOMATIC SETTINGS
autoscale = 1

  POPULATION
popsiz = auto
select_pressure = auto
n_islands = auto
maxops = auto
niche_siz = auto

  GENETIC OPERATORS
pt_crosswt = auto
allele_mutatewt = auto
migratewt = auto

  FLOOD FILL
do_cavity = 1
cavity_file = {os.path.join(args.outfolder_gold,"res_file.txt")}
floodfill_center = list_of_residues

  DATA FILES
ligand_data_file {args.ligand_sdf} 20
param_file = DEFAULT
set_ligand_atom_types = 0
set_protein_atom_types = 0
directory = output
tordist_file = DEFAULT
make_subdirs = 0
save_lone_pairs = 1
fit_points_file = fit_pts.mol2
read_fitpts = 0

  FLAGS
internal_ligand_h_bonds = 0
flip_free_corners = 0
match_ring_templates = 0
flip_amide_bonds = 0
flip_planar_n = 1 flip_ring_NRR flip_ring_NHR
flip_pyramidal_n = 0
rotate_carboxylic_oh = flip
use_tordist = 1
postprocess_bonds = 1
rotatable_bond_override_file = DEFAULT
solvate_all = 1

  TERMINATION
early_termination = 1
n_top_solutions = 3
rms_tolerance = 1.5

  CONSTRAINTS
force_constraints = 0

  COVALENT BONDING
covalent = 0

  SAVE OPTIONS
save_score_in_file = 1
save_protein_torsions = 1
directory = {os.path.join(args.outfolder_gold,'output')}

  FITNESS FUNCTION SETTINGS
initial_virtual_pt_match_max = 3
relative_ligand_energy = 0
gold_fitfunc_path = plp
score_param_file = DEFAULT

  PROTEIN DATA
protein_datafile = {args.receptor_pdb}


    """

    with open(os.path.join(args.outfolder_gold, 'gold.in'), 'w') as f:
        f.write(conf_in)

    logger.info('Writing gold conf file... Done.')

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
        receptor = Chem.MolFromPDBFile(args.receptor_pdb, removeHs=False, sanitize=True)
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
        receptor = Chem.MolFromPDBFile(args.lepro_pdb, removeHs=False, sanitize=True)
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

    elif tool=="gold":
        receptor = Chem.MolFromPDBFile(args.receptor_pdb, removeHs=False, sanitize=True)
        if receptor is None:
            raise RuntimeError(f"Failed to load receptor from PDB: {args.receptor_pdb}")

        # Find out how many ligands we have by finding out how many .sdf files with following naming pattern:
        # gold_soln_{ligand_name}_m1_{pose_number}.sdf
        sdf_files = glob.glob(os.path.join(args.outfolder_gold, 'output', 'gold_soln_*.sdf')) 
        sdf_files = [file for file in sdf_files if file.endswith('.sdf')]
        num_ligands = len(sdf_files)
        for i in range(1, num_ligands+1):
            # Find out base filename of args.ligand_sdf, without its extension
            ligand_file = os.path.basename(args.ligand_sdf)
            ligand_file_stem = os.path.splitext(ligand_file)[0]
            # Construct the ligand sdf file name
            ligand_sdf = os.path.join(args.outfolder_gold, 'output', f"gold_soln_{ligand_file_stem}_m1_{i}.sdf")
            ligand_supplier = Chem.SDMolSupplier(ligand_sdf, removeHs=False)
            ligands = [m for m in ligand_supplier if m is not None]
            if not ligands:
                raise RuntimeError(f"No valid molecules found in SDF file: {ligand_file}")
            
            ligand = ligands[0]
            docked_complex = Chem.CombineMols(receptor, ligand)
            Chem.MolToPDBFile(docked_complex, os.path.join(args.outfolder_gold, f"complex_{i}.pdb"))
    
    logger.info('Making complex... Done.')

def parse_smina(args, logger):
    logger.info('Parsing smina output...')

    # Find out the .sdf output files in args.outfolder_smina, with the following pattern:
    # out_{pose_number}.sdf
    sdf_files = glob.glob(os.path.join(args.outfolder_smina, 'out_*.sdf'))
    sdf_files = [file for file in sdf_files if file.endswith('.sdf')]
    num_ligands = len(sdf_files)
    if num_ligands == 0:
        raise RuntimeError(f"No valid SDF files found in directory: {args.outfolder_smina}")
    logger.info(f'Found {num_ligands} SDF files in {args.outfolder_smina}')
    # Create a dataframe to store the results
    results = pd.DataFrame(columns=['Pose', 'SMINA_Score'])
    for i in range(1, num_ligands+1):
        ligand_sdf = os.path.join(args.outfolder_smina, f"out_{i}.sdf")
        ligand_supplier = Chem.SDMolSupplier(ligand_sdf, removeHs=False)
        ligands = [m for m in ligand_supplier if m is not None]
        if not ligands:
            raise RuntimeError(f"No valid molecules found in SDF file: {ligand_sdf}")
        ligand = ligands[0]
        smina_score = float(ligand.GetProp("minimizedAffinity"))
        results.loc[i-1] = [int(i), smina_score]

    # Save the dataframe to a CSV file in args.outfolder_smina
    results.to_csv(os.path.join(args.outfolder_smina, 'results.csv'), index=False)
    logger.info('Parsing smina output... Done.')

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

    # Parse smina output
    parse_smina(args, logger)
 
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

def run_gold(args, logger):
    logger.info('Running gold...')

    write_gold_res_file(args, logger)

    write_gold_conf_file(args, logger)

    subprocess.call(f"{args.gold_path} {os.path.join(args.outfolder_gold, 'gold.in')}", shell=True)

    make_complex(args, logger, tool="gold")

    logger.info('Running gold... Done.')

def consensus_dock(args, logger):
    # Convert receptor pdb to pdbqt format
    logger.info('Starting consensus_dock...')
    args.receptor_pdbqt = os.path.join(args.outfolder_input, 'receptor.pdbqt')
    pdb_to_pdbqt(args.receptor_pdb, args.receptor_pdbqt, logger, pH=args.pH)

    # Get stem from file name of args.ligand_sdf
    sdf_path = Path(args.ligand_sdf)
    sdf_stem = sdf_path.stem
    args.ligand_mol2 = os.path.join(args.outfolder_input, sdf_stem + '.mol2')

    # Convert ligand sdf to mol2
    to_mol2(args.ligand_sdf, 'LIG', args.ligand_mol2, logger)

    # Convert protein pdb to protein mol2
    args.receptor_mol2 = os.path.join(args.outfolder_input, 'receptor.mol2')
    to_mol2(args.receptor_pdb, 'PRO', args.receptor_mol2, logger)

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
    #run_gd3(args, logger)

    # Run gold docking
    run_gold(args, logger)

    logger.info('########## Finished consensus_docker.py #########')

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="CLI tool to perform consensus docking simulations using Smina and LeDock."
    )

    parser.add_argument('--outfolder', type=str, help='Base output directory (default: current directory)')
    parser.add_argument('--smina_path', type=str, default=False, help='Path to Smina executable')
    parser.add_argument('--ledock_path', type=str, default=False, help='Path to LeDock executable')
    parser.add_argument('--lepro_path', type=str, default='lepro', help='Path to lepro executable')
    parser.add_argument('--gold_path', type=str, default=False, help='Path to gold executable')
    parser.add_argument('--gd3_path', type=str, help='Path to GalaxyDock3 directory', default=False)
    parser.add_argument('--pH', type=float, default=7.4, help='pH value for adding missing hydrogens (default: 7.4)')
    parser.add_argument('--receptor_pdb', type=str, help='Path to receptor PDB file')
    parser.add_argument('--ligand_sdf', type=str, help='Path to ligand SDF file')
    parser.add_argument('--pocket_pdb', type=str, help='Path to pocket PDB file')
    parser.add_argument('--exhaustiveness', type=int, default=12, help='Exhaustiveness value for Smina (default: 12)')
    parser.add_argument('--num_modes', type=int, default=20, help='Number of modes for Smina (default: 20)')
    parser.add_argument('--num_threads', type=int, default=1, help='Number of threads for Smina (default: 1)')
    parser.add_argument('--cutoff_value', type=float, default=-7.0, help='SMINA_Score cutoff value for analysis (default: -7.0)')
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

    # Create output directory for gold within outfolder
    os.makedirs(os.path.join(args.outfolder, 'gold'), exist_ok=False)
    args.outfolder_gold = os.path.join(args.outfolder, 'gold')

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