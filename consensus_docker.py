from pathlib import Path
import subprocess, os, glob, multiprocessing, argparse, logging, warnings, uuid, sys, shutil, time
from datetime import datetime
import numpy as np
import pandas as pd
from logging.handlers import RotatingFileHandler
import re

# filter warnings
warnings.filterwarnings("ignore")


def _configure_openbabel_logging_silently():
    """Best-effort: silence OpenBabel logs without importing at module import time."""
    try:
        from openbabel import pybel
        pybel.ob.obErrorLog.SetOutputLevel(0)
    except Exception:
        # OpenBabel not installed or not needed for current workflow
        pass

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
    _configure_openbabel_logging_silently()
    from openbabel import pybel
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
    
    # Store the original working directory
    original_cwd = os.getcwd()
    
    # Create a temporary directory with the shortest possible path
    # Try different locations in order of preference (shortest path first)
    temp_dir_path = None
    max_attempts = 100  # Maximum attempts to find a unique directory name
    
    # Try different root locations for the shortest possible path
    possible_roots = ["/tmp", "/var/tmp", "/", "/home", original_cwd]
    
    for root in possible_roots:
        for attempt in range(max_attempts):
            # Create a more unique temporary directory name
            # Include process ID, timestamp, and random UUID
            timestamp = int(datetime.now().timestamp() * 1000000)  # microseconds
            process_id = os.getpid()
            random_suffix = uuid.uuid4().hex[:8]
            temp_dir_name = f"lp_{process_id}_{timestamp}_{random_suffix}"
            
            candidate_path = os.path.join(root, temp_dir_name)
            
            # Check if directory already exists
            if os.path.exists(candidate_path):
                continue  # Try next name
            
            try:
                # Try to create the directory
                os.makedirs(candidate_path, exist_ok=False)  # Don't allow existing
                
                # Test if we can write to it
                test_file = os.path.join(candidate_path, "test")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                
                temp_dir_path = candidate_path
                logger.info(f'Using temporary directory: {temp_dir_path}')
                break
                
            except (OSError, PermissionError, FileExistsError):
                # If we can't use this location or it already exists, try next name
                # DO NOT remove existing directories - they might be in use by parallel processes
                continue
        
        # If we found a working directory, break out of the root loop
        if temp_dir_path is not None:
            break
    
    if temp_dir_path is None:
        raise RuntimeError(f"Could not create a unique temporary directory after {max_attempts} attempts in any accessible location")
    
    try:
        # Copy receptor PDB to temporary directory for shorter path usage
        temp_receptor_name = "receptor.pdb"
        temp_receptor_path = os.path.join(temp_dir_path, temp_receptor_name)
        shutil.copy2(args.receptor_pdb, temp_receptor_path)
        logger.debug(f'Copied receptor PDB to temporary directory: {temp_receptor_path}')
        
        # Change to the temporary directory before running lepro
        os.chdir(temp_dir_path)
        
        # Run lepro with the local receptor file (shortest possible path)
        subprocess.run([args.lepro_path, temp_receptor_name])
        
        # Check if pro.pdb was generated successfully
        temp_pro_path = os.path.join(temp_dir_path, 'pro.pdb')
        if not os.path.exists(temp_pro_path):
            raise RuntimeError(f"lepro failed to generate pro.pdb in temporary directory: {temp_dir_path}")
        
        # Check if pro.pdb is not empty
        if os.path.getsize(temp_pro_path) == 0:
            raise RuntimeError(f"lepro generated an empty pro.pdb file. This may be due to path length issues or other lepro errors.")
        
        # Ensure the destination directory exists before moving pro.pdb
        # Use absolute path to ensure correct destination regardless of current working directory
        ledock_output_dir = os.path.abspath(args.outfolder_ledock)
        os.makedirs(ledock_output_dir, exist_ok=True)
        logger.debug(f'Ensured destination directory exists: {ledock_output_dir}')
        
        # Move pro.pdb to the ledock output directory using absolute paths
        final_pro_path = os.path.join(ledock_output_dir, 'pro.pdb')
        shutil.move(temp_pro_path, final_pro_path)
        logger.debug(f'Successfully moved pro.pdb from {temp_pro_path} to {final_pro_path}')
        
        logger.info('Preparing input pdb for LeDock by lepro exe... Done.')
        
        # Return final path of pro.pdb
        return final_pro_path
        
    finally:
        # Always restore the original working directory first
        os.chdir(original_cwd)
        
        # Check if we should keep the temporary directory for debugging
        keep_temp_dir = os.environ.get('LEPRO_DEBUG_KEEP_TEMP', '').lower() in ('1', 'true', 'yes', 'on')
        
        if keep_temp_dir:
            logger.info(f"DEBUG: Keeping temporary directory for debugging: {temp_dir_path}")
            logger.info(f"DEBUG: You can manually run lepro by going to: {temp_dir_path}")
            logger.info(f"DEBUG: Command to run: {args.lepro_path} receptor.pdb")
            logger.info(f"DEBUG: To disable this behavior, unset LEPRO_DEBUG_KEEP_TEMP environment variable")
        else:
            # Clean up the temporary directory - this is critical for parallel runs
            if temp_dir_path and os.path.exists(temp_dir_path):
                try:
                    shutil.rmtree(temp_dir_path)
                    logger.debug(f"Successfully removed temporary directory: {temp_dir_path}")
                except Exception as e:
                    logger.warning(f"Could not remove temporary directory {temp_dir_path}: {e}")
                    # Try to remove it again after a short delay
                    try:
                        import time
                        time.sleep(0.1)
                        shutil.rmtree(temp_dir_path)
                        logger.debug(f"Successfully removed temporary directory on second attempt: {temp_dir_path}")
                    except Exception as e2:
                        logger.error(f"Failed to remove temporary directory even on second attempt {temp_dir_path}: {e2}")

def to_mol2(infile, mol_name, mol2_filepath, logger):

    logger.info('Converting protein/ligand to mol2 format...')
    from rdkit import Chem
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
    try:
        from opencadd.io.dataframe import DataFrame
        structure_df = DataFrame.from_file(args.pocket_pdb)
        positions = np.array([structure_df["atom.x"].values,structure_df["atom.y"].values,structure_df["atom.z"].values])
        min_coords = np.min(positions,axis=1)
        max_coords = np.max(positions,axis=1)
        pocket_center = list(map(str,(max_coords + min_coords) / 2))
        pocket_size = list(map(str,(max_coords - min_coords) + 5))
        
        # Enhanced debugging for pocket coordinates
        logger.info(f"Calculated pocket coordinates:")
        logger.info(f"  min_coords: {min_coords} (type: {type(min_coords)})")
        logger.info(f"  max_coords: {max_coords} (type: {type(max_coords)})")
        logger.info(f"  pocket_center: {pocket_center} (type: {type(pocket_center)})")
        logger.info(f"  pocket_size: {pocket_size} (type: {type(pocket_size)})")
        
        # Validate that all values are proper strings
        for i, (center, size) in enumerate(zip(pocket_center, pocket_size)):
            logger.info(f"  Coordinate {i}: center='{center}' (type: {type(center)}), size='{size}' (type: {type(size)})")
            # Check for any problematic characters or formatting
            if not isinstance(center, str) or not isinstance(size, str):
                logger.warning(f"  Non-string coordinate detected at index {i}")
            if '{' in center or '}' in center or '{' in size or '}' in size:
                logger.warning(f"  Suspicious format characters found at index {i}")
        
        logger.info('Getting pocket coordinates... Done.')
        return pocket_center, pocket_size, min_coords, max_coords
        
    except Exception as e:
        logger.error(f"Error calculating pocket coordinates: {e}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

def write_gold_res_file(args, logger):
    
    logger.info('Writing gold res file...')
    # create a PDBParser object
    from Bio.PDB import PDBParser
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
    _configure_openbabel_logging_silently()
    from openbabel import pybel
    if tool == "smina":
        docked_poses_path = Path(os.path.join(args.outfolder_smina, 'out.sdf'))
        molecules = pybel.readfile("sdf", str(docked_poses_path))
        for i, molecule in enumerate(molecules, 1):
            molecule.write("sdf", os.path.join(args.outfolder_smina, f"out_{i}.sdf"), overwrite=True)

    elif tool == "gnina":
        docked_poses_path = Path(os.path.join(args.outfolder_gnina, 'out.sdf'))
        molecules = pybel.readfile("sdf", str(docked_poses_path))
        for i, molecule in enumerate(molecules, 1):
            molecule.write("sdf", os.path.join(args.outfolder_gnina, f"out_{i}.sdf"), overwrite=True)

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
    from rdkit import Chem

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

    elif tool == "gnina":
        receptor = Chem.MolFromPDBFile(args.receptor_pdb, removeHs=False, sanitize=True)
        if receptor is None:
            raise RuntimeError(f"Failed to load receptor from PDB: {args.receptor_pdb}")

        # Find out how many ligands we have
        ligand_sdf = os.path.join(args.outfolder_gnina, 'out.sdf')
        ligand_supplier = Chem.SDMolSupplier(ligand_sdf, removeHs=False)
        ligands = [m for m in ligand_supplier if m is not None]
        if not ligands:
            raise RuntimeError(f"No valid molecules found in SDF file: {ligand_file}")
        
        num_ligands = len(list(ligands))

        for i in range(1, num_ligands+1):
            ligand = ligands[i-1]
            
            docked_complex = Chem.CombineMols(receptor, ligand)
            Chem.MolToPDBFile(docked_complex, os.path.join(args.outfolder_gnina, f"complex_{i}.pdb"))

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
    from rdkit import Chem

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

def parse_gnina(args, logger):
    logger.info('Parsing gnina output...')

    # Read the gnina log file
    log_file = os.path.join(args.outfolder_gnina, 'out.log')
    if not os.path.exists(log_file):
        raise RuntimeError(f"Gnina log file not found: {log_file}")
    
    # Parse the log file to extract affinity, CNN pose score, and CNN affinity
    results = pd.DataFrame(columns=['Pose', 'GNINA_Affinity', 'GNINA_CNN_pose', 'GNINA_CNN_affinity'])
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Find the line with "mode |  affinity" to locate the start of results
    # The header spans two lines, followed by a separator line (-----)
    result_start_idx = None
    for i, line in enumerate(lines):
        if 'mode |' in line and 'affinity' in line:
            # Skip the header lines (2 lines) and the separator line (1 line)
            result_start_idx = i + 3
            break
    
    if result_start_idx is None:
        raise RuntimeError(f"Could not find results in gnina log file: {log_file}")
    
    # Parse the results
    pose_num = 1
    for line in lines[result_start_idx:]:
        line = line.strip()
        if not line or line.startswith('---'):
            continue
        
        # Parse the line format:
        # mode | affinity | intramol | CNN pose score | CNN affinity
        parts = line.split()
        if len(parts) < 5:
            break  # End of results
        
        try:
            mode = int(parts[0])
            affinity = float(parts[1])
            # Skip intramol (parts[2])
            cnn_pose = float(parts[3])
            cnn_affinity = float(parts[4]);
            
            results.loc[pose_num - 1] = [pose_num, affinity, cnn_pose, cnn_affinity]
            pose_num += 1
        except (ValueError, IndexError):
            # Skip lines that don't match the expected format
            continue
    
    if len(results) == 0:
        raise RuntimeError(f"No valid results found in gnina log file: {log_file}")
    
    # Save the dataframe to a CSV file in args.outfolder_gnina
    results.to_csv(os.path.join(args.outfolder_gnina, 'results.csv'), index=False)
    logger.info(f'Parsing gnina output... Done. Found {len(results)} poses.')

def parse_ledock(args, logger):
    logger.info('Parsing ledock output...')
    # Find out the .sdf output files in args.outfolder_smina, with the following pattern:
    # #out_{pose_number}.dok

    start_pattern = r"Score:"
    end_pattern = r"kcal/mol"

    dok_files = glob.glob(os.path.join(args.outfolder_ledock, 'out_*.dok'))
    dok_files = [file for file in dok_files if file.endswith('.dok')]

    if not dok_files:
        logger.error(f'No .dok files found in the specified folder: {args.input_folder}')
        return None

    logger.info(f'Found {len(dok_files)} .dok files.')

    # Start a dataframe to store results
    df = pd.DataFrame(columns=['Pose', 'LeDock_Score'])
    for file_name in dok_files:
        # Find out the pose number from the file name
        pose_number = int(os.path.basename(file_name).split('_')[1].split('.')[0])
        with open(file_name, "r") as file:
            contents = file.read()
            match = re.search(start_pattern + r"(.*?)" + end_pattern, contents)
            if match:
                score = float(match.group(1).strip())
            else:
                logger.error(f"Score not found in {file_name}")
                continue

        # Append the results to the dataframe
        df.loc[len(df)] = [pose_number, score]

    # Sort the dataframe by score in descending order
    results = df.sort_values(by='LeDock_Score', ascending=True)

    # Save the dataframe to a CSV file in args.outfolder_ledock
    results.to_csv(os.path.join(args.outfolder_ledock, 'results.csv'), index=False)
    logger.info('Parsing ledock output... Done.') 

def parse_gold(args, logger):
    logger.info('Parsing gold output...')
    
    # Find out the file ending with .rnk in args.outfolder_gold
    rnk_file = glob.glob(os.path.join(args.outfolder_gold, 'output', '*.rnk'))
    rnk_file = rnk_file[0]

    if not rnk_file:
        logger.error(f'No .rnk files found in the specified folder: {args.outfolder_gold}')
        return None

    with open(rnk_file, "r") as file:
        lines = file.readlines()
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 2:
                continue
            # If the first element in parts is "Mol"
            if parts[0] == "Mol":
                col_names = ["Pose", parts[2], parts[3], parts[4], parts[5], parts[6], parts[7], parts[8], parts[9]]
                df = pd.DataFrame(columns=col_names)
                continue

            # If the first element can be converted to integer
            try:
                pose_number = int(parts[0])
                row = [pose_number] + [float(x) for x in parts[1:]]
                df.loc[len(df)] = row
                
            except ValueError:
                continue
        
    # Sort the dataframe by score in descending order
    results = df.sort_values(by=col_names[1], ascending=False)
    # Save the dataframe to a CSV file in args.outfolder_gold
    results.to_csv(os.path.join(args.outfolder_gold, 'results.csv'), index=False)
    logger.info('Parsing gold output... Done.')

def calculate_rmsd(args, logger):
    logger.info('Calculating rmsd...')
    t0 = time.perf_counter()
    rmsd_result = []

    try:
        from prody import parsePDB, calcRMSD
    except Exception as e:
        raise RuntimeError(
            "ProDy is required for RMSD calculation but could not be imported. "
            "Install prody in this environment and retry."
        ) from e

    # Define available tools and their configurations
    tools_config = {
        'smina': {
            'folder': getattr(args, 'outfolder_smina', None),
            'score_column': 'SMINA_Score'
        },
        'gnina': {
            'folder': getattr(args, 'outfolder_gnina', None),
            'score_column': 'GNINA_Affinity'
        },
        'ledock': {
            'folder': getattr(args, 'outfolder_ledock', None),
            'score_column': 'LeDock_Score'
        },
        'gold': {
            'folder': getattr(args, 'outfolder_gold', None),
            'score_column': None  # Will use iloc[0, 1] for GOLD
        }
    }

    # Collect available tools with their complex files and results
    available_tools = {}
    
    # Cache parsed ligand selections to avoid re-parsing PDB files N^2 times
    parsed_pose_cache = {}

    def _load_pose_selection(pdb_path):
        cached = parsed_pose_cache.get(pdb_path)
        if cached is not None:
            return cached
        ag = parsePDB(pdb_path)
        sel = ag.select("hetero and noh") if ag is not None else None
        parsed_pose_cache[pdb_path] = sel
        return sel

    for tool_name, config in tools_config.items():
        folder = config['folder']
        if folder and os.path.exists(folder):
            complex_files = sorted(glob.glob(os.path.join(folder, "complex_*.pdb")))
            results_file = os.path.join(folder, 'results.csv')
            
            if complex_files and os.path.exists(results_file):
                try:
                    results_df = pd.read_csv(results_file)
                    # Pre-map pose -> score for faster lookups
                    pose_to_score = {}
                    if 'Pose' in results_df.columns and len(results_df.columns) >= 2:
                        for _, row in results_df.iterrows():
                            try:
                                pose_num = int(row['Pose'])
                            except Exception:
                                continue
                            if config['score_column'] and config['score_column'] in results_df.columns:
                                pose_to_score[pose_num] = row[config['score_column']]
                            else:
                                # GOLD special handling (score in second column)
                                pose_to_score[pose_num] = row.iloc[1]

                    available_tools[tool_name] = {
                        'complex_files': complex_files,
                        'results': results_df,
                        'pose_to_score': pose_to_score,
                        'score_column': config['score_column']
                    }
                    logger.info(f"Found {len(complex_files)} poses for {tool_name}")
                except Exception as e:
                    logger.warning(f"Could not load {tool_name} results: {e}")

    # Get list of tool names
    tool_names = list(available_tools.keys())
    
    if len(tool_names) < 2:
        logger.warning(f'Not enough tools with valid results for RMSD calculation. Found: {tool_names}')
        return

    logger.info(f"Calculating RMSD between tools: {', '.join(tool_names)}")

    # Calculate RMSD for all pairs of tools
    for i in range(len(tool_names)):
        for j in range(i + 1, len(tool_names)):
            tool1_name = tool_names[i]
            tool2_name = tool_names[j]
            
            tool1_data = available_tools[tool1_name]
            tool2_data = available_tools[tool2_name]
            
            logger.info(f"Comparing {tool1_name} vs {tool2_name}...")
            pair_t0 = time.perf_counter()
            
            for out1 in tool1_data['complex_files']:
                pose_number1 = re.search(r'complex_(\d+)\.pdb', out1).group(1)

                pose_num1_int = int(pose_number1)
                score1 = tool1_data.get('pose_to_score', {}).get(pose_num1_int)
                if score1 is None:
                    continue
                
                for out2 in tool2_data['complex_files']:
                    pose_number2 = re.search(r'complex_(\d+)\.pdb', out2).group(1)

                    pose_num2_int = int(pose_number2)
                    score2 = tool2_data.get('pose_to_score', {}).get(pose_num2_int)
                    if score2 is None:
                        continue
                    
                    # Calculate RMSD
                    try:
                        pose1 = _load_pose_selection(out1)
                        pose2 = _load_pose_selection(out2)
                        if pose1 is None or pose2 is None:
                            continue
                        rmsd = calcRMSD(pose1, pose2)
                        
                        rmsd_result.append({
                            'Tool1': tool1_name.upper(),
                            'Tool2': tool2_name.upper(),
                            'PoseNumber1': pose_number1,
                            'PoseNumber2': pose_number2,
                            'Score1': score1,
                            'Score2': score2,
                            'File1': out1.split("/")[-1],
                            'File2': out2.split("/")[-1],
                            'RMSD': rmsd
                        })
                    except Exception as e:
                        logger.warning(f"Error calculating RMSD for {out1} vs {out2}: {e}")

            logger.info(f"Completed {tool1_name} vs {tool2_name} in {time.perf_counter() - pair_t0:.2f}s")

    if rmsd_result:
        rmsd_result = pd.DataFrame(rmsd_result)
        rmsd_result.to_csv(os.path.join(args.outfolder, 'final_results.csv'), index=False)
        logger.info(f'Calculating rmsd... Done. Calculated {len(rmsd_result)} RMSD comparisons.')
        logger.info(f'Results saved to: {os.path.join(args.outfolder, "final_results.csv")}')
    else:
        logger.warning('No RMSD results calculated. Not enough valid docking outputs.')

    logger.info(f"RMSD calculation total time: {time.perf_counter() - t0:.2f}s")

    
def run_smina_single(args, logger, exhaustiveness_val, temp_outdir=None):
    """Run a single Smina docking with specified exhaustiveness value"""
    output_dir = temp_outdir if temp_outdir else args.outfolder_smina
    
    logger.info(f'Running smina with exhaustiveness {exhaustiveness_val}...')
    
    # Enhanced debugging - log the state of all critical variables
    logger.info(f"=== SMINA DEBUG INFO (Exhaustiveness {exhaustiveness_val}) ===")
    logger.info(f"smina_path: {args.smina_path}")
    logger.info(f"receptor_pdbqt: {args.receptor_pdbqt}")
    logger.info(f"ligand_sdf: {args.ligand_sdf}")
    logger.info(f"pocket_center: {args.pocket_center} (type: {type(args.pocket_center)})")
    logger.info(f"pocket_size: {args.pocket_size} (type: {type(args.pocket_size)})")
    logger.info(f"output_dir: {output_dir}")
    logger.info(f"num_modes: {args.num_modes}")
    logger.info(f"num_threads: {args.num_threads}")
    
    # Detailed inspection of pocket coordinates
    try:
        logger.info(f"pocket_center elements: [{args.pocket_center[0]}, {args.pocket_center[1]}, {args.pocket_center[2]}]")
        logger.info(f"pocket_center element types: [{type(args.pocket_center[0])}, {type(args.pocket_center[1])}, {type(args.pocket_center[2])}]")
        logger.info(f"pocket_size elements: [{args.pocket_size[0]}, {args.pocket_size[1]}, {args.pocket_size[2]}]")
        logger.info(f"pocket_size element types: [{type(args.pocket_size[0])}, {type(args.pocket_size[1])}, {type(args.pocket_size[2])}]")
    except Exception as coord_error:
        logger.error(f"Error inspecting coordinate elements: {coord_error}")
        logger.error(f"pocket_center raw: {repr(args.pocket_center)}")
        logger.error(f"pocket_size raw: {repr(args.pocket_size)}")
    
    # Build the command as a single string with safer formatting
    try:
        # Convert coordinates to strings with explicit error handling
        center_x_str = str(args.pocket_center[0])
        center_y_str = str(args.pocket_center[1])
        center_z_str = str(args.pocket_center[2])
        size_x_str = str(args.pocket_size[0])
        size_y_str = str(args.pocket_size[1])
        size_z_str = str(args.pocket_size[2])
        
        logger.info(f"Converted coordinates - center: [{center_x_str}, {center_y_str}, {center_z_str}], size: [{size_x_str}, {size_y_str}, {size_z_str}]")
        
        cmd = (f"{args.smina_path} -r {args.receptor_pdbqt} -l {args.ligand_sdf} "
               f"--center_x {center_x_str} --center_y {center_y_str} --center_z {center_z_str} "
               f"--size_x {size_x_str} --size_y {size_y_str} --size_z {size_z_str} "
               f"--out {os.path.join(output_dir, 'out.sdf')} "
               f"--num_modes {args.num_modes} --exhaustiveness {exhaustiveness_val} "
               f"--cpu {args.num_threads} --log {os.path.join(output_dir, 'out.log')}")
        
        logger.info(f"Constructed command: {cmd}")
        
    except Exception as e:
        logger.error(f"Error constructing smina command: {e}")
        logger.error(f"pocket_center content: {args.pocket_center}")
        logger.error(f"pocket_size content: {args.pocket_size}")
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Exception args: {e.args}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise
    
    logger.info(f"=== END SMINA DEBUG INFO ===")
    
    # Execute the command and capture any errors
    try:
        result = subprocess.call(cmd, shell=True)
        if result != 0:
            logger.warning(f"Smina command returned non-zero exit code: {result}")
        logger.info(f'Running smina with exhaustiveness {exhaustiveness_val}... Done.')
    except Exception as exec_error:
        logger.error(f"Error executing smina command: {exec_error}")
        logger.error(f"Command that failed: {cmd}")
        raise

def check_smina_convergence(prev_results, current_results, logger, rmsd_threshold=1.5, score_threshold=0.1):
    """
    Check if Smina results have converged between two exhaustiveness levels
    
    Parameters:
    -----------
    prev_results : pd.DataFrame
        Results from previous exhaustiveness level
    current_results : pd.DataFrame  
        Results from current exhaustiveness level
    logger : logging.Logger
        Logger instance for output
    rmsd_threshold : float, default=1.5
        RMSD threshold in Angstroms for convergence (default: 1.5)
    score_threshold : float, default=0.1
        Score difference threshold in kcal/mol for convergence (default: 0.1)
    
    Returns:
    --------
    bool : True if converged, False otherwise
    """
    if prev_results is None or current_results is None:
        return False
    
    # Compare best scores (lowest values)
    prev_best_score = prev_results['SMINA_Score'].min()
    current_best_score = current_results['SMINA_Score'].min()
    score_diff = abs(prev_best_score - current_best_score)
    
    logger.info(f"Score comparison: Previous best = {prev_best_score:.3f}, Current best = {current_best_score:.3f}, Difference = {score_diff:.3f}")
    
    # Check score convergence
    score_converged = score_diff <= score_threshold
    
    # Always calculate RMSD between best poses for convergence check
    rmsd_converged = False
    rmsd_value = None
    
    try:
        # Get the best poses from both runs
        prev_best_idx = prev_results['SMINA_Score'].idxmin()
        current_best_idx = current_results['SMINA_Score'].idxmin()
        
        prev_best_pose = prev_results.loc[prev_best_idx, 'Pose']
        current_best_pose = current_results.loc[current_best_idx, 'Pose']
        
        # Ensure pose numbers are integers
        prev_best_pose = int(prev_best_pose)
        current_best_pose = int(current_best_pose)
        
        logger.debug(f"Previous best pose number: {prev_best_pose}")
        logger.debug(f"Current best pose number: {current_best_pose}")
        
        # Load the complex PDB files for RMSD calculation
        prev_complex = f"complex_{prev_best_pose}.pdb"
        current_complex = f"complex_{current_best_pose}.pdb"
        
        logger.debug(f"Looking for previous complex: {prev_complex}")
        logger.debug(f"Looking for current complex: {current_complex}")
        
        # Check if files exist in temporary directories
        prev_complex_path = None
        current_complex_path = None
        
        # Find the files in the appropriate directories
        if hasattr(check_smina_convergence, 'prev_temp_dir') and check_smina_convergence.prev_temp_dir:
            prev_complex_path = os.path.join(check_smina_convergence.prev_temp_dir, prev_complex)
            logger.debug(f"Previous complex path: {prev_complex_path}")
            logger.debug(f"Previous complex exists: {os.path.exists(prev_complex_path)}")
        else:
            logger.warning("Previous temp dir attribute not set or is None")
            
        if hasattr(check_smina_convergence, 'current_temp_dir') and check_smina_convergence.current_temp_dir:
            current_complex_path = os.path.join(check_smina_convergence.current_temp_dir, current_complex)
            logger.debug(f"Current complex path: {current_complex_path}")
            logger.debug(f"Current complex exists: {os.path.exists(current_complex_path)}")
        else:
            logger.warning("Current temp dir attribute not set or is None")
        
        if prev_complex_path and current_complex_path and os.path.exists(prev_complex_path) and os.path.exists(current_complex_path):
            logger.debug(f"Both complex files found, calculating RMSD...")
            # Calculate RMSD between best poses
            pose1 = parsePDB(prev_complex_path).select("hetero and noh")
            pose2 = parsePDB(current_complex_path).select("hetero and noh")
            rmsd_value = calcRMSD(pose1, pose2)
            
            logger.info(f"RMSD between best poses: {rmsd_value:.3f}")
            rmsd_converged = rmsd_value <= rmsd_threshold
        else:
            # More detailed debugging for missing files
            logger.warning("Could not calculate RMSD - complex PDB files not found. Cannot assess convergence properly.")
            logger.warning(f"Expected previous complex: {prev_complex_path}")
            logger.warning(f"  - Path exists: {os.path.exists(prev_complex_path) if prev_complex_path else 'N/A'}")
            logger.warning(f"  - Path is file: {os.path.isfile(prev_complex_path) if prev_complex_path and os.path.exists(prev_complex_path) else 'N/A'}")
            logger.warning(f"Expected current complex: {current_complex_path}")
            logger.warning(f"  - Path exists: {os.path.exists(current_complex_path) if current_complex_path else 'N/A'}")
            logger.warning(f"  - Path is file: {os.path.isfile(current_complex_path) if current_complex_path and os.path.exists(current_complex_path) else 'N/A'}")
            
            logger.info("Available files in previous temp dir:")
            if hasattr(check_smina_convergence, 'prev_temp_dir') and check_smina_convergence.prev_temp_dir and os.path.exists(check_smina_convergence.prev_temp_dir):
                files = os.listdir(check_smina_convergence.prev_temp_dir)
                complex_files = [f for f in files if f.startswith('complex_') and f.endswith('.pdb')]
                logger.info(f"  Total files: {len(files)}")
                logger.info(f"  Complex files: {complex_files}")
                for f in sorted(files):
                    logger.info(f"    {f}")
            else:
                logger.info("  Previous temp dir not accessible")
                
            logger.info("Available files in current temp dir:")
            if hasattr(check_smina_convergence, 'current_temp_dir') and check_smina_convergence.current_temp_dir and os.path.exists(check_smina_convergence.current_temp_dir):
                files = os.listdir(check_smina_convergence.current_temp_dir)
                complex_files = [f for f in files if f.startswith('complex_') and f.endswith('.pdb')]
                logger.info(f"  Total files: {len(files)}")
                logger.info(f"  Complex files: {complex_files}")
                for f in sorted(files):
                    logger.info(f"    {f}")
            else:
                logger.info("  Current temp dir not accessible")
            return False  # Cannot assess convergence without RMSD
            
    except Exception as e:
        logger.error(f"Error calculating RMSD for convergence check: {e}")
        return False  # Cannot assess convergence without RMSD
    
    # Both conditions must be satisfied for convergence
    converged = score_converged and rmsd_converged
    
    logger.info(f"Convergence check results:")
    logger.info(f"  Score converged (diff ≤ {score_threshold}): {score_converged} (diff = {score_diff:.3f})")
    
    # Safe formatting for RMSD value to avoid format specifier errors
    if rmsd_value is not None:
        try:
            rmsd_str = f"{float(rmsd_value):.3f}"
        except (ValueError, TypeError):
            rmsd_str = str(rmsd_value)
    else:
        rmsd_str = "N/A"
    
    logger.info(f"  RMSD converged (≤ {rmsd_threshold}): {rmsd_converged} (RMSD = {rmsd_str})")
    logger.info(f"  Overall convergence: {converged}")
    
    if converged:
        logger.info("Convergence achieved: BOTH score similarity AND low RMSD between best poses")
    else:
        if not score_converged and not rmsd_converged:
            logger.info("No convergence: BOTH score difference and RMSD are too high")
        elif not score_converged:
            logger.info("No convergence: Score difference is too high")
        elif not rmsd_converged:
            logger.info("No convergence: RMSD between best poses is too high")
    
    return converged

def run_smina(args, logger):
    if not hasattr(args, 'adaptive_exhaustiveness') or not args.adaptive_exhaustiveness:
        # Run standard Smina docking
        run_smina_single(args, logger, args.exhaustiveness)
        
        # Split docked poses
        split_mol(args, logger, tool="smina")
        
        # Make complex
        make_complex(args, logger, tool="smina")
        
        # Parse smina output
        parse_smina(args, logger)
        return
    
    # Adaptive exhaustiveness strategy
    logger.info('Running smina with adaptive exhaustiveness strategy...')
    
    # Preserve pocket coordinates - make deep copies to prevent modification
    original_pocket_center = args.pocket_center.copy() if hasattr(args.pocket_center, 'copy') else list(args.pocket_center)
    original_pocket_size = args.pocket_size.copy() if hasattr(args.pocket_size, 'copy') else list(args.pocket_size)
    
    logger.debug(f"Preserved original pocket_center: {original_pocket_center}")
    logger.debug(f"Preserved original pocket_size: {original_pocket_size}")
    
    # Generate exhaustiveness levels using configurable parameters
    exhaustiveness_levels = list(range(8, args.maximum_exhaustiveness + 1, args.exhaustiveness_increment))
    
    # Ensure we include the user-specified exhaustiveness if it's not in our list
    if args.exhaustiveness not in exhaustiveness_levels:
        exhaustiveness_levels.append(args.exhaustiveness)
        exhaustiveness_levels.sort()
    
    # Log the strategy being used
    logger.info(f"Adaptive exhaustiveness strategy:")
    logger.info(f"  Levels to try: {exhaustiveness_levels}")
    logger.info(f"  Increment: {args.exhaustiveness_increment}")
    logger.info(f"  Maximum: {args.maximum_exhaustiveness}")
    logger.info(f"  RMSD threshold: {args.convergence_rmsd_threshold} Å")
    logger.info(f"  Score threshold: {args.convergence_score_threshold} kcal/mol")
    
    prev_results = None
    best_exhaustiveness = None
    temp_directories = []
    
    try:
        for i, exhaustiveness_val in enumerate(exhaustiveness_levels):
            logger.info(f"Trying exhaustiveness level {exhaustiveness_val} ({i+1}/{len(exhaustiveness_levels)})")
            
            # Ensure pocket coordinates are preserved for each iteration
            args.pocket_center = original_pocket_center.copy()
            args.pocket_size = original_pocket_size.copy()
            logger.debug(f"Iteration {i+1}: pocket_center = {args.pocket_center}, pocket_size = {args.pocket_size}")
            
            # Create temporary directory for this run
            temp_dir = os.path.join(args.outfolder_smina, f"temp_exh_{exhaustiveness_val}")
            os.makedirs(temp_dir, exist_ok=True)
            temp_directories.append(temp_dir)
            
            try:
                # Run Smina with current exhaustiveness
                run_smina_single(args, logger, exhaustiveness_val, temp_dir)
                
                # Process results
                original_outfolder_smina = args.outfolder_smina
                args.outfolder_smina = temp_dir
                
                try:
                    split_mol(args, logger, tool="smina")
                    make_complex(args, logger, tool="smina")
                    parse_smina(args, logger)
                    
                    # Verify that complex files were created
                    complex_files = glob.glob(os.path.join(temp_dir, "complex_*.pdb"))
                    logger.debug(f"Complex files created in {temp_dir}: {[os.path.basename(f) for f in complex_files]}")
                    
                    # Load current results
                    current_results = pd.read_csv(os.path.join(temp_dir, 'results.csv'))
                    logger.debug(f"Loaded results from {temp_dir}: {len(current_results)} poses")
                    
                    # Store temp directory references for convergence check
                    if i > 0:
                        check_smina_convergence.prev_temp_dir = temp_directories[i-1]
                        logger.debug(f"Set prev_temp_dir to: {check_smina_convergence.prev_temp_dir}")
                    check_smina_convergence.current_temp_dir = temp_dir
                    logger.debug(f"Set current_temp_dir to: {check_smina_convergence.current_temp_dir}")
                    
                    # Check convergence (skip for first run)
                    if i > 0 and check_smina_convergence(prev_results, current_results, logger, 
                                                        rmsd_threshold=args.convergence_rmsd_threshold, 
                                                        score_threshold=args.convergence_score_threshold):
                        logger.info(f"Convergence achieved at exhaustiveness {exhaustiveness_val}")
                        best_exhaustiveness = exhaustiveness_val
                        break
                    
                    # Update for next iteration
                    prev_results = current_results.copy()
                    best_exhaustiveness = exhaustiveness_val
                    
                    # Ensure we try at least 2 levels
                    if i >= 1 and i < len(exhaustiveness_levels) - 1:
                        continue
                        
                finally:
                    args.outfolder_smina = original_outfolder_smina
                    
            except Exception as run_error:
                logger.error(f"Failed at exhaustiveness level {exhaustiveness_val}: {run_error}")
                # If this was not the first run and we have previous successful results, we can continue with those
                if i > 0 and best_exhaustiveness is not None:
                    logger.info(f"Continuing with previous successful results (exhaustiveness {best_exhaustiveness})")
                    break
                else:
                    # If this was the first run, we have no results to fall back on
                    raise
        
        # Copy results from best exhaustiveness level to main output directory
        best_temp_dir = os.path.join(args.outfolder_smina, f"temp_exh_{best_exhaustiveness}")
        
        logger.info(f"Using results from exhaustiveness level {best_exhaustiveness}")
        
        # Ensure we have a valid best_temp_dir and it exists
        if best_exhaustiveness is None or not os.path.exists(best_temp_dir):
            # Find the most recent successful run
            successful_temp_dirs = [d for d in temp_directories if os.path.exists(d) and os.path.exists(os.path.join(d, 'results.csv'))]
            if successful_temp_dirs:
                # Use the last successful directory
                best_temp_dir = successful_temp_dirs[-1]
                best_exhaustiveness = int(os.path.basename(best_temp_dir).split('_')[-1])
                logger.info(f"No convergence achieved, using results from last successful run: exhaustiveness {best_exhaustiveness}")
            else:
                logger.error("No successful runs found - cannot copy results to main directory")
                return
        
        # Copy all result files from best run to main output directory
        try:
            for filename in os.listdir(best_temp_dir):
                if filename.startswith('temp_exh_'):
                    continue  # Skip other temp directories
                src = os.path.join(best_temp_dir, filename)
                dst = os.path.join(args.outfolder_smina, filename)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                    logger.debug(f"Copied file: {filename}")
                elif os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                    logger.debug(f"Copied directory: {filename}")
            
            logger.info(f"Successfully copied results from {best_temp_dir} to main smina directory")
        except Exception as e:
            logger.error(f"Failed to copy results from temporary directory: {e}")
            # Don't return here - still try to clean up temp directories
        
        logger.info(f'Adaptive exhaustiveness completed. Final exhaustiveness: {best_exhaustiveness}')
        
    except Exception as e:
        logger.error(f"Error during adaptive exhaustiveness: {e}")
        # Try to preserve the most recent successful results before failing
        successful_temp_dirs = [d for d in temp_directories if os.path.exists(d) and os.path.exists(os.path.join(d, 'results.csv'))]
        if successful_temp_dirs:
            try:
                recovery_temp_dir = successful_temp_dirs[-1]
                recovery_exhaustiveness = int(os.path.basename(recovery_temp_dir).split('_')[-1])
                logger.info(f"Attempting to recover results from exhaustiveness {recovery_exhaustiveness}")
                
                for filename in os.listdir(recovery_temp_dir):
                    if filename.startswith('temp_exh_'):
                        continue
                    src = os.path.join(recovery_temp_dir, filename)
                    dst = os.path.join(args.outfolder_smina, filename)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                
                logger.info(f"Successfully recovered results from exhaustiveness {recovery_exhaustiveness}")
            except Exception as recovery_error:
                logger.error(f"Failed to recover results: {recovery_error}")
        
        # Re-raise the original exception
        raise
        
    finally:
        # Clean up temporary directories (but preserve them if debugging is enabled)
        keep_temp_dirs = os.environ.get('SMINA_DEBUG_KEEP_TEMP', '').lower() in ('1', 'true', 'yes', 'on')
        
        if keep_temp_dirs:
            logger.info(f"DEBUG: Keeping Smina temporary directories for debugging:")
            for temp_dir in temp_directories:
                if os.path.exists(temp_dir):
                    logger.info(f"  - {temp_dir}")
            logger.info("DEBUG: To disable this behavior, unset SMINA_DEBUG_KEEP_TEMP environment variable")
        else:
            # Clean up temporary directories
            for temp_dir in temp_directories:
                if os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                        logger.debug(f"Removed temporary directory: {temp_dir}")
                    except Exception as e:
                        logger.warning(f"Could not remove temporary directory {temp_dir}: {e}")
        
        # Clean up convergence check attributes
        if hasattr(check_smina_convergence, 'prev_temp_dir'):
            delattr(check_smina_convergence, 'prev_temp_dir')
        if hasattr(check_smina_convergence, 'current_temp_dir'):
            delattr(check_smina_convergence, 'current_temp_dir')

def run_gnina_single(args, logger):
    """Run Gnina docking"""
    logger.info('Running gnina...')
    
    # Enhanced debugging - log the state of all critical variables
    logger.info(f"=== GNINA DEBUG INFO ===")
    logger.info(f"gnina_path: {args.gnina_path}")
    logger.info(f"receptor_pdb: {args.receptor_pdb}")
    logger.info(f"ligand_sdf: {args.ligand_sdf}")
    logger.info(f"pocket_center: {args.pocket_center} (type: {type(args.pocket_center)})")
    logger.info(f"pocket_size: {args.pocket_size} (type: {type(args.pocket_size)})")
    logger.info(f"output_dir: {args.outfolder_gnina}")
    logger.info(f"num_modes: {args.num_modes}")
    logger.info(f"exhaustiveness: {args.exhaustiveness}")
    logger.info(f"num_threads: {args.num_threads}")
    
    # Build the command
    try:
        # Convert coordinates to strings with explicit error handling
        center_x_str = str(args.pocket_center[0])
        center_y_str = str(args.pocket_center[1])
        center_z_str = str(args.pocket_center[2])
        size_x_str = str(args.pocket_size[0])
        size_y_str = str(args.pocket_size[1])
        size_z_str = str(args.pocket_size[2])
        
        logger.info(f"Converted coordinates - center: [{center_x_str}, {center_y_str}, {center_z_str}], size: [{size_x_str}, {size_y_str}, {size_z_str}]")
        
        # Gnina uses receptor PDB (not PDBQT) and ligand SDF
        cmd = (f"{args.gnina_path} -r {args.receptor_pdb} -l {args.ligand_sdf} "
               f"--center_x {center_x_str} --center_y {center_y_str} --center_z {center_z_str} "
               f"--size_x {size_x_str} --size_y {size_y_str} --size_z {size_z_str} "
               f"-o {os.path.join(args.outfolder_gnina, 'out.sdf')} "
               f"--log {os.path.join(args.outfolder_gnina, 'out.log')} "
               f"--cpu {args.num_threads}")
        
        # Add num_modes if specified
        if hasattr(args, 'num_modes') and args.num_modes:
            cmd += f" --num_modes {args.num_modes}"
        
        # Add exhaustiveness if specified
        if hasattr(args, 'exhaustiveness') and args.exhaustiveness:
            cmd += f" --exhaustiveness {args.exhaustiveness}"
        
        logger.info(f"Constructed command: {cmd}")
        
    except Exception as e:
        logger.error(f"Error constructing gnina command: {e}")
        logger.error(f"pocket_center content: {args.pocket_center}")
        logger.error(f"pocket_size content: {args.pocket_size}")
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Exception args: {e.args}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise
    
    logger.info(f"=== END GNINA DEBUG INFO ===")
    
    # Execute the command and capture any errors
    try:
        result = subprocess.call(cmd, shell=True)
        if result != 0:
            logger.warning(f"Gnina command returned non-zero exit code: {result}")
        logger.info('Running gnina... Done.')
    except Exception as exec_error:
        logger.error(f"Error executing gnina command: {exec_error}")
        logger.error(f"Command that failed: {cmd}")
        raise

def run_gnina(args, logger):
    """Run Gnina docking"""
    # Run Gnina docking
    run_gnina_single(args, logger)
    
    # Split docked poses
    split_mol(args, logger, tool="gnina")
    
    # Make complex
    make_complex(args, logger, tool="gnina")
    
    # Parse gnina output
    parse_gnina(args, logger)
    
def run_ledock(args, logger):
    logger.info('Running LeDock...')

    # Create a temporary directory with the shortest possible path
    # Try different locations in order of preference (shortest path first)
    temp_dir_path = None
    max_attempts = 100  # Maximum attempts to find a unique directory name
    
    # Try different root locations for the shortest possible path
    possible_roots = ["/tmp", "/var/tmp", "/", "/home", os.getcwd()]
    
    for root in possible_roots:
        for attempt in range(max_attempts):
            # Create a more unique temporary directory name
            # Include process ID, timestamp, and random UUID
            timestamp = int(datetime.now().timestamp() * 1000000)  # microseconds
            process_id = os.getpid()
            random_suffix = uuid.uuid4().hex[:8]
            temp_dir_name = f"ld_{process_id}_{timestamp}_{random_suffix}"
            
            candidate_path = os.path.join(root, temp_dir_name)
            
            # Check if directory already exists
            if os.path.exists(candidate_path):
                continue  # Try next name
            
            try:
                # Try to create the directory
                os.makedirs(candidate_path, exist_ok=False)  # Don't allow existing
                
                # Test if we can write to it
                test_file = os.path.join(candidate_path, "test")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                
                temp_dir_path = candidate_path
                logger.info(f'LeDock: Using temporary directory: {temp_dir_path}')
                break
                
            except (OSError, PermissionError, FileExistsError):
                # If we can't use this location or it already exists, try next name
                # DO NOT remove existing directories - they might be in use by parallel processes
                continue
        
        # If we found a working directory, break out of the root loop
        if temp_dir_path is not None:
            break
    
    if temp_dir_path is None:
        raise RuntimeError(f"Could not create a unique temporary directory for LeDock after {max_attempts} attempts in any accessible location")
    
    # Store the original working directory
    original_cwd = os.getcwd()
    
    try:
        # Copy receptor PDB (pro.pdb from lepro) to temporary directory with short name
        temp_receptor_name = "receptor.pdb"
        temp_receptor_path = os.path.join(temp_dir_path, temp_receptor_name)
        shutil.copy2(args.lepro_pdb, temp_receptor_path)
        logger.debug(f'Copied receptor PDB to temporary directory: {temp_receptor_path}')
        
        # Copy ligand MOL2 file to temporary directory with short name
        temp_ligand_name = "ligand.mol2"
        temp_ligand_path = os.path.join(temp_dir_path, temp_ligand_name)
        shutil.copy2(args.ligand_mol2, temp_ligand_path)
        logger.debug(f'Copied ligand MOL2 to temporary directory: {temp_ligand_path}')
        
        # Create ligand.txt file in temporary directory
        temp_ligand_txt = os.path.join(temp_dir_path, 'ligand.txt')
        with open(temp_ligand_txt, "w") as f:
            f.write(temp_ligand_path)
        
        # Create dock.in file in temporary directory with short paths
        dock_in = f"""Receptor
{temp_receptor_name}

RMSD
1.0

Binding pocket
{args.min_coords[0]} {args.max_coords[0]} 
{args.min_coords[1]} {args.max_coords[1]}
{args.min_coords[2]} {args.max_coords[2]}

Number of binding poses
20

Ligands list
ligand.txt

END
"""
        
        temp_dock_in = os.path.join(temp_dir_path, 'dock.in')
        with open(temp_dock_in, 'w') as dock_in_f:
            dock_in_f.write(dock_in)
        
        # Change to the temporary directory before running LeDock
        os.chdir(temp_dir_path)
        
        # Run LeDock with the local dock.in file (shortest possible path)
        subprocess.call(f"{args.ledock_path} dock.in", shell=True)
        
        # Find all .dok files in the temporary directory and move them to ledock output folder
        dok_files = glob.glob(os.path.join(temp_dir_path, '*.dok'))
        
        if not dok_files:
            raise RuntimeError(f"LeDock failed to generate any .dok files in temporary directory: {temp_dir_path}")
        
        # Ensure the destination directory exists
        ledock_output_dir = os.path.abspath(args.outfolder_ledock)
        os.makedirs(ledock_output_dir, exist_ok=True)
        
        # Move all .dok files to the ledock output directory
        for dok_file in dok_files:
            dok_filename = os.path.basename(dok_file)
            # The main output file should be renamed to out.dok
            if dok_filename != 'dock.in':  # Skip non-dok files
                if len(dok_files) == 1 or 'out' in dok_filename.lower():
                    final_dok_path = os.path.join(ledock_output_dir, 'out.dok')
                else:
                    final_dok_path = os.path.join(ledock_output_dir, dok_filename)
                shutil.move(dok_file, final_dok_path)
                logger.debug(f'Successfully moved {dok_file} to {final_dok_path}')
        
        logger.info('Running LeDock... Done.')

        # Split docked poses
        split_mol(args, logger, tool="ledock")

        # Make complex
        make_complex(args, logger, tool="ledock")

        # Parse ledock output
        parse_ledock(args, logger)
        
    finally:
        # Always restore the original working directory first
        os.chdir(original_cwd)
        
        # Check if we should keep the temporary directory for debugging
        keep_temp_dir = os.environ.get('LEDOCK_DEBUG_KEEP_TEMP', '').lower() in ('1', 'true', 'yes', 'on')
        
        if keep_temp_dir:
            logger.info(f"DEBUG: Keeping LeDock temporary directory for debugging: {temp_dir_path}")
            logger.info(f"DEBUG: You can manually run LeDock by going to: {temp_dir_path}")
            logger.info(f"DEBUG: Command to run: {args.ledock_path} dock.in")
            logger.info(f"DEBUG: To disable this behavior, unset LEDOCK_DEBUG_KEEP_TEMP environment variable")
        else:
            # Clean up the temporary directory - this is critical for parallel runs
            if temp_dir_path and os.path.exists(temp_dir_path):
                try:
                    shutil.rmtree(temp_dir_path)
                    logger.debug(f"Successfully removed LeDock temporary directory: {temp_dir_path}")
                except Exception as e:
                    logger.warning(f"Could not remove LeDock temporary directory {temp_dir_path}: {e}")
                    # Try to remove it again after a short delay
                    try:
                        time.sleep(0.1)
                        shutil.rmtree(temp_dir_path)
                        logger.debug(f"Successfully removed LeDock temporary directory on second attempt: {temp_dir_path}")
                    except Exception as e2:
                        logger.error(f"Failed to remove LeDock temporary directory even on second attempt {temp_dir_path}: {e2}")

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

    # Parse gold output
    parse_gold(args, logger)

    logger.info('Running gold... Done.')

def consensus_dock(args, logger):
    # Convert receptor pdb to pdbqt format or use provided pdbqt
    logger.info('Starting consensus_dock...')

    if getattr(args, 'only_rmsd', False):
        logger.info("ONLY_RMSD mode enabled: skipping docking and conversions")
        calculate_rmsd(args, logger)
        logger.info('########## Finished consensus_docker.py #########')
        return
    
    # Determine which conversions are actually needed based on selected tools
    needs_pdbqt = args.use_smina  # Only Smina requires PDBQT
    needs_ligand_mol2 = args.use_ledock  # Only LeDock requires ligand MOL2
    needs_receptor_mol2 = False  # GalaxyDock3 is disabled, so never needed for now
    
    logger.info(f"File conversion requirements based on selected tools:")
    logger.info(f"  - PDBQT conversion needed: {needs_pdbqt} (Smina selected: {args.use_smina})")
    logger.info(f"  - Ligand MOL2 conversion needed: {needs_ligand_mol2} (LeDock selected: {args.use_ledock})")
    logger.info(f"  - Receptor MOL2 conversion needed: {needs_receptor_mol2}")
    
    # Handle receptor PDBQT - initialize from args or None
    original_receptor_pdbqt = getattr(args, 'receptor_pdbqt', None)
    args.receptor_pdbqt = None  # Reset, will set if needed
    
    if needs_pdbqt:
        if original_receptor_pdbqt:
            # Use the provided PDBQT file
            logger.info('Using provided receptor PDBQT file...')
            # Ensure input directory exists before copying
            os.makedirs(args.outfolder_input, exist_ok=True)
            # Copy the provided PDBQT file to the input folder for consistency
            receptor_pdbqt_filename = os.path.basename(original_receptor_pdbqt)
            args.receptor_pdbqt_final = os.path.join(args.outfolder_input, receptor_pdbqt_filename)
            shutil.copy2(original_receptor_pdbqt, args.receptor_pdbqt_final)
            args.receptor_pdbqt = args.receptor_pdbqt_final
            logger.info('Using provided receptor PDBQT file... Done.')
            
            # Check if receptor_pdb is also provided (needed for some tools)
            if not args.receptor_pdb:
                logger.warning('receptor_pdb not provided. Some tools (LeDock, GOLD, Gnina) require PDB format and will be skipped.')
        else:
            # Convert receptor pdb to pdbqt format (unless skipped)
            if not args.receptor_pdb:
                raise ValueError("Either --receptor_pdb or --receptor_pdbqt must be provided for Smina")
            
            if args.skip_pdb_to_pdbqt:
                logger.info('Skipping PDB to PDBQT conversion (--skip_pdb_to_pdbqt enabled)')
                logger.warning('Smina requires PDBQT file. Smina docking may fail without it.')
            else:
                # Ensure input directory exists before conversion
                os.makedirs(args.outfolder_input, exist_ok=True)
                args.receptor_pdbqt = os.path.join(args.outfolder_input, 'receptor.pdbqt')
                pdb_to_pdbqt(args.receptor_pdb, args.receptor_pdbqt, logger, pH=args.pH)
    else:
        logger.info('Skipping PDBQT conversion (not needed for selected tools)')

    # Get stem from file name of args.ligand_sdf
    sdf_path = Path(args.ligand_sdf)
    sdf_stem = sdf_path.stem

    # Convert ligand sdf to mol2 only if needed
    args.ligand_mol2 = None  # Initialize to None
    if needs_ligand_mol2:
        # Ensure input directory exists before conversion
        os.makedirs(args.outfolder_input, exist_ok=True)
        args.ligand_mol2 = os.path.join(args.outfolder_input, sdf_stem + '.mol2')
        to_mol2(args.ligand_sdf, 'LIG', args.ligand_mol2, logger)
    else:
        logger.info('Skipping ligand MOL2 conversion (not needed for selected tools)')

    # Convert protein pdb to protein mol2 only if needed
    args.receptor_mol2 = None  # Initialize to None
    if needs_receptor_mol2:
        if args.receptor_pdb:
            args.receptor_mol2 = os.path.join(args.outfolder_input, 'receptor.mol2')
            to_mol2(args.receptor_pdb, 'PRO', args.receptor_mol2, logger)
        else:
            logger.warning('Receptor MOL2 conversion needed but receptor_pdb not provided')
    else:
        logger.info('Skipping receptor MOL2 conversion (not needed for selected tools)')

    # Get pocket coordinates
    pocket_center, pocket_size, min_coords, max_coords = get_pocket_coords(args, logger)

    # Add pocket coordinates and ligands chunks to args
    args.pocket_center = pocket_center
    args.pocket_size = pocket_size
    args.min_coords = min_coords
    args.max_coords = max_coords
    
    # Log the assignment for debugging
    logger.info(f"Assigned pocket coordinates to args:")
    logger.info(f"  args.pocket_center: {args.pocket_center} (type: {type(args.pocket_center)})")
    logger.info(f"  args.pocket_size: {args.pocket_size} (type: {type(args.pocket_size)})")
    logger.info(f"  args.min_coords: {args.min_coords} (type: {type(args.min_coords)})")
    logger.info(f"  args.max_coords: {args.max_coords} (type: {type(args.max_coords)})")

    # Run selected docking programs
    tools_run = []
    
    # Run smina docking
    if args.use_smina:
        if args.receptor_pdbqt is None:
            logger.error("Cannot run Smina: receptor PDBQT file is not available")
        else:
            try:
                run_smina(args, logger)
                tools_run.append('smina')
            except Exception as e:
                logger.error(f"smina docking failed: {e}")

    # Run gnina docking
    if args.use_gnina:
        if not args.receptor_pdb:
            logger.error("Cannot run Gnina: receptor PDB file is not available")
        else:
            try:
                run_gnina(args, logger)
                tools_run.append('gnina')
            except Exception as e:
                logger.error(f"gnina docking failed: {e}")

    # Run LeDock docking (only if PDB is available and selected)
    if args.use_ledock:
        if not args.receptor_pdb:
            logger.warning('Skipping LeDock docking as it requires receptor PDB file')
        elif not args.ligand_mol2:
            logger.error('Cannot run LeDock: ligand MOL2 file is not available')
        else:
            try:
                args.lepro_pdb = lepro(args, logger)
                run_ledock(args, logger)
                tools_run.append('ledock')
            except Exception as e:
                logger.error(f"LeDock docking failed: {e}")

    # Run GalaxyDock3 docking (only if PDB is available)
    #if args.use_gd3 and args.receptor_pdb:
    #    try:
    #        run_gd3(args, logger)
    #        tools_run.append('gd3')
    #    except Exception as e:
    #        logger.error(f"GalaxyDock3 docking failed: {e}")

    # Run gold docking (only if PDB is available and selected)
    if args.use_gold:
        if not args.receptor_pdb:
            logger.warning('Skipping GOLD docking as it requires receptor PDB file')
        else:
            try:
                run_gold(args, logger)
                tools_run.append('gold')
            except Exception as e:
                logger.error(f"GOLD docking failed: {e}")

    # Calculate RMSD if multiple tools were run in this session OR if existing results exist
    tools_with_results = []
    
    # Check for results from current run
    if 'smina' in tools_run:
        tools_with_results.append('smina')
    if 'ledock' in tools_run:
        tools_with_results.append('ledock')
    if 'gold' in tools_run:
        tools_with_results.append('gold')
    if 'gnina' in tools_run:
        tools_with_results.append('gnina')
    
    # Check for existing results from previous runs
    if hasattr(args, 'has_existing_results') and args.has_existing_results:
        if os.path.exists(os.path.join(args.outfolder, 'smina', 'results.csv')) and 'smina' not in tools_with_results:
            tools_with_results.append('smina')
        if os.path.exists(os.path.join(args.outfolder, 'ledock', 'results.csv')) and 'ledock' not in tools_with_results:
            tools_with_results.append('ledock')
        if os.path.exists(os.path.join(args.outfolder, 'gold', 'results.csv')) and 'gold' not in tools_with_results:
            tools_with_results.append('gold')
        if os.path.exists(os.path.join(args.outfolder, 'gnina', 'results.csv')) and 'gnina' not in tools_with_results:
            tools_with_results.append('gnina')
    logger.info(f"Tools with results: {tools_with_results}")

    # Calculate RMSD if we have results from multiple tools and user hasn't disabled it
    if args.skip_rmsd:
        logger.info("Skipping RMSD calculation (--skip_rmsd flag enabled)")
        logger.info("Individual tool results are available in their respective folders:")
        for tool in tools_with_results:
            tool_folder = os.path.join(args.outfolder, tool)
            results_file = os.path.join(tool_folder, 'results.csv')
            if os.path.exists(results_file):
                logger.info(f"  - {tool.capitalize()}: {results_file}")
    elif len(tools_with_results) > 1:
        try:
            calculate_rmsd(args, logger)
            logger.info(f"RMSD calculation performed for tools: {', '.join(tools_with_results)}")
        except Exception as e:
            logger.error(f"RMSD calculation failed: {e}")
    else:
        logger.info(f"Skipping RMSD calculation - only {len(tools_with_results)} tool(s) have results: {', '.join(tools_with_results) if tools_with_results else 'none'}")

    logger.info(f"Docking completed. Tools run in this session: {', '.join(tools_run) if tools_run else 'none'}")
    logger.info(f"Tools with available results: {', '.join(tools_with_results) if tools_with_results else 'none'}")


    logger.info('########## Finished consensus_docker.py #########')

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="CLI tool to perform consensus docking simulations using Smina and LeDock."
    )

    parser.add_argument('--outfolder', type=str, help='Base output directory (default: current directory)')
    parser.add_argument('--smina_path', type=str, default=False, help='Path to smina executable')
    parser.add_argument('--gnina_path', type=str, default=False, help='Path to gnina executable')
    parser.add_argument('--ledock_path', type=str, default=False, help='Path to LeDock executable')
    parser.add_argument('--lepro_path', type=str, default='lepro', help='Path to lepro executable')
    parser.add_argument('--gold_path', type=str, default=False, help='Path to gold executable')
    parser.add_argument('--gd3_path', type=str, help='Path to GalaxyDock3 directory', default=False)
    parser.add_argument('--pH', type=float, default=7.4, help='pH value for adding missing hydrogens (default: 7.4)')
    parser.add_argument('--receptor_pdb', type=str, help='Path to receptor PDB file')
    parser.add_argument('--receptor_pdbqt', type=str, help='Path to receptor PDBQT file (optional, skips PDB to PDBQT conversion)')
    parser.add_argument('--ligand_sdf', type=str, help='Path to ligand SDF file')
    parser.add_argument('--pocket_pdb', type=str, help='Path to pocket PDB file')
    parser.add_argument('--exhaustiveness', type=int, default=12, help='Exhaustiveness value for smina (default: 12)')
    parser.add_argument('--num_modes', type=int, default=20, help='Number of modes for smina and gnina (default: 20)')
    parser.add_argument('--num_threads', type=int, default=1, help='Number of threads for smina and gnina (default: 1)')
    parser.add_argument('--cutoff_value', type=float, default=-7.0, help='SMINA_Score cutoff value for analysis (default: -7.0)')
    parser.add_argument('--use_smina', action='store_true', help='Use smina for docking')
    parser.add_argument('--use_gnina', action='store_true', help='Use gnina for docking')
    parser.add_argument('--use_ledock', action='store_true', help='Use LeDock for docking')
    parser.add_argument('--use_gold', action='store_true', help='Use GOLD for docking')
    parser.add_argument('--only_rmsd', action='store_true', help='Only calculate RMSD between different tools (no docking)')
    parser.add_argument('--adaptive_exhaustiveness', action='store_true', help='Use adaptive exhaustiveness strategy for smina (tries increasing levels starting from 8 until convergence or maximum reached)')
    parser.add_argument('--convergence_rmsd_threshold', type=float, default=1.5, help='RMSD threshold for adaptive exhaustiveness convergence (default: 1.5 Angstrom)')
    parser.add_argument('--convergence_score_threshold', type=float, default=0.1, help='Score difference threshold for adaptive exhaustiveness convergence (default: 0.1 kcal/mol)')
    parser.add_argument('--exhaustiveness_increment', type=int, default=8, help='Increment step for adaptive exhaustiveness levels (default: 8)')
    parser.add_argument('--maximum_exhaustiveness', type=int, default=32, help='Maximum exhaustiveness level for adaptive strategy (default: 32)')
    parser.add_argument('--skip_rmsd', action='store_true', help='Skip final RMSD calculation between different tools (keeps individual tool results only)')
    parser.add_argument('--skip_pdb_to_pdbqt', action='store_true', help='Skip conversion from PDB to PDBQT format (requires --receptor_pdbqt to be provided)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output directory if it exists')
    args = parser.parse_args()

    # If --only_rmsd is provided, explicitly disable docking tools
    if args.only_rmsd:
        args.use_smina = False
        args.use_gnina = False
        args.use_ledock = False
        args.use_gold = False
    
    # Validate receptor input arguments based on selected tools.
    # In --only_rmsd mode, receptor/ligand/pocket inputs are not required.
    if not args.only_rmsd:
        # First, determine what tools will be used
        temp_use_smina = args.use_smina
        temp_use_ledock = args.use_ledock
        temp_use_gold = args.use_gold
        temp_use_gnina = args.use_gnina

        # If no docking programs are specified and only RMSD is not selected, use all available ones (backward compatibility)
        if not temp_use_smina and not temp_use_ledock and not temp_use_gold and not temp_use_gnina:
            temp_use_smina = True
            temp_use_ledock = True
            temp_use_gold = True
            temp_use_gnina = True

        # Determine what receptor formats are actually needed
        needs_pdb = temp_use_ledock or temp_use_gold or temp_use_gnina  # These tools need PDB
        needs_pdbqt = temp_use_smina  # Only Smina needs PDBQT

        if not args.receptor_pdb and not args.receptor_pdbqt:
            parser.error("At least one of --receptor_pdb or --receptor_pdbqt must be provided")

        # Validate that we have the right format for selected tools
        if needs_pdb and not args.receptor_pdb:
            pdb_tools = []
            if temp_use_ledock:
                pdb_tools.append('LeDock')
            if temp_use_gold:
                pdb_tools.append('GOLD')
            if temp_use_gnina:
                pdb_tools.append('Gnina')
            if pdb_tools:
                parser.error(f"--receptor_pdb is required for the following selected tools: {', '.join(pdb_tools)}")

        if needs_pdbqt and args.skip_pdb_to_pdbqt and not args.receptor_pdbqt:
            parser.error("When using --skip_pdb_to_pdbqt with Smina, --receptor_pdbqt must be provided")

    # Now apply the default tool selection
    if not args.use_smina and not args.use_ledock and not args.use_gold and not args.use_gnina and not args.only_rmsd:
        args.use_smina = True
        args.use_ledock = True
        args.use_gold = True
        args.use_gnina = True
    
    # Validate that paths are provided for selected docking programs
    if not args.only_rmsd:
        if args.use_smina and not args.smina_path:
            parser.error("--smina_path must be provided when using smina")
        if args.use_ledock and not args.ledock_path:
            parser.error("--ledock_path must be provided when using LeDock")
        if args.use_gold and not args.gold_path:
            parser.error("--gold_path must be provided when using GOLD")
        if args.use_gnina and not args.gnina_path:
            parser.error("--gnina_path must be provided when using gnina")
    
    # Check if output directory exists
    if args.only_rmsd:
        if not args.outfolder:
            parser.error("--outfolder must be provided when using --only_rmsd")
        if not os.path.exists(args.outfolder):
            parser.error(f"Output directory '{args.outfolder}' does not exist for --only_rmsd")
        args.has_existing_results = True
    else:
        if os.path.exists(args.outfolder):
            if not args.overwrite:
                parser.error(f"Output directory '{args.outfolder}' already exists. Use --overwrite to overwrite it.")
            else:
                # Check for existing results from previous runs
                args.has_existing_results = any([
                    os.path.exists(os.path.join(args.outfolder, 'smina', 'results.csv')),
                    os.path.exists(os.path.join(args.outfolder, 'ledock', 'results.csv')),
                    os.path.exists(os.path.join(args.outfolder, 'gold', 'results.csv')),
                    os.path.exists(os.path.join(args.outfolder, 'gnina', 'results.csv'))
                ])
        else:
            args.has_existing_results = False

        # Create output directory if it doesn't exist
        os.makedirs(args.outfolder, exist_ok=args.overwrite)

    # Determine if input directory is needed (only if any conversions will happen)
    args.outfolder_input = os.path.join(args.outfolder, 'input')
    if not args.only_rmsd:
        needs_pdbqt = args.use_smina and not args.receptor_pdbqt  # Will need to convert PDB to PDBQT
        needs_ligand_mol2 = args.use_ledock  # LeDock needs MOL2
        needs_input_dir = needs_pdbqt or needs_ligand_mol2 or (args.use_smina and args.receptor_pdbqt)
        if needs_input_dir:
            os.makedirs(args.outfolder_input, exist_ok=args.overwrite)

    # Create output directories for selected docking programs
    # Always set directory attributes to avoid attribute errors
    args.outfolder_smina = os.path.join(args.outfolder, 'smina')
    args.outfolder_ledock = os.path.join(args.outfolder, 'ledock')
    args.outfolder_gold = os.path.join(args.outfolder, 'gold')
    args.outfolder_gd3 = os.path.join(args.outfolder, 'gd3')
    args.outfolder_gnina = os.path.join(args.outfolder, 'gnina')
    
    # Create directories only for selected programs
    if not args.only_rmsd:
        if args.use_smina:
            os.makedirs(args.outfolder_smina, exist_ok=args.overwrite)

        if args.use_ledock:
            os.makedirs(args.outfolder_ledock, exist_ok=args.overwrite)

        if args.use_gold:
            os.makedirs(args.outfolder_gold, exist_ok=args.overwrite)

        if args.use_gnina:
            os.makedirs(args.outfolder_gnina, exist_ok=args.overwrite)

    # Note: gd3 directory creation removed as GalaxyDock3 is currently disabled

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