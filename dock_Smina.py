
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:05:55 2023

@author: oykum
"""
#%%
# import libraries
import warnings
from pathlib import Path
import subprocess
import numpy as np

#import nglview as nv
from openbabel import pybel

from opencadd.structure.core import Structure
from opencadd.io.dataframe import DataFrame

# filter warnings
warnings.filterwarnings("ignore")
ob_log_handler = pybel.ob.OBMessageHandler()
pybel.ob.obErrorLog.SetOutputLevel(0)

#%%
def pdb_to_pdbqt(pdb_path, pdbqt_path, pH=7.4):
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
    molecule = list(pybel.readfile("pdb", str(pdb_path)))[0]
    # add hydrogens at given pH
    molecule.OBMol.CorrectForPH(pH)
    molecule.addh()
    # add partial charges to each atom
    for atom in molecule.atoms:
        atom.OBAtom.GetPartialCharge()
    molecule.write("pdbqt", str(pdbqt_path), overwrite=True)
    return

#%%
# convert protein to PDBQT format
receptor_fn = "3vd2_A" #change to your protein
pdb_to_pdbqt(receptor_fn+".pdb", receptor_fn+".pdbqt")

#%%
structure_df = DataFrame.from_file("O15350_3vd2_A.pdb_res_P_1.pdb")
positions = np.array([structure_df["atom.x"].values,structure_df["atom.y"].values,structure_df["atom.z"].values])
pocket_center = list(map(str,((np.max(positions,axis=1) + np.min(positions,axis=1)) / 2)))
pocket_size = list(map(str,((np.max(positions,axis=1) - np.min(positions,axis=1)) + 5)))

subprocess.call(f"smina -r {receptor_fn}.pdbqt -l 5,9-dimethyl.sdf \
    --center_x {pocket_center[0]} --center_y {pocket_center[1]} --center_z {pocket_center[2]} \
    --size_x {pocket_size[0]} --size_y {pocket_size[1]} --size_z {pocket_size[2]} --out screen_3vd2_5,9-dimethyl.sdf \
    --num_modes 12 --exhaustiveness 12 --cpu 12 --log screen_3vd2_5,9-dimethyl.log",shell=True)

