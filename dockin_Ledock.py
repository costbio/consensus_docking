
# -*- coding: utf-8 -*-
"""

@author: onur
"""
#%%
# import libraries
import warnings
from pathlib import Path
import subprocess
import numpy as np
import os
import glob
import multiprocessing
import pandas as pd


# Import nglview as nv
from openbabel import pybel
from opencadd.structure.core import Structure
from opencadd.io.dataframe import DataFrame

# Filter warnings
warnings.filterwarnings("ignore")
ob_log_handler = pybel.ob.OBMessageHandler()
pybel.ob.obErrorLog.SetOutputLevel(0)

#%%

def dockLeDock_single(inputs):
    ledock_bin = inputs[0]
    i = inputs[1]
    subprocess.call(f"{ledock_bin} dock_{i}.in", shell=True)

#%%

# Convert protein to PDBQT format
#receptor_fn = "30ef.pdb"
#lepro("30ef.pdb", os.path.abspath("/home/ssahin/share/apps/lepro_linux_x86"))

# Load the structure for pocket calculations
structure_df = DataFrame.from_file("3oef.pdb")
positions = np.array([structure_df["atom.x"].values, structure_df["atom.y"].values, structure_df["atom.z"].values])
pocket_min = np.min(positions, axis=1)
pocket_max = np.max(positions, axis=1)

print(pocket_min[0], pocket_max[0])
print(pocket_min[1], pocket_max[1])
print(pocket_min[2], pocket_max[2])

out_folder = "test"

# Get list of ligands (mol2 files) directly from the output folder
ligs_list = glob.glob(f"{out_folder}/*.mol2")
print(ligs_list)

num_threads = 60 
ligs_chunks = np.array_split(ligs_list, num_threads)
print(ligs_chunks)

for i in range(len(ligs_chunks)):
    chunk = ligs_chunks[i]
    chunk_text = "\n".join(chunk) + "\n"
    with open(f'ligands_{i}.txt', "w") as f:
        f.write(chunk_text)

    dock_in = f"""
    Receptor
    pro.pdb

    RMSD
    1.0

    Binding pocket
    {pocket_min[0]} {pocket_max[0]} 
    {pocket_min[1]} {pocket_max[1]} 
    {pocket_min[2]} {pocket_max[2]} 

    Number of binding poses
    20

    Ligands list
    ligands_{i}.txt

    END
    """

    with open(f'dock_{i}.in', 'w') as dock_in_f:
        dock_in_f.write(dock_in.strip() + "\n")

ledock_bin = os.path.abspath("/home/ssahin/share/apps/ledock")
ledock_bins = [ledock_bin] * num_threads
print('buraya kadar geldim.')

pool = multiprocessing.Pool(num_threads)
pool.map(dockLeDock_single, zip(ledock_bins, np.arange(0, num_threads, 1)))
pool.close()
pool.join()
