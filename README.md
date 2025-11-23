# Consensus Docking

A consensus molecular docking pipeline supporting multiple docking tools: **Smina**, **Gnina**, **LeDock**, and **GOLD**.

## Features

- **Multi-tool docking**: Run one or more docking tools and compare results
- **Adaptive exhaustiveness**: For Smina, automatically find optimal exhaustiveness level
- **RMSD analysis**: Compare poses across different docking tools
- **Automated workflow**: Handles file conversions, pose extraction, and result parsing

## Supported Docking Tools

| Tool | Input Format 
|------|-------------
| **Smina** | Receptor: PDBQT, Ligand: SDF 
| **Gnina** | Receptor: PDB, Ligand: SDF
| **LeDock** | Receptor: PDB, Ligand: SDF
| **GOLD** | Receptor: PDB, Ligand: SDF

## Installation

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate consensus_docking
```

**Optional**: If you need opencadd (for some pocket coordinate calculations):
```bash
pip install git+https://github.com/volkamerlab/opencadd.git
```

### 2. Install Docking Tools

You must install the docking tools separately:
- **Smina**: Download from https://sourceforge.net/projects/smina/
- **Gnina**: Download from https://github.com/gnina/gnina/releases
- **LeDock**: Download from http://www.lephar.com/software.htm
- **GOLD**: Requires license from CCDC

## Usage

### Basic Example

```bash
python consensus_docker.py \
  --receptor_pdb protein.pdb \
  --ligand_sdf ligand.sdf \
  --pocket_pdb pocket.pdb \
  --smina_path /path/to/smina \
  --gnina_path /path/to/gnina \
  --use_smina \
  --use_gnina \
  --outfolder results/
```

### Key Arguments

- `--receptor_pdb`: Protein structure (PDB format)
- `--receptor_pdbqt`: Protein structure (PDBQT format, optional)
- `--ligand_sdf`: Ligand structure (SDF format)
- `--pocket_pdb`: Binding pocket definition (PDB format)
- `--outfolder`: Output directory

### Tool Selection

- `--use_smina`: Enable Smina docking
- `--use_gnina`: Enable Gnina docking
- `--use_ledock`: Enable LeDock docking
- `--use_gold`: Enable GOLD docking

### Smina-Specific Options

- `--exhaustiveness 12`: Search exhaustiveness (default: 12)
- `--adaptive_exhaustiveness`: Auto-find optimal exhaustiveness level
- `--num_modes 20`: Number of binding modes (default: 20)
- `--num_threads 4`: CPU threads for docking

### Advanced Options

```bash
# Adaptive exhaustiveness for Smina
python consensus_docker.py \
  --receptor_pdb protein.pdb \
  --ligand_sdf ligand.sdf \
  --pocket_pdb pocket.pdb \
  --smina_path ./smina \
  --use_smina \
  --adaptive_exhaustiveness \
  --maximum_exhaustiveness 32 \
  --exhaustiveness_increment 8 \
  --convergence_rmsd_threshold 1.5 \
  --outfolder results/

# Skip RMSD calculation between tools
python consensus_docker.py \
  --receptor_pdb protein.pdb \
  --ligand_sdf ligand.sdf \
  --pocket_pdb pocket.pdb \
  --smina_path ./smina \
  --gnina_path ./gnina \
  --use_smina \
  --use_gnina \
  --skip_rmsd \
  --outfolder results/
```

## Output Structure

```
results/
├── input/
│   ├── receptor.pdbqt
│   ├── receptor.mol2
│   └── ligand.mol2
├── smina/
│   ├── out.sdf              # All poses
│   ├── out.log              # Docking log
│   ├── out_1.sdf, ...       # Individual poses
│   ├── complex_1.pdb, ...   # Receptor-ligand complexes
│   └── results.csv          # Scores
├── gnina/
│   ├── out.sdf
│   ├── out.log
│   ├── out_1.sdf, ...
│   ├── complex_1.pdb, ...
│   └── results.csv          # Affinity, CNN pose score, CNN affinity
├── ledock/
│   └── ...
├── gold/
│   └── ...
└── log_*.log
```

## Output Scores

### Smina
- `SMINA_Score`: Binding affinity (kcal/mol)

### Gnina
- `GNINA_Affinity`: Binding affinity (kcal/mol)
- `GNINA_CNN_pose`: CNN pose score
- `GNINA_CNN_affinity`: CNN affinity prediction

### LeDock
- `LeDock_Score`: Binding score

### GOLD
- `Gold.PLP.Fitness`: PLP fitness score

## Notes

- **Gnina** does not support adaptive exhaustiveness
- **LeDock** and **GOLD** require receptor in PDB format
- **Smina** requires receptor in PDBQT format (auto-converted if PDB provided)
- RMSD calculation requires results from multiple tools
