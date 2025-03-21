import subprocess
import os
import sys
import logging
from datetime import datetime

# GalaxyDock3 için logging ayarları
def setup_logging(log_file):
    logger = logging.getLogger("galaxydock")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# GalaxyDock3 fonksiyonu ile paralel optimize edilmiş bir çalışma
def run_GalaxyDock3_opt(opt, logger):
    logger.info("Running GalaxyDock3 optimization...")
    
    HOME = os.path.abspath(opt.home_dir)
    pdb_fn = opt.pdb_fn
    lig_fn = opt.lig_fn
    EXEC_GALAXYDOCK3 = f'{HOME}/bin/GalaxyDock3'

    # Optimize edilmiş grid boyutlarını belirliyoruz
    grid_size = opt.grid_size if opt.grid_size else "0.5"  # Default grid size 0.5 Å
    search_method = opt.search_method if opt.search_method else "MonteCarlo"  # Default search method
    
    # Input dosyasını özelleştiriyoruz
    INPUT = file(f'{HOME}/input/galaxydock.in').read()
    input = INPUT.replace('[GALAXY_DOCK_HOME]', HOME)
    input = input.replace('[RECEPTOR_PDB]', pdb_fn)
    input = input.replace('[LIGAND_MOL2]', lig_fn)
    input = input.replace('[GRID_BOX_CENTER]', ' '.join([opt.cntr_x, opt.cntr_y, opt.cntr_z]))
    input = input.replace('[GRID_SIZE]', grid_size)  # Grid boyutunu ekliyoruz
    input = input.replace('[SEARCH_METHOD]', search_method)  # Search methodu belirliyoruz

    fout = file('galaxydock.in', 'wt')
    fout.write(input)
    fout.close()

    # Çalışmayı başlatıyoruz
    try:
        subprocess.run([EXEC_GALAXYDOCK3, "galaxydock.in"], check=True)
        logger.info("GalaxyDock3 optimization completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error occurred during docking: {e}")
        sys.exit(1)

# Ana fonksiyon
def main():
    parser = argparse.ArgumentParser(description="Optimized GalaxyDock3 script")
    parser.add_argument('--home_dir', type=str, required=True, help='GalaxyDock3 directory path')
    parser.add_argument('--pdb_fn', type=str, required=True, help='Receptor PDB file')
    parser.add_argument('--lig_fn', type=str, required=True, help='Ligand MOL2 file')
    parser.add_argument('--cntr_x', type=str, required=True, help='X coordinate for docking')
    parser.add_argument('--cntr_y', type=str, required=True, help='Y coordinate for docking')
    parser.add_argument('--cntr_z', type=str, required=True, help='Z coordinate for docking')
    parser.add_argument('--grid_size', type=str, help='Grid size for docking (default 0.5)')
    parser.add_argument('--search_method', type=str, help='Search method for docking (MonteCarlo, GridSearch)')
    parser.add_argument('--outfolder', type=str, required=True, help='Output folder for logs')
    args = parser.parse_args()

    # Log dosyasını ayarlıyoruz
    log_file = os.path.join(args.outfolder, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logging(log_file)

    # GalaxyDock3 scriptini çalıştırıyoruz
    run_GalaxyDock3_opt(args, logger)

if __name__ == "__main__":
    main()
