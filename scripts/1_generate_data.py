"""Generate synthetic dataset for training and testing."""
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.io import imsave
import sys
sys.path.append('.')

from src.simulation.litho_process import LithoSim
from src.simulation.defects import add_particle_defect, add_line_roughness


def main():
    print("Initializing lithography simulator...")
    sim = LithoSim()
    np.random.seed(42)

    clean_dir = Path("data/raw_synthetic/clean")
    defect_dir = Path("data/raw_synthetic/defect")
    clean_dir.mkdir(parents=True, exist_ok=True)
    defect_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating CLEAN images...")
    for i in tqdm(range(200)):
        aerial = sim.compute_aerial_image()
        img_8bit = (aerial * 255).astype(np.uint8)
        imsave(clean_dir / f"{i:04d}.png", img_8bit, check_contrast=False)

    print("\nGenerating DEFECTIVE images...")
    for i in tqdm(range(200)):
        aerial = sim.compute_aerial_image()
        if i % 2 == 0:
            aerial = add_particle_defect(aerial, particle_radius_px=6)
        else:
            aerial = add_line_roughness(aerial, roughness_amplitude=0.1)
        img_8bit = (aerial * 255).astype(np.uint8)
        imsave(defect_dir / f"{i:04d}.png", img_8bit, check_contrast=False)

    print(f"\n✓ Generated 400 images total")


if __name__ == "__main__":
    main()

