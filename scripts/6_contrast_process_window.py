import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.simulation.litho_process import LithoSim
from src.simulation.resist import remaining_resist_fraction

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

def contrast_curve():
    sim = LithoSim()
    mask = sim.create_mask()
    aerial = sim.compute_aerial_image(mask)

    dose_sweep = np.geomspace(5, 100, 30)
    remain = np.array([remaining_resist_fraction(aerial, d).mean() for d in dose_sweep])

    plt.figure(figsize=(6,4))
    plt.plot(np.log10(dose_sweep), remain, "o-")
    plt.xlabel("log10(Dose [mJ/cm^2])")
    plt.ylabel("Mean remaining resist fraction")
    plt.title("Resist contrast curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "contrast_curve.png", dpi=160)
    plt.close()
    print(f"Saved: {OUT / 'contrast_curve.png'}")

def process_window_toy():
    sim = LithoSim()
    mask = sim.create_mask()
    aerial_nom = sim.compute_aerial_image(mask)

    focus_vals = np.linspace(-0.15, 0.15, 7)
    dose_vals  = np.linspace(15, 30, 7)

    cd_map = np.zeros((len(focus_vals), len(dose_vals)), dtype=float)

    for i, f in enumerate(tqdm(focus_vals, desc="Process window")):
        alpha = 1.0 / (1.0 + abs(f)/0.2)
        aerial_f = aerial_nom ** alpha

        for j, dose in enumerate(dose_vals):
            rem = remaining_resist_fraction(aerial_f, dose)
            open_area = (rem < 0.5).astype(np.float32)
            band = open_area[:, open_area.shape[1]//2 - 6 : open_area.shape[1]//2 + 6]
            cd_map[i, j] = band.mean()

    plt.figure(figsize=(6,5))
    plt.imshow(cd_map, aspect="auto", origin="lower")
    plt.xticks(range(len(dose_vals)), [f"{d:.1f}" for d in dose_vals])
    plt.yticks(range(len(focus_vals)), [f"{f:.3f}" for f in focus_vals])
    plt.colorbar(label="CD proxy")
    plt.xlabel("Dose [mJ/cm^2]")
    plt.ylabel("Focus [um]")
    plt.title("Focus-Exposure process window")
    plt.tight_layout()
    plt.savefig(OUT / "process_window.png", dpi=160)
    plt.close()
    print(f"Saved: {OUT / 'process_window.png'}")

def main():
    contrast_curve()
    process_window_toy()

if __name__ == "__main__":
    main()
