#!/usr/bin/env python
"""
1. Dose sweep  → contrast curve  (remaining resist vs log₁₀ dose)
2. Focus + dose sweep → process window  (CD vs focus/exposure)
3. Saves ROC curve for ML detector
"""
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from itertools import product
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

# --- project imports -------------------------------------------------
from src.simulation.litho_process import LithoSim
from src.simulation.resist import remaining_resist_fraction
from src.detection.autoencoder_simple import SimpleAutoencoder
from src.detection.dataset import WaferDataset
import torch
# --------------------------------------------------------------------

OUT = Path("results/figures"); OUT.mkdir(parents=True, exist_ok=True)
sns.set_style("whitegrid")

# 1) Contrast curve ---------------------------------------------------
sim = LithoSim()
mask = sim.create_mask()
aerial = sim.compute_aerial_image(mask)

dose_sweep = np.geomspace(5, 100, 30)      # 5–100 mJ/cm²
remain = [remaining_resist_fraction(aerial, d).mean() for d in dose_sweep]

# contrast γ = (log D100 – log D0)⁻¹
D0   = dose_sweep[np.argmin(np.abs(np.array(remain) - 1.0))]   # ~no removal
D100 = dose_sweep[np.argmin(np.abs(np.array(remain) - 0.0))]   # full clear
gamma = 1 / (np.log10(D100) - np.log10(D0))

plt.figure(figsize=(6,4))
plt.plot(np.log10(dose_sweep), remain, 'o-')
plt.xlabel("log₁₀(Dose [mJ/cm²])"); plt.ylabel("Remaining Resist Fraction")
plt.title(f"Resist Contrast Curve  (γ ≈ {gamma:.2f})")
plt.tight_layout(); plt.savefig(OUT/"contrast_curve.png", dpi=160)
print(f"✓ Contrast curve saved  (γ≈{gamma:.2f})")

# 2) Focus–Exposure process window -----------------------------------
focus_vals = np.linspace(-0.15, 0.15, 7)      # ±0.15 µm defocus
dose_vals  = np.linspace(15, 30, 7)           # mJ/cm²
target_cd  = sim.cfg["pitch_px"] * sim.cfg["duty_cycle"] * sim.cfg["pixel_nm"]  # nm

cd_map = np.zeros((len(focus_vals), len(dose_vals)))
for i,f in enumerate(tqdm(focus_vals, desc="Process window")):
    for j,dose in enumerate(dose_vals):
        # crude focus model: PSF blur ∝ (1 + |focus|/DOF)
        oversize = 1 + abs(f)/0.2
        aerial_f = sim.compute_aerial_image(mask) ** (1/oversize)
        resist   = remaining_resist_fraction(aerial_f, dose) < 0.5
        # line width in nm (average over centre rows)
        cd_nm = resist[:, resist.shape[1]//2-5:resist.shape[1]//2+5].mean()*sim.cfg["pixel_nm"]*resist.shape[1]
        cd_map[i,j] = cd_nm

plt.figure(figsize=(5.5,4.5))
sns.heatmap(cd_map, xticklabels=np.round(dose_vals,1),
                     yticklabels=np.round(focus_vals,3),
                     cmap="viridis", cbar_kws={'label':'CD [nm]'})
plt.xlabel("Dose [mJ/cm²]"); plt.ylabel("Focus [µm]")
plt.title("Focus–Exposure Process Window")
plt.tight_layout(); plt.savefig(OUT/"process_window.png", dpi=160)
print("✓ Process window saved")

# 3) ROC curve for ML detector ---------------------------------------
device = torch.device('cpu')
model  = SimpleAutoencoder(16).to(device)
model.load_state_dict(torch.load("models/autoencoder_simple_best.pth", map_location=device))
model.eval()

ds_test = WaferDataset("data/processed_images/test")
loader  = torch.utils.data.DataLoader(ds_test, batch_size=1)
scores, labels = [], []

def local_score(x, r, crop=16, top=0.01):
    diff=(x-r).abs().squeeze(); diff=diff[crop:-crop,crop:-crop]
    flat = diff.view(-1).numpy()
    k=int(len(flat)*top); return flat[np.argpartition(flat,-k)[-k:]].mean()

for img,_ in tqdm(loader, desc="ROC"):
    r = model(img)
    scores.append(local_score(img, r))
    labels.append(1 if "defect_" in ds_test.image_paths[len(scores)-1].name else 0)

fpr,tpr,thr = roc_curve(labels, scores)
roc_auc = auc(fpr,tpr)

plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'k--'); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC – Anomaly Detector"); plt.legend(); plt.tight_layout()
plt.savefig(OUT/"roc_curve.png", dpi=160)
print("✓ ROC curve saved  (AUC≈%.3f)"%roc_auc)

