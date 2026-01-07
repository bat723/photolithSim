import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append('.')

from src.detection.autoencoder import ConvAutoencoder
from src.detection.dataset import WaferDataset


def anomaly_score(x, r, border=16, top_frac=0.01):
    # x, r: tensors shape (1,1,H,W)
    diff = (x - r).abs().squeeze(0).squeeze(0)  # (H,W)

    # crop borders to remove edge artifacts
    if border > 0:
        diff = diff[border:-border, border:-border]

    vals = diff.reshape(-1).cpu().numpy()

    # focus on worst pixels
    k = max(1, int(len(vals) * top_frac))
    topk = np.partition(vals, -k)[-k:]
    return float(np.mean(topk))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = ConvAutoencoder(latent_dim=64).to(device)
    model.load_state_dict(torch.load('models/autoencoder_best.pth', map_location=device))
    model.eval()

    test_dataset = WaferDataset('data/processed_images/test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")

    names, scores = [], []

    for idx, (x, _) in enumerate(tqdm(test_loader, desc="Inference")):
        x = x.to(device)
        with torch.no_grad():
            r = model(x)
        s = anomaly_score(x, r, border=16, top_frac=0.01)
        scores.append(s)
        names.append(test_dataset.image_paths[idx].name)

    scores = np.array(scores)

    # IMPORTANT: set threshold from CLEAN ONLY (not mixed data)
    clean_scores = scores[[i for i,n in enumerate(names) if n.startswith("clean_")]]
    thr = clean_scores.mean() + 4 * clean_scores.std()  # adjust 3-6 if needed

    pred_def = scores > thr
    print("\nResults:")
    print(f"  Mean score: {scores.mean():.6f}")
    print(f"  Threshold (clean mean + 4*std): {thr:.6f}")
    print(f"  Predicted CLEAN: {(~pred_def).sum()}")
    print(f"  Predicted DEFECT: {pred_def.sum()}")

    out = Path("results")
    out.mkdir(exist_ok=True)
    np.save(out / "anomaly_scores_localized.npy", scores)

    with open(out / "predictions_localized.txt", "w") as f:
        f.write(f"threshold={thr:.8f}\n")
        f.write("name,score,pred\n")
        order = np.argsort(-scores)
        for i in order:
            f.write(f"{names[i]},{scores[i]:.8f},{'DEFECT' if pred_def[i] else 'CLEAN'}\n")

    print("Saved:")
    print("  results/anomaly_scores_localized.npy")
    print("  results/predictions_localized.txt")


if __name__ == "__main__":
    main()
