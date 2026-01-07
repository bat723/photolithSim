"""Run trained model on test set to detect anomalies."""
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append('.')

from src.detection.autoencoder import ConvAutoencoder
from src.detection.dataset import WaferDataset


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load trained model
    print("\nLoading trained model...")
    model = ConvAutoencoder(latent_dim=64).to(device)
    model.load_state_dict(torch.load('models/autoencoder_best.pth', map_location=device))
    model.eval()
    print("✓ Model loaded")

    # Load test images
    print("\nLoading test images...")
    test_dataset = WaferDataset('data/processed_images/test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")

    # Run inference
    print("\nRunning inference...")
    errors = []
    image_names = []

    for idx, (image, _) in enumerate(tqdm(test_loader)):
        image = image.to(device)
        with torch.no_grad():
            reconstructed = model(image)
            error = torch.mean((image - reconstructed) ** 2).item()
        errors.append(error)
        image_names.append(test_dataset.image_paths[idx].name)

    errors = np.array(errors)

    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    np.save(results_dir / 'reconstruction_errors.npy', errors)

    # Calculate threshold (median + 1.5*IQR for outlier detection)
    q1, q3 = np.percentile(errors, [25, 75])
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr

    # Write predictions
    with open(results_dir / 'predictions.txt', 'w') as f:
        f.write(f"Threshold: {threshold:.6f}\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Image Name':<25} {'Error':>12} {'Prediction':>15}\n")
        f.write("-" * 70 + "\n")

        for name, error in sorted(zip(image_names, errors), key=lambda x: x[1], reverse=True):
            label = "DEFECT ⚠️" if error > threshold else "CLEAN ✓"
            f.write(f"{name:<25} {error:>12.6f} {label:>15}\n")

    # Summary statistics
    clean_count = np.sum(errors <= threshold)
    defect_count = np.sum(errors > threshold)

    print(f"\n✓ Inference complete!")
    print(f"  Total images: {len(errors)}")
    print(f"  Mean error: {errors.mean():.6f}")
    print(f"  Std error: {errors.std():.6f}")
    print(f"  Detection threshold: {threshold:.6f}")
    print(f"  Predicted CLEAN: {clean_count}")
    print(f"  Predicted DEFECT: {defect_count}")
    print(f"\n  Results saved to results/")


if __name__ == "__main__":
    main()
