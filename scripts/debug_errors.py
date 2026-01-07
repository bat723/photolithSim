"""Check if there's any separation between clean and defect errors."""
import numpy as np

errors = np.load('results/reconstruction_errors.npy')

# Read predictions to separate clean vs defect
with open('results/predictions.txt', 'r') as f:
    lines = f.readlines()[3:]  # Skip header

clean_errors = []
defect_errors = []

for line in lines:
    if 'clean_' in line.lower():
        error = float(line.split()[1])
        clean_errors.append(error)
    elif 'defect_' in line.lower():
        error = float(line.split()[1])
        defect_errors.append(error)

clean_errors = np.array(clean_errors)
defect_errors = np.array(defect_errors)

print("=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)
print(f"\nCLEAN images ({len(clean_errors)} images):")
print(f"  Mean: {clean_errors.mean():.6f}")
print(f"  Min:  {clean_errors.min():.6f}")
print(f"  Max:  {clean_errors.max():.6f}")
print(f"  Std:  {clean_errors.std():.6f}")

print(f"\nDEFECT images ({len(defect_errors)} images):")
print(f"  Mean: {defect_errors.mean():.6f}")
print(f"  Min:  {defect_errors.min():.6f}")
print(f"  Max:  {defect_errors.max():.6f}")
print(f"  Std:  {defect_errors.std():.6f}")

print(f"\nDIFFERENCE:")
print(f"  Defect mean - Clean mean: {defect_errors.mean() - clean_errors.mean():.6f}")
print(f"  Ratio (Defect/Clean): {defect_errors.mean() / clean_errors.mean():.2f}x")

if defect_errors.mean() < clean_errors.mean() * 1.5:
    print("\n⚠️  WARNING: Defects are NOT significantly different from clean!")
    print("   The model is too powerful and reconstructs everything well.")
    print("   SOLUTION: Retrain with a MUCH SIMPLER model.")
else:
    print("\n✓ Good separation! Just need to adjust threshold.")

