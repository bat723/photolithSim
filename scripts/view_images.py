"""Quick image viewer for generated data."""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

# View a few clean images
clean_dir = Path("data/raw_synthetic/clean")
defect_dir = Path("data/raw_synthetic/defect")

clean_images = sorted(list(clean_dir.glob("*.png")))[:6]
defect_images = sorted(list(defect_dir.glob("*.png")))[:6]

fig, axes = plt.subplots(2, 6, figsize=(15, 5))
fig.suptitle('Generated Lithography Images', fontsize=16)

# Show clean images
for i, img_path in enumerate(clean_images):
    img = mpimg.imread(img_path)
    axes[0, i].imshow(img, cmap='gray')
    axes[0, i].set_title(f'Clean {i}')
    axes[0, i].axis('off')

# Show defect images
for i, img_path in enumerate(defect_images):
    img = mpimg.imread(img_path)
    axes[1, i].imshow(img, cmap='gray')
    axes[1, i].set_title(f'Defect {i}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('results/sample_images.png', dpi=150, bbox_inches='tight')
print("✓ Sample grid saved to results/sample_images.png")
print("\nTo view it, run: open results/sample_images.png")
