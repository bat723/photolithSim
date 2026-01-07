"""Quick check: view some test images to see defects."""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

test_dir = Path("data/processed_images/test")
clean_images = sorted(list(test_dir.glob("clean_*.png")))[:3]
defect_images = sorted(list(test_dir.glob("defect_*.png")))[:3]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Test Images: Are Defects Visible?', fontsize=16)

for i, img_path in enumerate(clean_images):
    img = mpimg.imread(img_path)
    axes[0, i].imshow(img, cmap='gray')
    axes[0, i].set_title(f'Clean {i}')
    axes[0, i].axis('off')

for i, img_path in enumerate(defect_images):
    img = mpimg.imread(img_path)
    axes[1, i].imshow(img, cmap='gray')
    axes[1, i].set_title(f'Defect {i}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('results/debug_defects.png', dpi=150, bbox_inches='tight')
print("✓ Saved to: results/debug_defects.png")
print("\nNow visually inspect: Are the defects obvious?")
print("If defects are barely visible, we need to make them stronger!")
