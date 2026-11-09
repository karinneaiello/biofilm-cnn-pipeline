"""Show one spinning-disk image under each thresholding method (for slides)."""
import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from release_preprocess import apply_threshold

# trim the list depending on what to visualize
METHODS = ["otsu", "isodata", "li", "triangle", "yen", "adaptive"]


def load_normalized(path):
    """Same steps as _build_pairs_spinning_disk: grayscale, divide by max."""
    img = tifffile.imread(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    return img / img.max() if img.max() > 0 else img


def save_one(arr, title, path, vmax=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(arr, cmap="gray", vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(image_path, out_dir="outputs/threshold_visualization"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img = load_normalized(image_path)

    # show the og as the CNN sees it & a contrast-stretched copy that's easier to see on a slide
    save_one(img, "Original (as the CNN sees it)", out / "original.png")
    save_one(img, "Original (display-stretched)", out / "original_display_stretched.png",
             vmax=float(np.percentile(img, 99.5)))

    # make one png per method & collect masks for the grid
    masks = {}
    for m in METHODS:
        mask = apply_threshold(img, m)
        masks[m] = mask
        pct = 100.0 * float(mask.mean())
        save_one(mask, f"{m}  ({pct:.1f}% foreground)", out / f"{m}.png")
        print(f"{m:>18}: {pct:.1f}% foreground")

    # make a side by side grid with the original first then every method
    n = len(METHODS) + 1
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=float(np.percentile(img, 99.5)))
    axes[0].set_title("Original (display-stretched)")
    for ax, m in zip(axes[1:], METHODS):
        ax.imshow(masks[m], cmap="gray", vmin=0, vmax=1)
        ax.set_title(m)
    for ax in axes:
        ax.axis("off")
    fig.savefig(out / "all_methods.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
    
