import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from skimage.morphology import (
    opening, closing, disk,
    remove_small_objects, dilation
)
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage.segmentation import flood
from tqdm import tqdm



# ============================================================
# UTILITAIRES
# ============================================================

def rgb_to_luminance(img_rgb):
    img = img_rgb.astype(np.float32)
    R, G, B = img[..., 0], img[..., 1], img_rgb[..., 2]
    return 0.2126 * R + 0.7152 * G + 0.0722 * B


def local_std(image, win=9):
    image_n = (image - image.min()) / (image.max() - image.min() + 1e-6)
    image_u8 = img_as_ubyte(image_n)
    mean = rank.mean(image_u8, disk(win // 2))
    mean_sq = rank.mean(image_u8 ** 2, disk(win // 2))
    return np.sqrt(mean_sq - mean ** 2)


# ============================================================
# DÉTECTION PANACHE VISIBLE — NOYAU + EXTENSION
# ============================================================

def detect_panache_visible(
    image_rgb,
    roi,
    S_min=120,
    k_core=2.2,
    k_edge=1.4,
    win_tex=9,
    show=True
):
    """
    Détection et segmentation complète d'un panache visible.
    Méthode :
      1. noyau (anomalie forte)
      2. extension conditionnelle (zones diffuses connectées)
    """

    # --------------------------------------------------------
    # 1. EXTRACTION ROI
    # --------------------------------------------------------
    x1, y1, x2, y2 = roi
    roi_rgb = image_rgb[y1:y2, x1:x2]

    # --------------------------------------------------------
    # 2. LUMINANCE
    # --------------------------------------------------------
    Y = rgb_to_luminance(roi_rgb)
    Yn = (Y - np.median(Y)) / (np.std(Y) + 1e-6)

    # --------------------------------------------------------
    # 3. FOND + ANOMALIE
    # --------------------------------------------------------
    Y_bg = np.percentile(Yn, 30)
    deltaY = np.abs(Yn - Y_bg)

    sigma = np.std(Yn)

    # --------------------------------------------------------
    # 4. NOYAU (SEUIL STRICT)
    # --------------------------------------------------------
    thr_core = k_core * sigma
    mask_core = deltaY > thr_core

    mask_core = opening(mask_core, disk(2))
    mask_core = remove_small_objects(mask_core, min_size=S_min)

    # --------------------------------------------------------
    # 5. EXTENSION CONDITIONNELLE
    # --------------------------------------------------------
    thr_edge = k_edge * sigma
    mask_ext = np.zeros_like(mask_core)

    seeds = np.argwhere(mask_core)

    for y, x in tqdm(seeds, desc="Extension du panache", unit="seed"):
        flooded = flood(deltaY, (y, x), tolerance=thr_edge)
        mask_ext |= flooded

    mask_ext = closing(mask_ext, disk(3))
    mask_ext = remove_small_objects(mask_ext, min_size=S_min)

    # --------------------------------------------------------
    # 6. TEXTURE (DIAGNOSTIC)
    # --------------------------------------------------------
    sigma_local = local_std(Yn, win=win_tex)

    # --------------------------------------------------------
    # 7. MÉTRIQUES
    # --------------------------------------------------------
    area = int(mask_ext.sum())

    if area > 0:
        mean_anomaly = float(np.mean(deltaY[mask_ext]))
        texture_panache = float(np.median(sigma_local[mask_ext]))
        texture_fond = float(np.median(sigma_local[~mask_ext]))
        diffusion_index = texture_fond - texture_panache
    else:
        mean_anomaly = 0.0
        diffusion_index = 0.0

    panache = area >= S_min

    metrics = {
        "panache": panache,
        "area_px": area,
        "mean_anomaly": mean_anomaly,
        "diffusion_index": diffusion_index,
        "thr_core": thr_core,
        "thr_edge": thr_edge,
    }

    # --------------------------------------------------------
    # 8. FIGURES DEBUG
    # --------------------------------------------------------
    if show:
        fig, ax = plt.subplots(2, 4, figsize=(18, 8))

        ax[0, 0].imshow(image_rgb)
        ax[0, 0].add_patch(
            plt.Rectangle((x1, y1), x2-x1, y2-y1,
                          edgecolor="red", facecolor="none", lw=2)
        )
        ax[0, 0].set_title("Image visible + ROI")
        ax[0, 0].axis("off")

        ax[0, 1].imshow(roi_rgb)
        ax[0, 1].set_title("ROI (RGB)")
        ax[0, 1].axis("off")

        ax[0, 2].imshow(Y, cmap="gray")
        ax[0, 2].set_title("Luminance Y")
        ax[0, 2].axis("off")

        ax[0, 3].imshow(deltaY, cmap="inferno")
        ax[0, 3].set_title("|ΔY| (anomalie)")
        ax[0, 3].axis("off")

        ax[1, 0].imshow(mask_core, cmap="gray")
        ax[1, 0].set_title("Noyau (anomalie forte)")
        ax[1, 0].axis("off")

        ax[1, 1].imshow(mask_ext, cmap="gray")
        ax[1, 1].set_title("Masque étendu (final)")
        ax[1, 1].axis("off")

        ax[1, 2].imshow(sigma_local, cmap="viridis")
        ax[1, 2].set_title("Texture σ locale (diagnostic)")
        ax[1, 2].axis("off")

        ax[1, 3].imshow(Y, cmap="gray")
        ax[1, 3].imshow(mask_ext, cmap="Reds", alpha=0.5)
        ax[1, 3].set_title(
            f"Résultat : {'PANACHE' if panache else 'PAS DE PANACHE'}"
        )
        ax[1, 3].axis("off")

        plt.tight_layout()
        plt.show()

    return panache, mask_ext, metrics


# ============================================================
# EXEMPLE D'UTILISATION
# ============================================================

if __name__ == "__main__":
    IMAGE_PATH = "CAGED/image.png"
    ROI = (0, 0, 200, 120)  # x1, y1, x2, y2

    img = np.array(Image.open(IMAGE_PATH).convert("RGB"))

    panache, mask, metrics = detect_panache_visible(
        img,
        ROI,
        show=True
    )

    print("Panache détecté :", panache)
    for k, v in metrics.items():
        print(f"{k} : {v}")

