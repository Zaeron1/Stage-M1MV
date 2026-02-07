# fuse_physical.py
# ------------------------------------------------------------
# FUSE — Détection physique dirigée (seed + montée)
# ------------------------------------------------------------

import numpy as np
from PIL import Image
from skimage.filters import sobel
from skimage.morphology import disk, opening, closing
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter


def _luminance(rgb):
    rgb = rgb.astype(np.float32)
    return (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )


def _local_sigma(img, win=7):
    mu = uniform_filter(img, win)
    mu2 = uniform_filter(img**2, win)
    return np.sqrt(np.clip(mu2 - mu**2, 0, None))


def fuse_detect(
    image_path,
    ROI,
    *,
    seed_xy,          # (x, y) EN COORD ROI
    S_min=200,
    min_slope_deg=1,  # pente minimale imposée
    show=True,
    save_fig=None,
):
    """
    Détection physique dirigée de fumerolle.

    Retourne
    --------
    detected : bool
    mask_final : np.ndarray (ROI)
    metrics : dict
    """

    # --------------------------------------------------------
    # 1) LOAD + ROI
    # --------------------------------------------------------
    rgb_full = np.array(Image.open(image_path).convert("RGB"))
    x1, y1, x2, y2 = ROI
    rgb = rgb_full[y1:y2, x1:x2]

    sx, sy = seed_xy
    if not (0 <= sx < rgb.shape[1] and 0 <= sy < rgb.shape[0]):
        raise ValueError("Seed hors ROI")

    # --------------------------------------------------------
    # 2) PRETRAITEMENT
    # --------------------------------------------------------
    Y = _luminance(rgb)
    Yn = (Y - np.median(Y)) / (np.std(Y) + 1e-6)

    # --------------------------------------------------------
    # 3) INDICES PHYSIQUES
    # --------------------------------------------------------
    dY = np.clip(Yn - np.percentile(Yn, 30), 0, None)
    sigma = _local_sigma(Yn, win=7)
    edge = sobel(Yn)

    score = dY + 0.7 * sigma + 0.3 * edge

    # --------------------------------------------------------
    # 4) PRE-MASQUE
    # --------------------------------------------------------
    t = np.percentile(score, 90)
    pre_mask = score > t
    pre_mask = opening(pre_mask, disk(1))
    pre_mask = closing(pre_mask, disk(2))

    # --------------------------------------------------------
    # 5) COMPOSANTE CONNECTÉE À LA SEED
    # --------------------------------------------------------
    lab = label(pre_mask)
    lab_seed = lab[sy, sx]

    if lab_seed == 0:
        return False, np.zeros_like(pre_mask), {"reason": "seed_not_connected"}

    mask = lab == lab_seed

    if mask.sum() < S_min:
        return False, np.zeros_like(pre_mask), {"reason": "too_small"}

    # --------------------------------------------------------
    # 6) CONTRAINTE DE MONTÉE (PHYSIQUE)
    # --------------------------------------------------------
    yy, xx = np.nonzero(mask)
    cy = yy.mean()

    # doit monter → barycentre au-dessus de la seed
    if cy >= sy:
        return False, np.zeros_like(pre_mask), {"reason": "not_ascending"}

    # pente moyenne
    dy = sy - yy
    dx = np.abs(xx - sx) + 1e-6
    slope = np.degrees(np.arctan(dy / dx)).mean()

    if slope < min_slope_deg:
        return False, np.zeros_like(pre_mask), {"reason": "slope_too_low"}

    detected = True

    metrics = {
        "area_px": int(mask.sum()),
        "mean_score": float(score[mask].mean()),
        "slope_deg": float(slope),
        "reason": "ok",
    }

    # --------------------------------------------------------
    # 7) FIGURE DIAGNOSTIC
    # --------------------------------------------------------
    fig, ax = plt.subplots(1, 5, figsize=(20, 4))
    for a in ax:
        a.axis("off")

    ax[0].set_title("RGB (ROI)")
    ax[0].imshow(rgb)

    ax[1].set_title("dY")
    ax[1].imshow(dY, cmap="inferno")

    ax[2].set_title("sigma")
    ax[2].imshow(sigma, cmap="inferno")

    ax[3].set_title("score")
    ax[3].imshow(score, cmap="inferno")

    overlay = rgb.copy()
    overlay[mask] = [255, 0, 0]
    ax[4].set_title(f"mask_final ({metrics['reason']})")
    ax[4].imshow(overlay)

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return detected, mask, metrics
