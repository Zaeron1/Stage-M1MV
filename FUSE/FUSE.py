import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import cv2
from ultralytics import SAM

from skimage.morphology import (
    opening, closing, disk,
    remove_small_objects, dilation
)
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage.segmentation import flood


# ============================================================
# UTILITAIRES BASIQUES
# ============================================================

def rgb_to_luminance(img_rgb):
    img = img_rgb.astype(np.float32)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    return 0.2126 * R + 0.7152 * G + 0.0722 * B


def local_std(image, win=9):
    image_n = (image - image.min()) / (image.max() - image.min() + 1e-6)
    image_u8 = img_as_ubyte(image_n)
    mean = rank.mean(image_u8, disk(win // 2))
    mean_sq = rank.mean(image_u8 ** 2, disk(win // 2))
    return np.sqrt(np.maximum(mean_sq - mean ** 2, 0))


def dominance_ratio(arr):
    if arr is None or arr.size < 20:
        return 0.0

    p50 = np.percentile(arr, 50)
    p90 = np.percentile(arr, 90)
    p99 = np.percentile(arr, 99)

    denom = p90 - p50
    if denom <= 0:
        return 0.0

    return (p99 - p90) / denom


# ============================================================
# SAM — INITIALISATION
# ============================================================

sam_model = SAM("FUSE/sam_b.pt")


def sam_candidates_roi(image_rgb, roi):
    x1, y1, x2, y2 = roi
    roi_img = image_rgb[y1:y2, x1:x2]
    h, w, _ = roi_img.shape

    points_pos = [
        [w // 2, int(0.25 * h)],
        [w // 2, int(0.45 * h)],
    ]

    points_neg = [
        [w // 2, int(0.90 * h)],
        [int(0.10 * w), int(0.50 * h)],
        [int(0.90 * w), int(0.50 * h)],
    ]

    points = points_pos + points_neg
    labels = [1]*len(points_pos) + [0]*len(points_neg)

    results = sam_model(
        roi_img,
        points=points,
        labels=labels,
        conf=0.25
    )

    if results[0].masks is None:
        return []

    return results[0].masks.data.cpu().numpy()


def score_sam_mask(mask, deltaY, sigma_local, S_min):
    area = int(mask.sum())
    if area < S_min:
        return -np.inf

    valsY = deltaY[mask]
    valsS = sigma_local[mask]

    if valsY.size < 20 or valsS.size < 20:
        return -np.inf

    domY = dominance_ratio(valsY)
    domS = dominance_ratio(valsS)

    width = np.max(mask.sum(axis=0))
    height = mask.shape[0]
    elongation = height / (width + 1e-6)

    return (
        2.0 * domY +
        1.5 * domS +
        0.001 * area +
        0.5 * elongation
    )


# ============================================================
# DÉTECTION PANACHE (PHYSIQUE + SAM)
# ============================================================

def detect_panache_visible(image_rgb, roi, S_min=80, show=True):

    x1, y1, x2, y2 = roi
    roi_rgb = image_rgb[y1:y2, x1:x2]

    # --- luminance
    Y = rgb_to_luminance(roi_rgb)
    Yn = (Y - np.median(Y)) / (np.std(Y) + 1e-6)

    # --- anomalies
    Y_bg = np.percentile(Yn, 30)
    deltaY = np.abs(Yn - Y_bg)
    sigma_local = local_std(Yn, win=9)

    domY = dominance_ratio(deltaY)
    domS = dominance_ratio(sigma_local)

    if domY < 1.2 and domS < 1.2:
        return False, np.zeros_like(Y, bool), {
            "panache": False,
            "reason": "ROI homogène",
            "domY": domY,
            "domS": domS
        }

    # --- noyau strict
    thr_core = 2.2 * np.std(Yn)
    mask_core = deltaY > thr_core
    mask_core = opening(mask_core, disk(2))
    mask_core = remove_small_objects(mask_core, min_size=S_min)

    if mask_core.sum() == 0:
        return False, np.zeros_like(Y, bool), {
            "panache": False,
            "reason": "Noyau vide"
        }

    # --- extension conditionnelle
    thr_edge = 1.1 * np.std(Yn)
    sigma_thr = np.percentile(sigma_local, 70)

    mask_ext = np.zeros_like(mask_core)

    for y, x in np.argwhere(mask_core):
        flooded = flood(deltaY, (y, x), tolerance=thr_edge)
        flooded &= (deltaY > thr_edge) | (sigma_local > sigma_thr)
        mask_ext |= flooded

    mask_ext = dilation(mask_ext, disk(2))
    mask_ext = closing(mask_ext, disk(3))
    mask_ext = remove_small_objects(mask_ext, min_size=S_min)

    if mask_ext.sum() == 0:
        return False, np.zeros_like(Y, bool), {
            "panache": False,
            "reason": "Extension vide"
        }

    # --- SAM
    sam_masks = sam_candidates_roi(image_rgb, roi)

    best_score = -np.inf
    best_mask = None

    for m in sam_masks:
        score = score_sam_mask(m, deltaY, sigma_local, S_min)
        if score > best_score:
            best_score = score
            best_mask = m

    if best_mask is None:
        return False, np.zeros_like(Y, bool), {
            "panache": False,
            "reason": "SAM rejeté"
        }

    # --- fusion finale
    mask_final = best_mask & mask_ext
    mask_final = remove_small_objects(mask_final, min_size=S_min)

    if mask_final.sum() < S_min:
        return False, np.zeros_like(Y, bool), {
            "panache": False,
            "reason": "Masque final vide"
        }

    metrics = {
        "panache": True,
        "area_px": int(mask_final.sum()),
        "domY": domY,
        "domS": domS,
        "sam_score": best_score
    }

    if show:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(roi_rgb)
        ax[0].set_title("ROI")
        ax[0].axis("off")

        ax[1].imshow(mask_ext, cmap="gray")
        ax[1].set_title("Masque physique")
        ax[1].axis("off")

        ax[2].imshow(roi_rgb)
        ax[2].imshow(mask_final, cmap="Reds", alpha=0.5)
        ax[2].set_title("Masque final (SAM ∩ Physique)")
        ax[2].axis("off")

        plt.tight_layout()
        plt.show()

    return True, mask_final, metrics


# ============================================================
# EXEMPLE
# ============================================================

if __name__ == "__main__":

    IMAGE_PATH = "/Users/alexandremichaux/Documents/UCA/Cours/Stage/projet/Code/Stage-M1/FUSE/images/image.png"
    ROI = (0, 0, 700, 500)

    img = np.array(Image.open(IMAGE_PATH).convert("RGB"))

    panache, mask, metrics = detect_panache_visible(
        img,
        ROI,
        show=True
    )

    print("Panache détecté :", panache)
    for k, v in metrics.items():
        print(f"{k} : {v}")

