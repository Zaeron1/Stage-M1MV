# ============================================================
# fuse_detection.py — VERSION QUI MARCHE EN PRATIQUE
# - Ne lève PAS d'exception: renvoie detected=False + error=...
# - Toujours: rgb_full + plume_mask(full) + debug(6 panneaux)
# - Adaptatif sombre: CLAHE sur L + score clair/peu saturé + 3 passes relax
# ============================================================

import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple, Dict, Any

Rect = Tuple[int, int, int, int]

def _clamp(rect: Rect, W: int, H: int):
    x1,y1,x2,y2 = map(int, rect)
    x1 = max(0, min(x1, W)); x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H)); y2 = max(0, min(y2, H))
    if x2 <= x1 or y2 <= y1: return None
    return (x1,y1,x2,y2)

def _seed_center(SEED: Optional[Rect]):
    if SEED is None: return None
    x1,y1,x2,y2 = map(int, SEED)
    return ((x1+x2)//2, (y1+y2)//2)

def _clahe(u8, clip=2.0):
    return cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8,8)).apply(u8)

def _local_sigma(x_f, win=9):
    k = (win, win)
    mu  = cv2.boxFilter(x_f, -1, k, normalize=True)
    mu2 = cv2.boxFilter(x_f*x_f, -1, k, normalize=True)
    return np.sqrt(np.maximum(mu2 - mu*mu, 0.0))

def chercher_fumerolle(
    img_path: str,
    mask_path: str,
    ROI: Rect,
    SEED: Optional[Rect] = None,
    *,
    return_debug: bool = True,
) -> Dict[str, Any]:

    out: Dict[str, Any] = {"detected": False, "area_px": 0, "error": ""}

    # ---- load image ----
    rgb_full = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    Hf, Wf = rgb_full.shape[:2]

    # ---- allowed mask (full) ----
    m = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    if m.shape != (Hf, Wf):
        m = cv2.resize(m, (Wf, Hf), interpolation=cv2.INTER_NEAREST)
    allowed_full = (m > 127)

    ROI2 = _clamp(ROI, Wf, Hf)
    if ROI2 is None:
        err = "ROI hors image"
        plume = np.zeros((Hf, Wf), dtype=bool)
        out.update({"rgb_full": rgb_full, "plume_mask": plume, "ROI": ROI2, "SEED": SEED, "error": err})
        if return_debug:
            out["debug"] = {"visible": rgb_full, "Y": None, "sigma": None, "cor": None, "edge": None, "overlay": rgb_full}
        return out

    x1,y1,x2,y2 = ROI2
    rgb = rgb_full[y1:y2, x1:x2]
    allowed = allowed_full[y1:y2, x1:x2]
    if int(allowed.sum()) < 50:
        err = "Trop peu de pixels autorisés ROI"
        plume = np.zeros((Hf, Wf), dtype=bool)
        out.update({"rgb_full": rgb_full, "plume_mask": plume, "ROI": ROI2, "SEED": SEED, "error": err})
        if return_debug:
            out["debug"] = {"visible": rgb_full, "Y": None, "sigma": None, "cor": None, "edge": None, "overlay": rgb_full}
        return out

    # ---- seed centre -> coords ROI ----
    seed_xy = _seed_center(SEED)
    seed_roi = None
    if seed_xy is not None:
        sx, sy = seed_xy
        if x1 <= sx < x2 and y1 <= sy < y2:
            seed_roi = (sx - x1, sy - y1)

    # ---- features (adaptatifs) ----
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[...,0].astype(np.uint8)
    L_eq = _clahe(L, clip=2.0).astype(np.float32)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    S = hsv[...,1].astype(np.float32)

    Lf = (L_eq / 255.0).astype(np.float32)
    sigma = _local_sigma(Lf, win=9).astype(np.float32)

    edge = (cv2.Canny(L_eq.astype(np.uint8), 40, 120).astype(np.float32) / 255.0)
    edge *= allowed

    # score = clair + peu saturé + un peu texture/edge
    Lmu, Lsd = float(np.mean(L_eq[allowed])), float(np.std(L_eq[allowed]) + 1e-6)
    Smu, Ssd = float(np.mean(S[allowed])),    float(np.std(S[allowed])    + 1e-6)
    Ln = (L_eq - Lmu) / Lsd
    Sn = (S    - Smu) / Ssd

    score = (1.2*Ln) + (0.9*(-Sn)) + (0.25*sigma) + (0.25*edge)
    score *= allowed

    # ---- 3 passes: seuil score + seuil saturation + dilatation ----
    passes = [
        (96, 55, 1),
        (92, 65, 2),
        (88, 75, 3),
    ]

    best = np.zeros_like(allowed, dtype=bool)
    best_area = 0
    reason = "no_candidate"

    for p_score, p_sat, dil in passes:
        t  = np.percentile(score[allowed], p_score)
        ts = np.percentile(S[allowed], p_sat)

        cand = (score > t) & (S < ts) & allowed

        if dil > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dil+1, 2*dil+1))
            cand = (cv2.dilate(cand.astype(np.uint8)*255, k) > 0) & allowed

        n, labc, stats, _ = cv2.connectedComponentsWithStats((cand.astype(np.uint8)*255), 8)
        if n <= 1:
            reason = "no_components"
            continue

        target = 0
        if seed_roi is not None:
            sx, sy = seed_roi
            Lseed = int(labc[sy, sx])
            if Lseed != 0:
                target = Lseed
            else:
                # plus proche du seed
                best_d = 1e30
                for Llab in range(1, n):
                    x, y, w, h, area = stats[Llab]
                    if area < 150: 
                        continue
                    cx, cy = x + w/2.0, y + h/2.0
                    d = (cx - sx)**2 + (cy - sy)**2
                    if d < best_d:
                        best_d = d
                        target = Llab
        else:
            areas = stats[1:, cv2.CC_STAT_AREA]
            target = 1 + int(np.argmax(areas))

        if target == 0:
            reason = "no_target"
            continue

        mask = (labc == target)
        ys = np.where(mask)[0]
        if ys.size == 0:
            reason = "empty"
            continue
        height = int(ys.max() - ys.min() + 1)
        if height < 10:
            reason = "too_flat"
            continue

        area = int(mask.sum())
        if area > best_area:
            best_area = area
            best = mask
            reason = "OK"
            break

    plume_full = np.zeros((Hf, Wf), dtype=bool)
    plume_full[y1:y2, x1:x2] = best

    roi_overlay = rgb.copy()
    roi_overlay[best] = (255, 0, 0)

    out["detected"] = bool(best_area > 0)
    out["area_px"]  = int(best_area)
    out["error"]    = "" if out["detected"] else reason
    out["rgb_full"] = rgb_full
    out["plume_mask"] = plume_full
    out["ROI"] = ROI2
    out["SEED"] = SEED

    if return_debug:
        out["debug"] = {
            "visible": rgb_full,
            "Y": L_eq,
            "sigma": sigma,
            "cor": score,
            "edge": edge,
            "overlay": roi_overlay,
        }

    return out
