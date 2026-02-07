import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple, Dict, Any

Rect = Tuple[int, int, int, int]

# Defaults (tu ajustes depuis main si besoin)
S_MIN = 8
WIN   = 9
W_TEX = 0.6
P_CORE, P_EDGE, P_TEX, P_SIG = 98, 92, 80, 85
OPEN_R, CLOSE_R, EDGE_CLOSE_R = 2, 2, 2


def clamp(rect: Rect, W: int, H: int) -> Optional[Rect]:
    x1, y1, x2, y2 = map(int, rect)
    x1 = max(0, min(x1, W)); x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H)); y2 = max(0, min(y2, H))
    return None if (x2 <= x1 or y2 <= y1) else (x1, y1, x2, y2)


def morph(b: np.ndarray, open_r: int = 0, close_r: int = 0) -> np.ndarray:
    u = (b.astype(np.uint8) * 255)
    if open_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*open_r+1, 2*open_r+1))
        u = cv2.morphologyEx(u, cv2.MORPH_OPEN, k)
    if close_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*close_r+1, 2*close_r+1))
        u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, k)
    return (u > 0)


def chercher_fumerolle(
    img_path: str,
    mask_path: str,
    ROI: Rect,
    SEED: Optional[Rect] = None,
    *,
    s_min: int = S_MIN,
    win: int = WIN,
    w_tex: float = W_TEX,
    p_core: float = P_CORE,
    p_edge: float = P_EDGE,
    p_tex: float = P_TEX,
    p_sig: float = P_SIG,
    open_r: int = OPEN_R,
    close_r: int = CLOSE_R,
    edge_close_r: int = EDGE_CLOSE_R,
    return_debug: bool = True,
) -> Dict[str, Any]:
    """
    Ne fait AUCUNE figure.
    Retourne masques + métriques, et (optionnellement) matrices debug pour que main.py fasse les figures.
    """

    rgb_full = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    Hf, Wf = rgb_full.shape[:2]

    m = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    if m.shape != (Hf, Wf):
        m = cv2.resize(m, (Wf, Hf), interpolation=cv2.INTER_NEAREST)
    allowed_full = (m > 127)

    ROI2 = clamp(ROI, Wf, Hf)
    if ROI2 is None:
        raise RuntimeError("ROI hors image")
    SEED2 = clamp(SEED, Wf, Hf) if SEED is not None else None

    x1, y1, x2, y2 = ROI2
    rgb = rgb_full[y1:y2, x1:x2]
    allowed = allowed_full[y1:y2, x1:x2]
    H, W = rgb.shape[:2]
    if int(allowed.sum()) < 50:
        raise RuntimeError("Trop peu de pixels autorisés dans la ROI")

    seed_roi = np.zeros((H, W), dtype=bool)
    if SEED2 is not None:
        sx1, sy1, sx2, sy2 = SEED2
        sx1 -= x1; sx2 -= x1; sy1 -= y1; sy2 -= y1
        sx1 = max(0, min(sx1, W)); sx2 = max(0, min(sx2, W))
        sy1 = max(0, min(sy1, H)); sy2 = max(0, min(sy2, H))
        if sx2 > sx1 and sy2 > sy1:
            seed_roi[sy1:sy2, sx1:sx2] = True
    seed_roi &= allowed

    f = rgb.astype(np.float32)
    Y = 0.2126*f[..., 0] + 0.7152*f[..., 1] + 0.0722*f[..., 2]

    Yv = Y[allowed]
    med = np.median(Yv)
    sd  = np.std(Yv) + 1e-6
    Yn  = (Y - med) / sd

    bg = np.percentile(Yn[allowed], 30)
    dY = np.abs(Yn - bg)

    Yn2 = Yn.copy()
    Yn2[~allowed] = bg
    k = (win, win)
    mu  = cv2.boxFilter(Yn2, -1, k, normalize=True)
    mu2 = cv2.boxFilter(Yn2*Yn2, -1, k, normalize=True)
    sig = np.sqrt(np.maximum(mu2 - mu*mu, 0.0))

    dY *= allowed
    sig *= allowed

    dv = dY[allowed]
    sv = sig[allowed]
    t_core = np.percentile(dv, p_core)
    t_edge = np.percentile(dv, p_edge)
    t_tex  = np.percentile(dv, p_tex)
    t_sig  = np.percentile(sv, p_sig)

    core = (dY > t_core) & allowed
    edge = ((dY > t_edge) | ((sig > t_sig) & (dY > t_tex))) & allowed

    core = morph(core, open_r=open_r, close_r=0)
    edge = morph(edge, open_r=0, close_r=edge_close_r)

    if int(core.sum()) < s_min:
        raise RuntimeError("Core trop petit (pas de détection)")

    n, lab, stats, _ = cv2.connectedComponentsWithStats((edge.astype(np.uint8) * 255), 8)
    if n <= 1:
        raise RuntimeError("Aucune composante")

    seed_labels = np.unique(lab[seed_roi]) if seed_roi.any() else np.unique(lab[core])
    seed_labels = seed_labels[seed_labels != 0]
    if seed_labels.size == 0:
        raise RuntimeError("Aucune composante ne touche seed/core")

    score_px = (dY + w_tex * sig).astype(np.float32)
    flat = lab.reshape(-1)
    sum_score = np.bincount(flat, weights=score_px.reshape(-1), minlength=n)
    count_px  = np.bincount(flat, minlength=n).astype(np.float32)
    mean_score = sum_score / (count_px + 1e-6)
    core_cnt = np.bincount(flat, weights=core.reshape(-1).astype(np.float32), minlength=n)

    best = 0
    best_s = -1e30
    for L in seed_labels.astype(int):
        if stats[L, cv2.CC_STAT_AREA] < s_min:
            continue
        if core_cnt[L] < max(10, s_min // 8):
            continue
        if mean_score[L] > best_s:
            best, best_s = L, mean_score[L]
    if best == 0:
        raise RuntimeError("Aucun candidat ne passe les contraintes")

    mask_roi = morph((lab == best), open_r=0, close_r=close_r)
    detected = int(mask_roi.sum()) >= s_min

    mask_roi_u8 = (mask_roi.astype(np.uint8) * 255)
    mask_full_u8 = np.zeros((Hf, Wf), dtype=np.uint8)
    mask_full_u8[y1:y2, x1:x2] = mask_roi_u8

    roi_overlay = rgb.copy()
    roi_overlay[mask_roi] = (255, 0, 0)

    area_px = int(mask_roi.sum())

    out: Dict[str, Any] = {
        "detected": bool(detected),
        "area_px": int(area_px),
        "best_label": int(best),
        "mask_full_u8": mask_full_u8,
        "mask_roi_u8": mask_roi_u8,
        "ROI": ROI2,
        "SEED": SEED2,
    }

    if return_debug:
        out["debug"] = {
            "rgb": rgb,
            "allowed": allowed,
            "dY": dY,
            "sig": sig,
            "core": core,
            "edge": edge,
            "roi_overlay": roi_overlay,
            "seed_roi": seed_roi,
        }

    return out
