# FUSE2_min_show.py — détection fumerolle (ROI + seed + masque exclusion) + debug images + show=True
# Image = matrices. On calcule: luminance Y, anomalie |ΔY|, texture σ locale, segmentation core/edge, composantes, choix meilleur label.

import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
BASE = os.path.dirname(os.path.abspath(__file__))
IMG_PATH  = os.path.join(BASE, "images", "image2.png")
MASK_PATH = os.path.join(BASE, "images", "mask2.png")      # blanc=autorisé, noir=exclu
ROI   = (1059, 119, 1659, 468)
SEED  = (1258, 295, 1330, 342)
OUT   = os.path.join(BASE, "exemple_output"); os.makedirs(OUT, exist_ok=True)

SHOW = True   # 
S_MIN = 80
WIN   = 9
W_TEX = 0.6
P_CORE, P_EDGE, P_TEX, P_SIG = 98, 92, 80, 85
OPEN_R, CLOSE_R, EDGE_CLOSE_R = 2, 2, 2

# ---------- HELPERS (courts) ----------
def clamp(rect, W, H):
    x1,y1,x2,y2 = map(int, rect)
    x1 = max(0, min(x1, W)); x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H)); y2 = max(0, min(y2, H))
    return None if (x2<=x1 or y2<=y1) else (x1,y1,x2,y2)

def morph(b, open_r=0, close_r=0):
    u = (b.astype(np.uint8) * 255)
    if open_r>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*open_r+1, 2*open_r+1))
        u = cv2.morphologyEx(u, cv2.MORPH_OPEN, k)
    if close_r>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*close_r+1, 2*close_r+1))
        u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, k)
    return (u > 0)

def save_gray_bool(b, path):
    Image.fromarray((b.astype(np.uint8)*255)).save(path)

def save_gray_u8(u8, path):
    Image.fromarray(u8).save(path)

def save_heatmap(mat, path, cmap="inferno"):
    m = mat.astype(np.float32)
    m = (m - m.min()) / (m.max() - m.min() + 1e-6)
    u8 = (m * 255).astype(np.uint8)
    rgb = cv2.applyColorMap(u8, getattr(cv2, "COLORMAP_INFERNO") if cmap=="inferno" else cv2.COLORMAP_VIRIDIS)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path)

# ---------- LOAD ----------
rgb_full = np.array(Image.open(IMG_PATH).convert("RGB"), dtype=np.uint8)
Hf, Wf = rgb_full.shape[:2]

m = np.array(Image.open(MASK_PATH).convert("L"), dtype=np.uint8)
if m.shape != (Hf, Wf): m = cv2.resize(m, (Wf, Hf), interpolation=cv2.INTER_NEAREST)
allowed_full = (m > 127)

ROI = clamp(ROI, Wf, Hf); SEED = clamp(SEED, Wf, Hf) if SEED is not None else None
if ROI is None: raise RuntimeError("ROI hors image")

# ---------- ROI ----------
x1,y1,x2,y2 = ROI
rgb = rgb_full[y1:y2, x1:x2]
allowed = allowed_full[y1:y2, x1:x2]
H,W = rgb.shape[:2]
if int(allowed.sum()) < 50: raise RuntimeError("Trop peu de pixels autorisés dans la ROI")

# ---------- SEED in ROI ----------
seed_roi = np.zeros((H,W), dtype=bool)
if SEED is not None:
    sx1,sy1,sx2,sy2 = SEED
    sx1 -= x1; sx2 -= x1; sy1 -= y1; sy2 -= y1
    sx1 = max(0, min(sx1, W)); sx2 = max(0, min(sx2, W))
    sy1 = max(0, min(sy1, H)); sy2 = max(0, min(sy2, H))
    if sx2>sx1 and sy2>sy1: seed_roi[sy1:sy2, sx1:sx2] = True
seed_roi &= allowed

# ---------- LUMINANCE (matrix) ----------
f = rgb.astype(np.float32)
Y = 0.2126*f[...,0] + 0.7152*f[...,1] + 0.0722*f[...,2]

# ---------- ROBUST NORMALIZE ----------
Yv = Y[allowed]
med = np.median(Yv); sd = np.std(Yv) + 1e-6
Yn = (Y - med) / sd

# ---------- ANOMALY ----------
bg = np.percentile(Yn[allowed], 30)
dY = np.abs(Yn - bg)

# ---------- TEXTURE σ LOCAL ----------
Yn2 = Yn.copy(); Yn2[~allowed] = bg
k = (WIN, WIN)
mu  = cv2.boxFilter(Yn2, -1, k, normalize=True)
mu2 = cv2.boxFilter(Yn2*Yn2, -1, k, normalize=True)
sig = np.sqrt(np.maximum(mu2 - mu*mu, 0.0))

# ---------- APPLY allowed ----------
dY *= allowed
sig *= allowed

# ---------- THRESHOLDS ----------
dv = dY[allowed]; sv = sig[allowed]
t_core = np.percentile(dv, P_CORE)
t_edge = np.percentile(dv, P_EDGE)
t_tex  = np.percentile(dv, P_TEX)
t_sig  = np.percentile(sv, P_SIG)

# ---------- SEGMENT ----------
core = (dY > t_core) & allowed
edge = ((dY > t_edge) | ((sig > t_sig) & (dY > t_tex))) & allowed

core = morph(core, open_r=OPEN_R, close_r=0)
edge = morph(edge, open_r=0, close_r=EDGE_CLOSE_R)

# ---------- CC + SELECT ----------
if int(core.sum()) < S_MIN: raise RuntimeError("Core trop petit (pas de détection)")
n, lab, stats, _ = cv2.connectedComponentsWithStats((edge.astype(np.uint8)*255), 8)
if n <= 1: raise RuntimeError("Aucune composante")

seed_labels = np.unique(lab[seed_roi]) if seed_roi.any() else np.unique(lab[core])
seed_labels = seed_labels[seed_labels != 0]
if seed_labels.size == 0: raise RuntimeError("Aucune composante ne touche seed/core")

score_px = (dY + W_TEX*sig).astype(np.float32)
flat = lab.reshape(-1)
sum_score = np.bincount(flat, weights=score_px.reshape(-1), minlength=n)
count_px  = np.bincount(flat, minlength=n).astype(np.float32)
mean_score = sum_score / (count_px + 1e-6)
core_cnt = np.bincount(flat, weights=core.reshape(-1).astype(np.float32), minlength=n)

best = 0; best_s = -1e30
for L in seed_labels.astype(int):
    if stats[L, cv2.CC_STAT_AREA] < S_MIN: continue
    if core_cnt[L] < max(10, S_MIN//8): continue
    if mean_score[L] > best_s: best, best_s = L, mean_score[L]
if best == 0: raise RuntimeError("Aucun candidat ne passe les contraintes")

mask_roi = morph((lab == best), open_r=0, close_r=CLOSE_R)
detected = int(mask_roi.sum()) >= S_MIN

# ---------- BUILD FULL MASK + OVERLAY ----------
mask_roi_u8 = (mask_roi.astype(np.uint8)*255)
mask_full_u8 = np.zeros((Hf,Wf), dtype=np.uint8); mask_full_u8[y1:y2, x1:x2] = mask_roi_u8

dbg = rgb_full.copy()
cv2.rectangle(dbg, (x1,y1), (x2,y2), (255,0,0), 2)
if SEED is not None:
    sx1,sy1,sx2,sy2 = SEED
    cv2.rectangle(dbg, (sx1,sy1), (sx2,sy2), (0,255,0), 2)
red = dbg.copy(); red[mask_full_u8 > 0] = (255,0,0)
dbg = cv2.addWeighted(dbg, 0.70, red, 0.30, 0)

roi_overlay = rgb.copy()
roi_overlay[mask_roi] = (255,0,0)

# ---------- SAVE ALL DEBUG IMAGES ----------
Image.fromarray(rgb).save(os.path.join(OUT, "roi_rgb.png"))
save_gray_bool(allowed, os.path.join(OUT, "allowed_roi.png"))
save_heatmap(dY,  os.path.join(OUT, "deltaY.png"), cmap="inferno")
save_heatmap(sig, os.path.join(OUT, "sigma.png"),  cmap="viridis")
save_gray_bool(core, os.path.join(OUT, "core.png"))
save_gray_bool(edge, os.path.join(OUT, "edge.png"))
Image.fromarray(roi_overlay).save(os.path.join(OUT, "overlay_roi.png"))

save_gray_u8(mask_roi_u8, os.path.join(OUT, "mask_fumerolle_roi.png"))
save_gray_u8(mask_full_u8, os.path.join(OUT, "mask_fumerolle_full.png"))
Image.fromarray(dbg).save(os.path.join(OUT, "debug_overlay.png"))

# ---------- SHOW DEBUG FIGURE ----------
if SHOW:
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    ax[0,0].imshow(rgb);        ax[0,0].set_title("ROI");          ax[0,0].axis("off")
    ax[0,1].imshow(dY, cmap="inferno"); ax[0,1].set_title("|ΔY|"); ax[0,1].axis("off")
    ax[0,2].imshow(sig, cmap="viridis");ax[0,2].set_title("σ local");ax[0,2].axis("off")
    ax[0,3].imshow(allowed, cmap="gray");ax[0,3].set_title("Allowed");ax[0,3].axis("off")

    ax[1,0].imshow(core, cmap="gray"); ax[1,0].set_title("Core");  ax[1,0].axis("off")
    ax[1,1].imshow(edge, cmap="gray"); ax[1,1].set_title("Edge");  ax[1,1].axis("off")
    ax[1,2].imshow(roi_overlay);       ax[1,2].set_title("Overlay");ax[1,2].axis("off")

    seed_vis = rgb.copy()
    if seed_roi.any():
        g = np.zeros_like(seed_vis); g[...,1] = (seed_roi.astype(np.uint8)*255)
        seed_vis = cv2.addWeighted(seed_vis, 0.85, g, 0.15, 0)
    ax[1,3].imshow(seed_vis); ax[1,3].set_title("Seed"); ax[1,3].axis("off")

    plt.suptitle(f"Detected={detected} | area={int(mask_roi.sum())} px | best={int(best)}")
    plt.tight_layout()
    plt.show(block=True)

print("Detected:", detected, "| area:", int(mask_roi.sum()), "| best_label:", int(best))
