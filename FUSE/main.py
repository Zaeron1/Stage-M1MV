# ============================================================
# main.py — Pipeline fumerolles (MÉTÉO VIA CSV)
# ============================================================

import os
import re
import csv
import shutil
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from fuse_detection import chercher_fumerolle

Rect = Tuple[int, int, int, int]

# ============================================================
# CONFIG
# ============================================================

ROOT = os.path.dirname(os.path.abspath(__file__))

IMAGES_DIR = os.path.join(ROOT, "data", "images")
METEO_CSV  = os.path.join(ROOT, "data", "meteo.csv")
MASK_PATH  = os.path.join(ROOT, "data", "mask2.jpg")

OUT_DIR    = os.path.join(ROOT, "out")
OUT_IMG    = os.path.join(OUT_DIR, "images")
OUT_FIG    = os.path.join(OUT_DIR, "figures")
OUT_CSV    = os.path.join(OUT_DIR, "selection.csv")

ROI  = (270, 145, 420, 215)
SEED = (365, 180, 370, 184)

MAX_DT_METEO_S = 15 * 60  # ±15 min

# filtres météo
MAX_WIND_KMH   = 10000
MAX_RAIN_MM    = 10000
MIN_IRRAD_WM2  = 0
T_MIN_C        = 0
T_MAX_C        = 10000

# ============================================================
# UTILS
# ============================================================

def log(msg: str) -> None:
    print(msg, flush=True)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def parse_image_timestamp(fname: str) -> str:
    """
    Extrait YYYYMMDDTHHMM depuis YYYYMMDDTHHMMTU_xxx.jpg
    """
    m = re.search(r"(\d{8}T\d{4})TU", fname)
    if not m:
        raise ValueError("Nom image sans horodatage TU")
    return m.group(1)

def ts_to_datetime(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y%m%dT%H%M")

def safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan

# ============================================================
# METEO VIA CSV
# ============================================================

def load_meteo_csv(path: str):
    log(f"[LOAD] météo CSV : {path}")

    times = []
    data = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row["timestamp"]
            dt = ts_to_datetime(ts)
            times.append(dt)
            data.append({
                "dt": dt,
                "Tair":  safe_float(row["T_air"]),
                "Irrad": safe_float(row["Irrad"]),
                "Vvent": safe_float(row["V_vent"]),
                "Rain":  safe_float(row["Pluie"]),
            })

    times = np.array(times)

    log(f"[OK] {len(times)} lignes météo CSV")
    log(f"     première : {data[0]['dt']}")
    log(f"     dernière : {data[-1]['dt']}")

    return times, data

def nearest_meteo(dt_img: datetime, meteo_times, meteo_data):
    dts = np.array([abs((t - dt_img).total_seconds()) for t in meteo_times])
    i = int(np.argmin(dts))
    return meteo_data[i], dts[i]

def meteo_pass(m: Dict[str, Any]) -> Tuple[bool, str]:
    if np.isnan(m["Vvent"]) or m["Vvent"] > MAX_WIND_KMH:
        return False, "vent"
    if np.isnan(m["Rain"]) or m["Rain"] > MAX_RAIN_MM:
        return False, "pluie"
    if np.isnan(m["Irrad"]) or m["Irrad"] < MIN_IRRAD_WM2:
        return False, "irradiance"
    if np.isnan(m["Tair"]) or not (T_MIN_C <= m["Tair"] <= T_MAX_C):
        return False, "température"
    return True, "OK"

# ============================================================
# FIGURES
# ============================================================

def save_debug(res: Dict[str, Any], out_png: str) -> None:
    dbg = res.get("debug")
    if dbg is None:
        return

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(dbg["rgb"]); ax[0].set_title("RGB ROI"); ax[0].axis("off")
    ax[1].imshow(dbg["dY"], cmap="inferno"); ax[1].set_title("dY"); ax[1].axis("off")
    ax[2].imshow(dbg["roi_overlay"]); ax[2].set_title("Overlay"); ax[2].axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# ============================================================
# MAIN
# ============================================================

def main() -> None:
    log("========== START ==========")

    ensure_dir(OUT_DIR)
    ensure_dir(OUT_IMG)
    ensure_dir(OUT_FIG)

    meteo_times, meteo_data = load_meteo_csv(METEO_CSV)

    images = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".png"))
    ])

    log(f"[OK] {len(images)} images trouvées")

    rows = []

    for fname in images:
        img_path = os.path.join(IMAGES_DIR, fname)

        try:
            ts_img = parse_image_timestamp(fname)
            dt_img = ts_to_datetime(ts_img)
        except Exception as e:
            log(f"[SKIP] {fname} : {e}")
            continue

        m, dt_sec = nearest_meteo(dt_img, meteo_times, meteo_data)
        time_ok = dt_sec <= MAX_DT_METEO_S
        met_ok, met_reason = meteo_pass(m)

        log(f"[CHECK] {fname}")
        log(f"        img = {dt_img}")
        log(f"        met = {m['dt']}  Δt={int(dt_sec)}s")
        log(f"        time_ok={time_ok}  met_ok={met_ok} ({met_reason})")

        detected = 0
        area_px = 0
        err = ""

        if time_ok and met_ok:
            try:
                res = chercher_fumerolle(
                    img_path=img_path,
                    mask_path=MASK_PATH,
                    ROI=ROI,
                    SEED=SEED,
                    return_debug=True,
                )

                detected = int(res["detected"])
                area_px = int(res["area_px"])

                if detected:
                    shutil.copy2(img_path, os.path.join(OUT_IMG, fname))
                    save_debug(res, os.path.join(OUT_FIG, fname.replace(".jpg", "_diag.png")))

            except Exception as e:
                err = str(e)
                log(f"[ERR] détection : {err}")

        rows.append({
            "image": fname,
            "img_utc": dt_img.isoformat(),
            "meteo_utc": m["dt"].isoformat(),
            "dt_s": int(dt_sec),
            "Tair": m["Tair"],
            "Irrad": m["Irrad"],
            "Vvent": m["Vvent"],
            "Rain": m["Rain"],
            "time_ok": int(time_ok),
            "meteo_ok": int(met_ok),
            "meteo_reason": met_reason,
            "detected": detected,
            "area_px": area_px,
            "error": err,
        })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    log(f"[OK] CSV écrit : {OUT_CSV}")
    log("========== END ==========")

if __name__ == "__main__":
    main()
