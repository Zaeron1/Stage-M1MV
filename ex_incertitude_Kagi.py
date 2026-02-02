import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# ============================================================
# MODELE KAGIYAMA
# ============================================================

def kagiyama_model(x, C):
    return C * x**(2/3)

# ============================================================
# ANALYSE D'UNE IMAGE
# ============================================================

def analyze_single_plume(x, h):
    """
    x, h : arrays numpy des points du panache
    retourne : dictionnaire avec C, erreurs, métriques
    """

    # Ajustement
    popt, pcov = curve_fit(kagiyama_model, x, h)
    C = popt[0]
    sigma_C = np.sqrt(pcov[0, 0])

    # Prédiction
    h_pred = kagiyama_model(x, C)
    residuals = h - h_pred

    # Métriques
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((h - np.mean(h))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Proxy thermique
    Q = C**3
    sigma_Q = 3 * C**2 * sigma_C

    return {
        "C": C,
        "sigma_C": sigma_C,
        "Q": Q,
        "sigma_Q": sigma_Q,
        "RMSE": rmse,
        "R2": r2,
        "x": x,
        "h": h,
        "h_pred": h_pred,
        "residuals": residuals
    }

# ============================================================
# EXEMPLE : DONNÉES SYNTHÉTIQUES (POUR ILLUSTRATION)
# ============================================================

np.random.seed(0)

n_images = 20
results = []

for t in range(n_images):
    x = np.linspace(1, 100, 30)
    C_true = 0.05 + 0.005*np.sin(t/4)
    noise = np.random.normal(0, 0.5 + 0.05*t, size=len(x))
    h = C_true * x**(2/3) + noise

    res = analyze_single_plume(x, h)
    res["time"] = t
    results.append(res)

# ============================================================
# TABLE DE SORTIE
# ============================================================

df = pd.DataFrame([{
    "time": r["time"],
    "C": r["C"],
    "sigma_C": r["sigma_C"],
    "Q": r["Q"],
    "sigma_Q": r["sigma_Q"],
    "RMSE": r["RMSE"],
    "R2": r["R2"]
} for r in results])

print(df)

# ============================================================
# PLOTS
# ============================================================

# --- Plot A : exemple panache + modèle ---
r0 = results[0]

plt.figure()
plt.scatter(r0["x"], r0["h"], label="Données")
plt.plot(r0["x"], r0["h_pred"], color="red", label="Kagiyama fit")
plt.xlabel("x")
plt.ylabel("h")
plt.title("Panache + modèle Kagiyama")
plt.legend()
plt.show()

# --- Plot B : résidus ---
plt.figure()
plt.scatter(r0["x"], r0["residuals"])
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("x")
plt.ylabel("Résidus")
plt.title("Résidus du fit")
plt.show()

# --- Plot C : évolution temporelle de C ---
plt.figure()
plt.errorbar(df["time"], df["C"], yerr=df["sigma_C"], fmt="o")
plt.xlabel("Temps")
plt.ylabel("C")
plt.title("Évolution temporelle de C")
plt.show()

# --- Plot D : évolution temporelle de Q ---
plt.figure()
plt.errorbar(df["time"], df["Q"], yerr=df["sigma_Q"], fmt="o")
plt.xlabel("Temps")
plt.ylabel("Q_fum (proxy)")
plt.title("Évolution temporelle de Q_fum")
plt.show()
