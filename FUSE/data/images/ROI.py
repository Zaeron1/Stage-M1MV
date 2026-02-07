import cv2
import matplotlib.pyplot as plt

# === Charger l'image ===
image_path = "/Users/alexandremichaux/Documents/UCA/Cours/Stage/projet/Stage-M1MV/FUSE/data/images/20231220T1309TU_GPVHOU2_br.jpg"   # mettre le chemin de ton image ici
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# === Fonction appelée quand la souris bouge ===
def on_move(event):
    if event.xdata is not None and event.ydata is not None:
        x = int(event.xdata)
        y = int(event.ydata)
        print(f"x={x}, y={y}")

# === Affichage ===
fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title("Bouge la souris sur l'image pour voir les coordonnées")

# === Connexion de l'événement souris ===
fig.canvas.mpl_connect("motion_notify_event", on_move)

plt.show()
