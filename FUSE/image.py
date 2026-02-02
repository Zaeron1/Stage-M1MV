import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# ============================
# CONFIG
# ============================
IMAGE_PATH = "/Users/alexandremichaux/Documents/UCA/Cours/Stage/projet/Code/Stage-M1/FUSE/images/image2.png"  # adapte si besoin

# ============================
# STATE
# ============================
rectangles = []
press_event = None
current_rect = None

# ============================
# CALLBACKS
# ============================
def on_press(event):
    global press_event, current_rect
    if event.inaxes is None:
        return
    press_event = (int(event.xdata), int(event.ydata))
    current_rect = patches.Rectangle(
        press_event, 0, 0,
        linewidth=2, edgecolor="red", facecolor="none"
    )
    ax.add_patch(current_rect)
    fig.canvas.draw()


def on_motion(event):
    global current_rect
    if press_event is None or event.inaxes is None:
        return

    x0, y0 = press_event
    x1, y1 = int(event.xdata), int(event.ydata)

    x = min(x0, x1)
    y = min(y0, y1)
    w = abs(x1 - x0)
    h = abs(y1 - y0)

    current_rect.set_xy((x, y))
    current_rect.set_width(w)
    current_rect.set_height(h)

    fig.canvas.draw_idle()


def on_release(event):
    global press_event, current_rect
    if press_event is None or event.inaxes is None:
        return

    x0, y0 = press_event
    x1, y1 = int(event.xdata), int(event.ydata)

    x_min, x_max = sorted([x0, x1])
    y_min, y_max = sorted([y0, y1])

    rectangles.append((x_min, y_min, x_max, y_max))
    print(f"RECTANGLE = ({x_min}, {y_min}, {x_max}, {y_max})")

    press_event = None
    current_rect = None


# ============================
# MAIN
# ============================
img = Image.open(IMAGE_PATH)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(img)
ax.set_title("Click & drag to draw rectangles")
ax.axis("off")

fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("motion_notify_event", on_motion)
fig.canvas.mpl_connect("button_release_event", on_release)

plt.show()

print("\nALL RECTANGLES:")
for r in rectangles:
    print(r)
