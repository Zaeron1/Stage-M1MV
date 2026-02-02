# CAGED — Visible Volcanic Plume Detection

This repository implements **CAGED** (*Contrast-based Anomaly Growth for Exhalation Detection*),
a **non-AI**, **fully explainable** method to detect and segment **volcanic plumes in visible images**.

The method is designed for **fixed cameras** and **long time series**, and is suitable for scientific studies and monitoring applications.

---

## Principle

A visible volcanic plume is detected as a **statistically significant luminance anomaly** that is:

* spatially coherent,
* connected,
* and distinct from a stable background.

The segmentation is performed in **two steps**:

1. **Core detection** (high-confidence anomaly)
2. **Conditional growth** toward diffuse plume edges

---

## Processing pipeline

1. **RGB → Luminance conversion**
   Converts the image to grayscale luminance to remove color dependence.

2. **Local normalization**
   Removes global illumination variations (daylight, exposure).

3. **Background estimation**
   The background luminance is estimated using a low percentile of the ROI.

4. **Luminance anomaly map**
   Absolute deviation from the background is computed.

5. **Core plume detection**
   Pixels with strong anomalies are selected as plume seeds.

6. **Conditional region growth**
   The plume is extended from the core to connected, lower-contrast pixels.

7. **Morphological cleaning**
   Removes noise and enforces spatial coherence.

8. **Quantitative outputs**

   * plume area (pixels)
   * mean luminance anomaly
   * optional texture diagnostics

---

## Outputs

* Binary plume mask
* Debug figures (ROI, anomaly map, core mask, final segmentation)
* Simple plume presence flag (True / False)

---

## Dependencies

* numpy
* pillow
* scikit-image
* matplotlib

Install with:

```
pip install -r requirements.txt
```

---


