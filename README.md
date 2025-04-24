# Fruit-Detection Project — Faster R-CNN with ResNet-34

End-to-end implementation of an object-detection pipeline that locates fruits in images.  
Key components:

* **Model**  – Faster R-CNN backbone = ResNet-34 (ImageNet weights)  
* **Custom anchors**  – sizes (32, 64, 128), aspect-ratios (0.8, 1.0, 1.25) to match near-square fruit shapes  
* **Automatic box generation**  – watershed-based conversion of segmentation masks → bounding boxes + NMS  
* **Rich data augmentation**  – flip, rotate (0 / 90 / 180 / 270 °) and optional color-jitter  
* **Metric suite**  – mAP, AP, FP/FN counts, precision-recall curves at IoU {0.3, 0.5, 0.7, 0.9}

---


Each mask is a single-channel PNG with fruit pixels = 255, background = 0.

---

## Training & Experiments

* **Optimizer**    SGD (lr = 0.005, momentum = 0.9, wd = 5 × 10⁻⁴)  
* **Batch size**   4  
* **Epochs**       5 per experiment  
* **Resolutions tested**  native, 300 × 300, 512 × 512  
* **Augmentation variants**

| Tag | Geometry | Color-jitter |
|-----|----------|--------------|
| NoAug | ✗ | ✗ |
| FlipRotate | ✓ | ✗ |
| FlipRotate + Color | ✓ | ✓ |

---

## Results (mAP@0.5)

| Aug | Res | Conf thr | mAP | AP | FP | FN |
|-----|-----|----------|-----|----|----|----|
| NoAug | native | 0.3 | **0.053** | 0.098 | 17 076 | 7 916 |
| NoAug | native | 0.7 | 0.198 | 0.185 | **379** | 8 811 |
| FlipRotate | native | 0.7 | **0.319** | **0.360** | 348 | 8 740 |
| FlipRotate + Color | native | 0.7 | 0.247 | 0.425 | 299 | 8 776 |

**Highlights**

* Flip + Rotate augmentation alone lifted mAP from **0.053 → 0.319** (6×) at high-precision threshold.
* Adding color-jitter kept high precision (299 FP) while boosting AP to **0.425**.
* Smaller (300²) or larger (512²) re-scales were less effective than training at original resolution.

---

## Quick Start

```bash
pip install torch torchvision opencv-python pillow matplotlib scikit-learn tqdm
python Fruit Detection.py          # runs training + evaluation with default settings

