# CSE 445 — Assignment 2: Full Notebook Explanation
## Cow Pose Estimation & Multi-Object Tracking

---

## Overview: What Is This Notebook Doing?

This notebook builds a complete **animal pose estimation and tracking system** for cows. At a high level:
1. It downloads a dataset of cow images with 26 body landmark (keypoint) annotations
2. It fixes broken labels in the dataset
3. It artificially multiplies the training data using augmentation
4. It trains a YOLO neural network to detect cows AND predict where each of their 26 body joints are
5. It evaluates how well the model performs
6. It runs two different object trackers (ByteTrack, BoT-SORT) on a video to follow individual cows over time
7. It compares both trackers and analyzes which keypoints the model predicts with highest confidence

---

## Section 1 — Package Installation

```python
%pip install ultralytics --quiet          # YOLOv26s training & inference
%pip install supervision --quiet          # ByteTrack / BoT-SORT wrapper + annotators
%pip install lapx --quiet                 # C++ LAP solver required by ByteTrack
%pip install albumentations --quiet
%pip install pandas matplotlib seaborn opencv-python tqdm pyyaml --quiet
```

**What it does:** Installs all required Python libraries before anything else runs.

**Why these specific packages?**
- `ultralytics`: The official package for YOLOv8/YOLO11/YOLOv26 models. It handles training, inference, and evaluation in one clean API.
- `supervision`: A computer vision utility library that wraps trackers (ByteTrack, BoT-SORT) and provides annotators for drawing boxes and tracks.
- `lapx`: ByteTrack's Hungarian algorithm for matching detected objects across frames requires a C++ Linear Assignment Problem (LAP) solver. Without `lapx`, ByteTrack simply won't run.
- `albumentations`: A high-performance image augmentation library that is "keypoint-aware" — when you flip or rotate an image, it also correctly transforms the keypoint coordinates.

---

## Section 2 — Library Imports & Global Configuration

```python
import os, json, random, shutil, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2
from PIL import Image
import torch
from ultralytics import YOLO
import albumentations as A
import supervision as sv
from tqdm import tqdm
```

### The matplotlib style settings:
```python
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
CMAP_CONF = LinearSegmentedColormap.from_list("conf", ["#e74c3c", "#f39c12", "#2ecc71"])
```

**Why:** Sets a consistent, clean visual style for all plots. `CMAP_CONF` creates a custom red→orange→green colormap used whenever we want to color-code something by confidence score (red = low confidence, green = high).

### Device Detection:
```python
DEVICE = "0" if torch.cuda.is_available() else "cpu"
```

**Why:** If a GPU is available (Kaggle gives a T4/P100), training uses it (device "0" = first GPU). Without GPU, training would take 10-50× longer.

### Output Folder Tree:
```python
subdirs = {
    "dataset": RESULTS_DIR / "01_dataset",
    "augment": RESULTS_DIR / "02_augmentation",
    "training": RESULTS_DIR / "03_training",
    ...
}
for d in subdirs.values():
    d.mkdir(parents=True, exist_ok=True)
```

**Why:** Creates a clean numbered folder structure upfront so every plot and CSV has a clear home. `exist_ok=True` means it won't crash if the folder already exists.

---

## Section 3 — Dataset Setup & Configuration

This is the most technically complex section. Several things happen here, including fixing a critical data quality problem.

### Keypoint Definitions:

```python
KEYPOINT_NAMES = ["L_Eye", "R_Eye", "Chin", "R_F_Hoof", ...]  # 26 names
NUM_KP = 26

FLIP_IDX = [1,0,2,4,3,6,5,7,9,8,11,10,12,13,15,14,17,16,18,19,21,20,22,23,24,25]
```

**What is `FLIP_IDX`?** When you horizontally flip a cow image for augmentation, left and right swap. `FLIP_IDX[i]` tells YOLO: "after a horizontal flip, keypoint `i` should swap with keypoint `FLIP_IDX[i]`." For example, L_Eye (index 0) swaps with R_Eye (index 1). Without this, the model would learn wrong anatomy after flipped images.

### KP_SIGMAS:
```python
KP_SIGMAS = [0.025, 0.025, 0.025, 0.070, 0.070, ...]
```

**What are sigmas?** These control how strictly the model is penalized for imprecise keypoint predictions during evaluation (OKS — Object Keypoint Similarity). Small sigma (0.025) = high precision required (eyes, nose — small, precise landmarks). Large sigma (0.070) = some slack allowed (hooves — large contact areas, harder to pin exactly). These values were chosen to match the anatomical size/precision of each landmark.

### The Critical Bug Fix — Zero-Size Bounding Boxes:

```python
def fix_label_file_inplace(path, num_kp, padding=0.05):
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        kp_data = parts[5:]
        if bw == 0 or bh == 0:   # ← THE PROBLEM
            xs, ys = [], []
            for i in range(0, num_kp * 3, 3):
                x = float(kp_data[i]); y = float(kp_data[i+1]); v = int(float(kp_data[i+2]))
                if v > 0 and x > 0 and y > 0:
                    xs.append(x); ys.append(y)
            if len(xs) >= 2:
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                bw = min((x_max - x_min) + padding * 2, 1.0)
                bh = min((y_max - y_min) + padding * 2, 1.0)
                cx = float(np.clip((x_min + x_max) / 2, bw/2, 1 - bw/2))
                cy = float(np.clip((y_min + y_max) / 2, bh/2, 1 - bh/2))
```

**The Problem:** Some label files in the dataset had `bw=0` and `bh=0` — meaning the bounding box had zero area. YOLO needs a valid bounding box to train on. These zero-size boxes cause NaN losses, crash training, or silently corrupt the model.

**Why this happens:** The original annotation tool likely only stored keypoints and left the bounding box blank for some images.

**The Fix:** If a box is zero-size, we *reconstruct* it from the keypoint coordinates. We find the min/max x and y of all visible keypoints, compute a bounding box around them, and add 5% padding on each side so the box has a small margin. `np.clip` ensures the center doesn't go out of bounds.

**Why this method?** It's the most faithful reconstruction possible without human re-annotation — the box is derived directly from where the actual body parts are.

### Dataset YAML:
```python
with open(DATA_YAML_PATH, "w") as f:
    yaml.dump({
        "path": str(DATASET_ROOT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["cow"],
        "kpt_shape": [NUM_KP, 3],
        "flip_idx": FLIP_IDX,
    }, f)
```

**Why `kpt_shape: [26, 3]`?** Each keypoint has 3 values: `(x, y, visibility)`. Visibility is 0 (absent), 1 (occluded), or 2 (fully visible). YOLO needs to know this shape to build the correct output head.

---

## Section 4 — Dataset Visualization

### Annotation Grid (3×3 samples):
```python
def parse_yolo_pose_label(label_path, img_w, img_h):
    # reads normalized YOLO coords and converts to pixel coords
    kps = [(int(kp_raw[i]*img_w), int(kp_raw[i+1]*img_h), int(kp_raw[i+2]))
           for i in range(0, NUM_KP*3, 3)]

def draw_pose_on_image(img_bgr, instances, bboxes=None):
    # draws skeleton lines between connected keypoints
    for i,j in SKELETON:
        if vi>0 and vj>0:  # only draw if both endpoints are visible
            cv2.line(out, (xi,yi), (xj,yj), (60,140,255), 2)
    # green dot = fully visible (v=2), orange dot = occluded (v=1)
    col = (0,220,80) if v==2 else (255,140,0)
    cv2.circle(out, (x,y), 5, col, -1)
```

**What this produces:** A 3×3 grid of training images with annotated keypoints and skeleton overlays drawn on top, so you can visually verify the labels look correct before training.

### Keypoint Visibility Distribution:
```python
vis_counts = np.zeros(NUM_KP, dtype=int)  # v=2 count per keypoint
occ_counts = np.zeros(NUM_KP, dtype=int)  # v=1 count per keypoint
abs_counts  = np.zeros(NUM_KP, dtype=int)  # v=0 count per keypoint
```

**What this tells you:** For each of the 26 keypoints, what percentage of the time is it fully visible vs. occluded vs. absent? This is critical because keypoints that are rarely visible (e.g., hooves when a cow is standing in grass) will be harder for the model to learn.

### BBox Distribution:
Three histograms showing the distribution of bounding box **width**, **aspect ratio (W/H)**, and **area** across all instances. This tells you what object scales and shapes the model needs to handle, which affects choices like anchor sizes and input resolution.

---

## Section 5 — Pose-Aware Augmentation Pipeline

### Why Augmentation?

The dataset has a limited number of images. Training on them repeatedly risks overfitting (the model memorizes training images instead of learning generalizable features). Augmentation artificially creates new training images by applying random transformations.

**The key challenge:** Normal augmentation (e.g., from torchvision) only transforms the image. If you flip an image, you must also flip the keypoints. If you rotate, you must rotate keypoints too. `albumentations` handles this automatically.

### The Augmentation Pipeline:
```python
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),                           # mirror left-right 50% of the time
    A.RandomBrightnessContrast(p=0.7),                 # simulate different lighting
    A.HueSaturationValue(p=0.5),                       # simulate color temperature changes
    A.GaussNoise(p=0.3),                               # simulate camera sensor noise
    A.GaussianBlur(p=0.2),                             # simulate motion blur / defocus
    A.RandomShadow(p=0.3),                             # simulate shadows from trees, fences
    A.ShiftScaleRotate(rotate_limit=10, p=0.5),        # slight perspective shifts
    A.CoarseDropout(num_holes_range=(1,4), p=0.2),     # simulate occlusion patches
    A.LongestMaxSize(max_size=640),                    # resize to model input size
    A.PadIfNeeded(min_height=640, min_width=640,       # letterbox padding (gray)
                  value=(114,114,114)),
],
keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
bbox_params=A.BboxParams(format="yolo", min_visibility=0.0))
```

**Why each transform:**
- `HorizontalFlip`: Cows appear from both sides. This doubles effective dataset size for free.
- `RandomBrightnessContrast` + `HueSaturationValue`: Outdoor lighting varies enormously (morning vs. noon, cloudy vs. sunny). This makes the model robust to that.
- `GaussNoise` + `GaussianBlur`: Real cameras add noise and blur, especially in video.
- `RandomShadow`: Farm environments have tree shadows and fence patterns.
- `ShiftScaleRotate`: Cows appear at different angles and distances from the camera.
- `CoarseDropout`: Simulates partial occlusion (another cow blocking part of the body). Random patches are blacked out.
- `LongestMaxSize` + `PadIfNeeded`: Resizes to 640×640 with gray padding — this is the standard "letterboxing" approach used by YOLO. Gray (114,114,114) is YOLO's conventional padding color.

**`remove_invisible=False`:** When a keypoint gets shifted out of frame, we keep it as (0,0,0) rather than removing it, so the label format stays consistent (always 26 keypoints per instance).

### AUG_MULTIPLIER = 8:
```python
tr_n = apply_augmentation_split("train", train_transform, AUG_MULTIPLIER)
```
**This means:** Every training image is augmented 8 times, creating 8 different versions. If you had 500 training images, you now have ~4000. The validation set is NOT augmented (only resized) because we want to evaluate on realistic images.

---

## Section 6 — Model Training

```python
model = YOLO("yolo26s-pose.pt")  # load pretrained weights (transfer learning)

train_results = model.train(
    data         = str(AUG_YAML.resolve()),
    epochs       = 100,
    patience     = 30,       # early stopping: stop if no improvement for 30 epochs
    imgsz        = 640,
    batch        = 16,
    device       = DEVICE,
    optimizer    = "AdamW",
    lr0          = 3e-4,     # initial learning rate
    lrf          = 0.01,     # final lr = lr0 × lrf (cosine annealing)
    weight_decay = 5e-4,     # L2 regularization to prevent overfitting
    warmup_epochs= 5,        # gradually increase lr from 0 for first 5 epochs
    amp          = True,     # mixed precision (fp16) — faster training on GPU
    pose         = 24.0,     # weight of keypoint loss in total loss
    box          = 5.0,      # weight of bounding box loss
    mosaic       = 1.0,      # mosaic augmentation (4 images merged into 1)
    fliplr       = 0.5,      # YOLO's own horizontal flip (in addition to albumentations)
)
```

**Why YOLOv26s-pose?** YOLO (You Only Look Once) is a single-stage detector — it predicts boxes and keypoints in one forward pass, making it much faster than two-stage methods. The "s" means "small" — smaller model, faster inference, but slightly lower accuracy than "m" or "l". Good tradeoff for real-time video tracking.

**Transfer learning (`yolo26s-pose.pt`):** We start from weights pretrained on a large general dataset (COCO). The model already knows how to detect animals and predict pose. We fine-tune it specifically on cows. This requires far less data and training time than training from scratch.

**Why AdamW?** AdamW is Adam with decoupled weight decay. It adapts the learning rate per-parameter and is more stable than SGD for fine-tuning. `lr0=3e-4` is a safe starting point for fine-tuning (much smaller than training from scratch).

**Why `pose=24.0` and `box=5.0`?** These are loss weights. Since the assignment is specifically about pose estimation, we weight the keypoint loss much higher (24×) so the model prioritizes getting keypoints right even at the cost of slightly less precise boxes.

**`patience=30`:** Early stopping. If validation mAP doesn't improve for 30 consecutive epochs, training stops automatically. This prevents wasted compute and overfitting.

**`amp=True`:** Automatic Mixed Precision. Uses float16 (half precision) where possible. Halves memory use and typically speeds up training by 1.5-2× on modern GPUs with almost no accuracy loss.

**The path issue that was fixed:**
```python
RUN_DIR = Path("runs/pose/runs") / RUN_NAME
```
Ultralytics internally prepends "pose" to the project directory. Since we set `project="runs"`, the actual save path becomes `runs/pose/runs/<run_name>`. This was discovered and corrected from an earlier version where the wrong path was used to load `best.pt`.

---

## Section 7 — Training Results: Loss & Metric Curves

```python
df_train = pd.read_csv(csv_path)
df_train.columns = df_train.columns.str.strip()  # remove whitespace from column names
```

**Why `.str.strip()`?** Ultralytics sometimes writes column names with leading spaces (e.g., `" metrics/mAP50(B)"`). Stripping ensures we can reliably find columns by name.

### Plot 1 — Loss Curves (3 panels: Box Loss, Pose Loss, Cls Loss):
```python
ax.axvline(best_ep, color="gray", ls=":", label=f"Best ep={best_ep}")
```

**What the losses mean:**
- **Box Loss**: How wrong the predicted bounding box coordinates are (center x, y, width, height). Lower = better box localization.
- **Pose Loss (Keypoint Loss)**: How wrong the predicted keypoint positions are. This is the most important loss for our task.
- **Cls Loss (Classification Loss)**: How confident the model is in assigning the "cow" class. Should drop quickly since there's only one class.

**What healthy training looks like:** Both train and val loss decrease together. If val loss starts rising while train loss keeps dropping, the model is overfitting.

### Plot 2 — mAP Curves:
**mAP@50**: Mean Average Precision at 50% IoU threshold. A prediction is "correct" if its box/keypoints overlap the ground truth by ≥50%.
**mAP@50-95**: Average mAP across IoU thresholds from 50% to 95% in 5% steps. Much stricter metric. This is the standard COCO benchmark.

**Pose mAP** is computed using OKS (Object Keypoint Similarity) instead of box IoU. OKS accounts for the fact that different keypoints have different expected precision (using the sigma values defined in Section 3).

### Plot 3 — Precision & Recall:
- **Precision**: Of all detections the model made, what fraction were actually correct cows? (Measures false positives)
- **Recall**: Of all actual cows in the images, what fraction did the model find? (Measures missed detections)

**The tradeoff:** Higher confidence threshold → higher precision, lower recall. Lower threshold → lower precision, higher recall.

### Plot 4 — Learning Rate Schedule:
Shows the cosine annealing decay from `lr0=3e-4` down to `lr0 × lrf = 3e-6`. After the 5-epoch warmup (linear ramp from 0), the LR smoothly decreases following a cosine curve. This prevents the optimizer from overshooting the optimal weights in later epochs.

### Plot 5 — Training Dashboard (3×3 grid):
A composite view of all 9 key training signals in one figure: train/val box loss, train/val pose loss, train/val cls loss, and all four mAP variants.

---

## Section 8 — Validation Metrics: Full Report Card

```python
val_metrics = best_model.val(data=str(AUG_YAML.resolve()), plots=True, verbose=False)

metrics_dict = {
    "mAP50 (pose)":    val_metrics.pose.map50,
    "mAP50-95 (pose)": val_metrics.pose.map,
    "mAP50 (box)":     val_metrics.box.map50,
    "mAP50-95 (box)":  val_metrics.box.map,
    "Precision (box)": val_metrics.box.mp,
    "Recall (box)":    val_metrics.box.mr,
    "F1 (box)":        2 * P * R / (P + R),
}
```

**Why run `.val()` separately?** The training loop reports approximate metrics during training (on mini-batches). The `.val()` call runs a full, precise evaluation on the entire validation set using the best saved weights.

**F1 Score:** Harmonic mean of Precision and Recall. More informative than either alone because it penalizes extreme imbalances (a model that only ever predicts "cow" has 100% recall but near-0% precision; F1 captures this failure).

**Interpreting the bar chart:** Bars are colored by the `CMAP_CONF` colormap (red→green). Reference lines at 0.5 (minimum acceptable) and 0.75 (good performance). Values above 0.75 in green indicate the model is performing well on that metric.

---

## Section 9 — Qualitative Inference on Validation Images

```python
CONF_THRESH = 0.30  # only show detections with >30% confidence

preds = best_model.predict(bgr, conf=CONF_THRESH, verbose=False)[0]
```

**Why `conf=0.30`?** Lower threshold → more detections but more false positives. 0.30 is a reasonable starting point for evaluation. In production you'd tune this based on precision/recall requirements.

```python
def draw_pose_results(frame_bgr, result, track_id=-1):
    color = PALETTE[track_id % len(PALETTE)] if track_id>=0 else (0,220,80)
    # draw skeleton lines only where both endpoints have conf > 0.3
    if conf[i]>0.3 and conf[j]>0.3:
        cv2.line(out, (xi,yi), (xj,yj), col, 2)
    # draw keypoints only where conf > 0.25
    if conf[idx]>0.25:
        cv2.circle(out, (x,y), 5, col, -1)
```

**Why different thresholds for lines vs. dots?** Skeleton lines are only drawn between two confident points, so both need the higher 0.3 threshold. Individual dots are drawn with a slightly lower threshold (0.25) to show more keypoints even if connections aren't drawn.

### Confidence Distribution Plot:
```python
all_box_conf.extend(preds.boxes.conf.cpu().numpy().tolist())
kc = preds.keypoints.conf.cpu().numpy()  # shape: (N_instances, 26)
all_pose_avg_conf.extend(kc.mean(axis=1).tolist())  # average over 26 keypoints
```

**Box confidence histogram:** Shows the distribution of detection confidence scores. Ideally most detections cluster near 1.0 (very confident). A flat distribution suggests the model is uncertain.

**Keypoint confidence histogram:** For each detected cow, the mean confidence across all 26 keypoints. A mean near 0.5-0.7 is typical — some keypoints are easy (body center, spine) and some are hard (hooves, hidden ear).

---

## Section 10 — ByteTrack

**What is tracking?** Detection (Section 9) finds cows in a single image. Tracking assigns consistent IDs to cows across video frames so you can follow individual animals over time.

### ByteTrack Configuration:
```python
BYTETRACK_CFG = dict(
    track_activation_threshold = 0.25,  # min confidence to start a new track
    lost_track_buffer          = 50,    # keep a "lost" track alive for 50 frames before deleting
    minimum_matching_threshold = 0.80,  # IoU overlap needed to match detection to existing track
    frame_rate                 = 30,
)
```

**How ByteTrack works:** ByteTrack is named for its key innovation — it uses *both* high-confidence AND low-confidence detections. Most trackers discard low-confidence detections. ByteTrack keeps them as "secondary" detections and uses them to rescue tracks that are about to be lost (e.g., a cow partially occluded behind a fence). This dramatically reduces ID switches.

**`lost_track_buffer=50`:** If a tracked cow disappears (walks behind another cow, exits frame), ByteTrack keeps the track "alive" for 50 frames. If the cow reappears within that window, it gets the same ID back. 50 frames at 30fps = 1.67 seconds.

### The `run_tracker_video` function:
```python
for fi in tqdm(range(n_proc)):
    ret, bgr = cap.read()
    res  = model.predict(bgr, conf=conf_thresh, verbose=False)[0]   # detect
    dets = sv.Detections.from_ultralytics(res)                      # convert format
    dets = tracker_obj.update_with_detections(dets)                 # assign track IDs

    # Trail drawing: each cow gets a colored "tail" showing its recent path
    trails[tid].append((cx,cy))
    pts = list(trails[tid])
    for k in range(1, len(pts)):
        a = k/len(pts)                    # alpha increases toward current position
        cv2.line(out, pts[k-1], pts[k], tuple(int(c*a) for c in col), 2)
```

**The trail effect:** Each cow's centroid history (up to 50 frames) is drawn as a fading colored line. Older positions are dimmer (alpha closer to 0), newer positions brighter. This makes movement paths visually clear.

### Statistics logged:
- **unique_ids**: Total number of different IDs assigned. Ideally this equals the true number of cows in the video. Higher numbers mean ID switches (same cow got a new ID after being lost).
- **avg_tracks_per_frame**: Average number of cows being tracked simultaneously.
- **id_switches_est**: Estimated number of times a track "gaps" (disappears for ≥1 frame then reappears with the same ID). This is an approximation — true ID switch counting requires ground truth.
- **track_continuity**: `avg_tracks_per_frame / unique_ids`. Score of 1.0 = perfect (every ID is active every frame). Lower = more fragmented tracks.

---

## Section 11 — BoT-SORT

```python
use_sv_botsort = hasattr(sv, "BotSort")
if use_sv_botsort:
    bs_tracker = sv.BotSort(**BOTSORT_CFG)
else:
    # Ultralytics native BoT-SORT via model.track(tracker='botsort.yaml')
    res_list = best_model.track(bgr, conf=CONF_VIDEO, tracker="botsort.yaml",
                                 persist=True)
```

**Why the fallback?** Different versions of the `supervision` library may or may not have `BotSort` built in. The code gracefully falls back to YOLO's own built-in BoT-SORT tracker if supervision doesn't have it.

**How BoT-SORT differs from ByteTrack:** BoT-SORT adds two improvements:
1. **Camera motion compensation (CMC):** Uses image registration to correct for camera movement before matching. This is crucial when the camera is panning or zooming, because object positions shift due to camera motion rather than object motion.
2. **Re-ID (Re-Identification) features:** Uses appearance embeddings (what the cow looks like) in addition to position to match tracks. This helps recover tracks after longer occlusions where IoU matching fails.

**`persist=True`:** In YOLO's native tracker, this tells the model to maintain tracking state between calls (so the tracker remembers previous frames).

---

## Section 12 — Tracker Comparison

### Quantitative Table:
```python
metric_keys = ["unique_ids", "avg_tracks_per_frame", "max_tracks_per_frame",
               "id_switches_est", "track_continuity", "mean_box_conf"]
```

**How to interpret:**
- **Unique IDs ↓**: Fewer is better — means fewer identity fragmentation events
- **ID Switches ↓**: Fewer is better — more stable identities
- **Track Continuity ↑**: Higher is better — tracks persist longer
- **Mean Box Confidence**: Both trackers use the same detector, so this should be similar

### Radar Chart (Spider Plot):
```python
bt_radar = [1 - bt_stats["unique_ids"]/uid_max,   # inverted: fewer IDs = better
            bt_stats["track_continuity"],
            bt_stats["max_tracks_per_frame"]/mc_max,
            bt_stats["avg_tracks_per_frame"]/at_max,
            bt_stats["mean_box_conf"]]
```

**Why invert unique_ids?** On the radar chart, larger area = better. Since fewer unique IDs is better (less fragmentation), we invert it as `1 - (value/max)`. This makes all dimensions point "outward = better."

**What the radar shows at a glance:** If ByteTrack's polygon is larger than BoT-SORT's, ByteTrack performed better overall. If they have different strengths in different dimensions (e.g., ByteTrack has better continuity but BoT-SORT has fewer ID switches), the radar makes that tradeoff visually obvious.

**Active tracks over time plots (Panels 5 & 6):** Show how many cows are being tracked frame-by-frame. Sharp drops indicate frames where the model missed detections (challenging lighting, occlusion, fast motion). Flat lines near a constant value indicate stable tracking.

---

## Section 13 — Keypoint Analysis

```python
for ip in val_imgs_kp:
    pred = best_model.predict(bgr, conf=0.20, verbose=False)[0]
    kc = pred.keypoints.conf.cpu().numpy()   # shape: (N_instances, 26)
    all_kp_conf.append(kc)

kp_conf_all = np.concatenate(all_kp_conf, axis=0)   # (total_instances, 26)
mean_conf   = kp_conf_all.mean(axis=0)              # mean for each of 26 keypoints
std_conf    = kp_conf_all.std(axis=0)               # variability
pct_high    = (kp_conf_all > 0.5).mean(axis=0) * 100  # % of time confidence > 0.5
```

**Why `conf=0.20` here?** Lower than the inference threshold (0.30) because we want to capture all detections to get a full picture of keypoint difficulty, including borderline detections.

### Plot 1 — Per-Keypoint Confidence Bar with Error Bars:
Each bar's height = mean model confidence for that keypoint. Error bars = standard deviation. Colors go from red (low confidence) to green (high confidence) using `CMAP_CONF`.

**What the error bars tell you:** Large error bars mean the model is very inconsistent on that keypoint — sometimes confident, sometimes not. This suggests the keypoint is hard to reliably see (e.g., Tail_Tip that might be wagging or hidden).

### Plot 2 — Body Region Grouped Heatmap:
```python
KP_REGIONS = {
    "Head / Face" : [0, 1, 2, 20, 21, 22, 23, 24],
    "Neck / Spine": [18, 13, 12, 7, 25, 19],
    "Front Legs"  : [8, 9, 14, 15, 3, 4],
    "Back Legs"   : [10, 11, 16, 17, 5, 6],
}
```

**What this shows:** Groups the 26 keypoints into 4 anatomical regions and computes average confidence for each. This answers questions like "Is the model better at the head or the legs?" — which is useful for understanding where more training data or annotation effort is needed.

**The heatmap cells:** Each cell shows the mean confidence for one keypoint within its region. Green = high confidence, Red = low. NaN cells (gray/masked) appear when a region has fewer than `max_kp_in_region` keypoints.

### Plot 3 — Top 5 / Bottom 5 Reliability Ranking:
Shows which specific keypoints are the model's strengths and weaknesses. This is the most actionable output — if "R_B_Hoof" consistently has low confidence, that tells you the back-right hoof annotation needs more diverse training examples.

---

## Section 14 — Results Export

```python
# Zip everything except model weights (too large)
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for fpath in sorted(RESULTS_DIR.rglob("*")):
        if fpath.is_file() and fpath.suffix.lower() not in {".pt"}:
            zf.write(fpath, fpath.relative_to(BASE_OUT))
```

**Why exclude `.pt` files?** The trained model weights (`best.pt`, `last.pt`) can be hundreds of megabytes. They're kept in the Kaggle run folder but excluded from the ZIP to keep the download manageable. The ZIP is meant for figures and CSVs (the "results"), not the model itself.

**`master_metrics.json`:** A single JSON file with all key numbers: validation metrics, tracker statistics, and per-keypoint confidence stats. This is useful for programmatic comparison across experiments without having to re-run anything.

**`results_manifest.csv`:** A list of every output file with its size, category, and type. Acts as a table of contents for the results folder.

---

## Summary of the Key Problems Solved and Decisions Made

| Problem | Root Cause | Solution Chosen | Why |
|---|---|---|---|
| Zero-size bounding boxes | Dataset annotation gaps | Reconstruct from keypoints | Most faithful fix without re-annotation |
| Limited dataset size | Only ~500 images | ×8 augmentation on train | Prevents overfitting while preserving anatomy |
| Wrong keypoint flip mapping | YOLO needs explicit swap pairs | `FLIP_IDX` array | Required by YOLO for correct horizontal flip behavior |
| Wrong run path | Ultralytics adds "pose/" prefix | Hardcode `runs/pose/runs/` | Empirically discovered, no YOLO API to suppress this |
| `sv.BotSort` may not exist | Supervision version differences | Try/except with YOLO native fallback | Makes code portable across environments |
| High pose loss weight | Pose estimation is primary goal | `pose=24.0` vs `box=5.0` | Explicitly prioritize what the assignment asks for |
