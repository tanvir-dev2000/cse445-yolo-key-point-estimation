# CSE 445 — Viva Q&A (35 Questions)

---

## Category 1: Object Detection

**Q1. What is YOLO and why is it preferred over two-stage detectors like Faster R-CNN for real-time tracking?**

YOLO (You Only Look Once) is a single-stage object detector that frames detection as a single regression problem — it predicts bounding boxes and class probabilities directly from the full image in one forward pass.

Two-stage detectors (Faster R-CNN) first generate region proposals, then classify each proposal — two separate networks, two forward passes.

Why YOLO wins for tracking:
- Much faster inference (30–100 FPS vs ~10 FPS for Faster R-CNN) — essential for video
- Single network is simpler to deploy and optimize
- Global context: YOLO sees the full image at once, so it makes fewer background false positives
- In this notebook, we need to run detection on every video frame, so speed is critical

---

**Q2. What does IoU (Intersection over Union) mean and why is it used as a detection metric?**

IoU measures the overlap between a predicted bounding box and the ground truth box.

> IoU = Area of Overlap / Area of Union

A value of 1.0 = perfect overlap. 0 = no overlap at all.

Why it's used: Raw pixel distance between box centers doesn't account for box size. A 10-pixel error on a tiny cow far away is catastrophic; the same error on a large nearby cow is negligible. IoU is scale-invariant and directly measures how "correct" the localization is.

In this notebook: mAP@50 means a detection is counted correct if IoU ≥ 0.50 with the ground truth. mAP@50-95 averages over thresholds from 50% to 95%.

---

**Q3. What is mAP (mean Average Precision) and how do mAP@50 and mAP@50-95 differ?**

Average Precision (AP) for a class = area under the precision-recall curve. mAP = mean of AP across all classes (here, just one class: "cow").

- **mAP@50:** A detection is "correct" if its IoU with ground truth ≥ 0.50. Relatively lenient — allows somewhat imprecise boxes.
- **mAP@50-95:** AP is computed at each IoU threshold from 0.50 to 0.95 in steps of 0.05, then averaged. This is the official COCO metric. Much stricter — requires precise localization to score well at thresholds like 0.90 or 0.95.

Rule of thumb: mAP@50-95 is roughly half the mAP@50 value for most models. A large gap between them means the model finds cows but doesn't localize them tightly.

---

**Q4. What is the difference between precision and recall, and what is the F1 score?**

> Precision = TP / (TP + FP)  ← of all detections, how many were real cows?
> Recall    = TP / (TP + FN)  ← of all real cows, how many were found?
> F1        = 2 × P × R / (P + R)

Precision penalizes false alarms (detecting background as a cow). Recall penalizes missed detections (real cows the model didn't find).

The tradeoff: Lowering the confidence threshold increases recall but drops precision. The F1 score is the harmonic mean — it's low if either P or R is low, forcing a balanced evaluation. We used conf=0.30 in the notebook as a starting tradeoff point.

---

## Category 2: Pose Estimation

**Q5. What is pose estimation and how does YOLO-Pose extend standard object detection?**

Pose estimation is the task of localizing anatomical keypoints (joints, landmarks) on a detected object in addition to its bounding box.

- Standard YOLO outputs: (class, confidence, x, y, w, h) per detection
- YOLO-Pose adds: (x₁,y₁,v₁, x₂,y₂,v₂, …, xₖ,yₖ,vₖ) — coordinates and visibility for each of K keypoints

This notebook uses 26 keypoints per cow (eyes, hooves, spine, shoulders, hips, etc.). The model is trained with an additional pose loss that penalizes wrong keypoint positions and wrong visibility flags.

---

**Q6. What is OKS (Object Keypoint Similarity) and how does it differ from IoU?**

OKS is the pose estimation equivalent of IoU — it measures how well predicted keypoints match ground truth keypoints.

> OKS = Σ exp(−dᵢ² / 2s²σᵢ²) · δ(vᵢ>0) / Σ δ(vᵢ>0)

Where dᵢ = Euclidean distance between predicted and GT keypoint i, s = object scale, σᵢ = per-keypoint sigma (tolerance).

Key difference from IoU: OKS accounts for the fact that different keypoints have different precision requirements. In our notebook, eyes have σ=0.025 (tight tolerance), while hooves have σ=0.070 (more slack — large contact area, harder to pin exactly). Pose mAP is computed using OKS instead of box IoU.

---

**Q7. What is the visibility flag (v) in YOLO-Pose labels and why does it matter?**

Each keypoint has 3 values: (x, y, v) where v is the visibility flag:
- **v=0:** Keypoint not labeled / absent. Coordinates are (0,0) and should be ignored.
- **v=1:** Keypoint is occluded (hidden behind another object). Position is approximate.
- **v=2:** Keypoint is fully visible. Position is exact.

Why it matters for training: The model only computes keypoint loss on visible/occluded points (v>0). Training on absent keypoints (v=0) would teach the model to predict phantom landmarks.

In the notebook's visualization: green dots = v=2 (visible), orange dots = v=1 (occluded).

---

**Q8. What is the purpose of FLIP_IDX in pose estimation training?**

Horizontal flipping is a common augmentation — mirror the image left-to-right. But for pose estimation, flipping an image doesn't automatically fix the keypoint labels.

Example: If L_Eye is at index 0 and R_Eye is at index 1, after a horizontal flip the Left Eye visually becomes the Right Eye. If you don't swap them in the label, the model learns that "Left Eye" appears on the right side of the body — wrong anatomy.

> FLIP_IDX = [1,0,2,4,3,6,5,7,9,8,11,10,...]
> After flip: keypoint[i] ← keypoint[FLIP_IDX[i]]

FLIP_IDX[0]=1 means L_Eye swaps with R_Eye. FLIP_IDX[2]=2 means Chin stays as Chin (it's on the midline). Without this, horizontal flip augmentation actively corrupts the training signal.

---

**Q9. What are KP_SIGMAS and how were the values chosen?**

KP_SIGMAS are per-keypoint tolerance values used in the OKS formula. A larger sigma means the metric allows more prediction error for that keypoint.

Assignment logic:
- σ = 0.025 (tight): Eyes, Nose, Chin, Ear bases, Mouth corner — small, precise landmarks
- σ = 0.035 (medium): Shoulders, Withers, Spine — medium-sized landmarks
- σ = 0.050 (relaxed): Hips, Knees, Tail Base — larger body regions
- σ = 0.070 (most relaxed): Hooves, Tail Tip — large contact surfaces, frequently occluded

These values reflect the physical size and annotation difficulty of each landmark, inspired by COCO human pose sigma conventions.

---

## Category 3: Data & Augmentation

**Q10. What was the zero-size bounding box problem and how was it fixed?**

The problem: Some label files had bw=0, bh=0 — a bounding box with zero area. YOLO needs a valid bounding box. Zero-area boxes cause the box loss to produce NaN gradients or contribute no training signal, silently degrading model performance.

Root cause: The original annotation tool only stored keypoint positions for some images and left the bounding box fields blank.

The fix: For any label with bw=0 or bh=0, reconstruct the bounding box from visible keypoints:
1. Collect all (x,y) pairs where v > 0
2. Compute min/max x and y to get the tight bounding box
3. Add 5% padding on each side for margin
4. Clip the center so the box doesn't go out of image bounds

---

**Q11. Why do we augment the training set but NOT the validation set?**

Training augmentation artificially diversifies the data, reducing overfitting and improving generalization.

The validation set must reflect the real distribution of data the model will encounter in deployment. If we augmented validation images, the metrics would no longer tell us how the model performs on actual, unmodified cow images.

In this notebook, the val transform is only resize + letterbox padding — same preprocessing YOLO expects at inference. No random flips, brightness changes, or noise — those would make val metrics unreliable.

---

**Q12. What is mosaic augmentation and why is it useful?**

Mosaic augmentation combines 4 training images into a single 2×2 grid. Each quadrant is a randomly cropped and scaled version of a different image.

Benefits:
- Each training batch effectively sees 4× more scene variety
- Forces the model to detect objects at small scales (squeezed into a quadrant), improving small object detection
- More diverse context in each sample — harder to overfit to specific backgrounds
- Increases batch diversity without increasing batch size (memory)

Originally introduced in YOLOv4, it's now standard in all YOLO versions.

---

**Q13. What is CoarseDropout and what real-world condition does it simulate?**

CoarseDropout (used with p=0.2) randomly masks out rectangular patches of an image by setting them to black.

Real-world simulation: It mimics partial occlusion — situations where another cow, a fence post, a feeding trough, or the camera edge blocks part of a cow's body.

Why this helps pose estimation specifically: When keypoints are covered by a patch during training, the model must learn to predict keypoints from surrounding body context rather than from the keypoint's exact pixel appearance. This makes it more robust to the v=1 (occluded) case in real data.

---

**Q14. Why is the padding color (114, 114, 114) used in letterboxing?**

When resizing to 640×640 while preserving aspect ratio, empty space must be filled. YOLO uses gray (114, 114, 114) as the conventional padding color.

Why 114: It's the empirically found mean pixel value of the ImageNet dataset, which most YOLO pretrained models were trained on. Using a color close to the dataset mean means padding regions have minimal activation in early convolutional layers — essentially invisible to the network. Using pure black (0) or white (255) would activate neurons in padded regions, creating edge artifacts near the letterbox borders.

---

## Category 4: Model Training

**Q15. What is transfer learning and why was it used instead of training from scratch?**

Transfer learning starts training from weights already learned on a large dataset (here, COCO with 80 categories) rather than random initialization.

Why it works: Low-level features (edges, textures, color gradients) learned from millions of images are universally useful. Only the high-level, task-specific layers need adjustment for cows and 26 keypoints.

Advantages:
- Requires far fewer cow images to reach good accuracy
- Converges in 100 epochs instead of potentially 1000+
- Lower risk of overfitting on the small cow dataset
- Better final performance than training from scratch with the same data

---

**Q16. What is AdamW and why was it chosen over SGD?**

AdamW is the Adam optimizer with decoupled weight decay (L2 regularization).

Adam maintains per-parameter adaptive learning rates using running estimates of gradient mean and variance. Parameters receiving small, consistent gradients get a larger effective LR; noisy, large gradients get a smaller one.

Weight decay decoupling: Standard Adam applies weight decay through the gradient, which interferes with the adaptive rate. AdamW applies it separately, making regularization more predictable.

Why not SGD: SGD needs more careful LR tuning and slower warmup. AdamW is more forgiving and converges faster during fine-tuning — ideal when adapting pretrained weights.

---

**Q17. What is warmup and cosine annealing in the learning rate schedule?**

**Warmup (warmup_epochs=5):** For the first 5 epochs, the LR increases linearly from ~0 to lr0=3e-4. Starting from a large LR immediately with pretrained weights can "shock" the weights and destroy the good initialization.

**Cosine annealing:** After warmup, the LR decreases following a cosine curve from lr0=3e-4 down to lr0×lrf = 3e-6. The cosine shape means:
- Fast initial decrease when LR is high (coarse optimization)
- Very slow decrease at the end (fine-grained refinement near the optimum)

This avoids the abrupt drops of step decay and generally yields better final accuracy than a constant LR.

---

**Q18. What is AMP (Automatic Mixed Precision) and what are its benefits?**

AMP automatically uses float16 (16-bit) arithmetic where safe and float32 (32-bit) where precision is critical (e.g., loss scaling, batch normalization).

Benefits:
- Speed: float16 operations are 2× faster on modern GPUs (tensor cores)
- Memory: float16 uses half the memory, allowing larger batch sizes
- Accuracy: Loss scaling prevents float16 underflow (very small gradients becoming zero)

In this notebook: amp=True roughly doubles training throughput on the Kaggle GPU with negligible accuracy loss (< 0.1% mAP difference typically).

---

**Q19. What is early stopping (patience=30) and why is it important?**

Early stopping monitors a validation metric (pose mAP) during training. If it doesn't improve for 30 consecutive epochs, training stops automatically.

Why it's important:
- Prevents overfitting: after the optimal point, further training causes the model to memorize training data
- Saves compute: no need to run all 100 epochs if the model peaks at epoch 60
- Selects best checkpoint: YOLO saves best.pt at the epoch with the highest val mAP, not the last epoch

30 epochs patience is generous — it allows the model to recover from temporary plateaus.

---

**Q20. Why is the pose loss weight (24.0) much higher than box loss (5.0)?**

The total YOLO loss is a weighted sum:
> Total loss = box_weight × box_loss + pose_weight × pose_loss + cls_weight × cls_loss

Reasoning:
- This is a pose estimation assignment — the primary goal is accurate keypoint prediction, not tight boxes
- Setting pose=24.0 forces the optimizer to treat keypoint error as ~5× more important than box error
- The model will sacrifice some box precision to get keypoints right — the correct tradeoff here
- With only 1 class (cow), cls_loss converges almost immediately anyway

If this were pure detection, you'd increase box weight. For pure tracking, you'd balance them equally.

---

## Category 5: Tracking

**Q21. What is multi-object tracking (MOT) and what makes it hard?**

MOT assigns consistent identities to multiple objects across video frames. It must solve two problems simultaneously:
- Detection: Find all cows in the current frame
- Association: Match each detection to an existing tracked cow (or start a new track)

What makes it hard:
- Occlusion: Cows walk behind each other — an ID disappears and must be recovered correctly
- ID switches: When two cows cross paths, the tracker may swap their identities
- Scale/appearance changes: The same cow looks different at different distances
- Detection failures: Low-confidence or missed detections break track continuity

---

**Q22. How does ByteTrack work and what is its key innovation?**

ByteTrack associates every detection — both high and low confidence — with existing tracks, instead of discarding low-confidence detections as noise.

Algorithm steps per frame:
1. Run detector → get detections with confidence scores
2. First association: Match high-confidence detections (> threshold) to existing tracks using IoU (Hungarian algorithm)
3. Second association: Take unmatched tracks + low-confidence detections → try to match them. This recovers tracks for cows that were briefly occluded
4. Start new tracks for unmatched high-confidence detections
5. Keep lost tracks alive for lost_track_buffer=50 frames before deleting

Key innovation: Using low-confidence detections in step 3 dramatically reduces ID switches compared to methods that discard them.

---

**Q23. How does BoT-SORT improve upon ByteTrack?**

BoT-SORT (Bootstrapped Object-Tracking SORT) adds two components on top of ByteTrack's association strategy:

**1. Camera Motion Compensation (CMC):** If the camera is moving (panning, zooming), all object positions shift uniformly. BoT-SORT uses image registration to estimate the camera's motion and correct object positions before matching. This prevents good tracks from being broken just because the camera moved.

**2. Re-ID (Re-Identification) features:** A separate appearance model extracts an embedding describing what each cow looks like. During association, both IoU distance AND appearance distance are used. This allows re-identifying a cow after a long occlusion — even if IoU with previous position is 0, appearance may still match.

When BoT-SORT outperforms ByteTrack: Crowded scenes with long occlusions, or videos with camera movement.

---

**Q24. What is track continuity and how is it calculated?**

Track continuity measures how persistently each track stays active across the video.

> Track Continuity = avg_tracks_per_frame / unique_ids

- Perfect score (1.0): Every unique ID is active in every frame — no track is ever lost or fragmented
- Low score (e.g., 0.3): Many unique IDs were assigned but most were short-lived — the same physical cow got several different IDs

Example: If 5 cows are in the video and the tracker correctly maintains 5 tracks throughout → continuity ≈ 1.0. If the tracker creates 20 short fragments for the same 5 cows → continuity ≈ 0.25.

---

**Q25. What does 'lost_track_buffer=50' mean and how does it affect tracking?**

lost_track_buffer=50 means a track is kept "alive" for up to 50 frames after the detector loses the cow, before it's permanently deleted.

At 30 FPS: 50 frames = ~1.67 seconds. If a cow walks behind a fence and reappears within 1.67 seconds, the tracker reconnects it to the original ID.

- Too small (e.g., 5): Tracks die quickly during occlusion → more ID switches when the cow reappears
- Too large (e.g., 500): Lost tracks accumulate → slower tracker, potential wrong re-associations

50 is a good default for outdoor livestock footage where cows may be briefly hidden by feeding structures.

---

## Category 6: Computer Vision Theory

**Q26. What is a convolutional neural network (CNN) and why is it used for image tasks?**

A CNN applies learnable filters (kernels) that slide spatially across the image to detect local features. Each layer learns increasingly abstract features: edges → textures → parts → objects.

Key properties:
- Local connectivity: Each neuron only looks at a small patch — captures local structure efficiently
- Weight sharing: The same filter is applied everywhere — a "cow eye detector" works anywhere in the image
- Translation equivariance: Moving a cow in the image shifts the feature map, but doesn't change what features are detected
- Hierarchical features: Deep stacking builds complex representations from simple ones

---

**Q27. What is non-maximum suppression (NMS) and why does YOLO need it?**

YOLO predicts multiple bounding boxes per grid cell. For a single cow, many overlapping boxes may all be predicted with high confidence.

NMS removes duplicate detections:
1. Sort all detections by confidence score, descending
2. Keep the highest-confidence detection
3. Remove all other detections that overlap it by IoU > threshold (typically 0.45–0.65)
4. Repeat for remaining detections

Result: Each object gets exactly one detection. Without NMS, a single cow might be reported 10–50 times with slightly different box coordinates, making counting and tracking impossible.

---

**Q28. What is the difference between anchor-based and anchor-free detectors?**

**Anchor-based** (older YOLO, Faster R-CNN): Pre-define fixed bounding box shapes and sizes (anchors) at each grid cell. The network predicts offsets from these anchors. Requires careful anchor design tuned to the dataset.

**Anchor-free** (modern YOLO v6+, FCOS, CenterNet): Directly predict the center point and dimensions of each object without predefined shapes. More flexible — no anchor tuning needed, generalizes better to unusual aspect ratios.

In this notebook: YOLOv26 uses an anchor-free approach — we don't define anchor shapes in dataset.yaml, only kpt_shape (keypoint structure).

---

**Q29. What is the Hungarian algorithm and why does ByteTrack use it?**

The Hungarian algorithm solves the Linear Assignment Problem (LAP): given a cost matrix (N detections × M tracks), find the one-to-one assignment that minimizes total cost, in O(N³) time.

In tracking: The cost between a detection and a track is (1 − IoU). We want to assign each new detection to the existing track it overlaps most, without giving two detections to one track.

Why not greedy: Greedy matching always takes the best pair locally, which can produce globally suboptimal assignments. The Hungarian algorithm guarantees the globally optimal assignment.

lapx in the notebook provides the C++ implementation required by ByteTrack.

---

**Q30. What is batch normalization and why is it important in deep networks?**

Batch normalization normalizes the activations of each layer to have zero mean and unit variance, then applies learned scale (γ) and shift (β) parameters.

> y = γ × (x − μ_batch) / σ_batch + β

Why it matters:
- Training stability: Prevents activations from growing very large or small (exploding/vanishing gradients)
- Faster convergence: Allows higher learning rates
- Regularization: The noise from computing stats on mini-batches acts as a regularizer
- Reduces internal covariate shift: Earlier layers can change without disrupting later layers as much

Modern YOLO uses BN after almost every convolutional layer.

---

**Q31. What is the role of the backbone, neck, and head in a YOLO architecture?**

**Backbone:** The feature extractor — a deep CNN that processes the input image and produces feature maps at multiple scales. Pretrained on large datasets.

**Neck:** A feature aggregation module (FPN or PANet) that combines feature maps from different backbone scales. Crucial for detecting objects of different sizes: small objects use high-resolution early features; large objects use low-resolution deep features.

**Head:** The task-specific output module. For YOLO-Pose, the head predicts:
- Bounding box coordinates and objectness score
- Class probabilities
- Keypoint coordinates (x,y) and visibility for all 26 keypoints

The head uses aggregated multi-scale features from the neck to make predictions at three different scales simultaneously.

---

## Category 7: Metrics & Evaluation

**Q32. Why might a model have high mAP@50 but low mAP@50-95?**

mAP@50 only requires 50% overlap — relatively easy to achieve even with imprecise localization. mAP@50-95 also evaluates at 60%, 70%, 80%, 90%, 95% overlap. At 90%+ IoU, even a few pixel error counts as a miss.

Common causes of the gap:
- Model finds cows correctly but box boundaries are imprecise (consistently slightly too large or small)
- Dataset labels themselves are inconsistent — if annotators drew boxes differently, the model can't be precisely consistent
- Small objects: pixel errors are proportionally larger relative to the object area

Interpretation: A large gap means the model is good at "finding" cows but needs improvement in "outlining" them precisely.

---

**Q33. What does the keypoint confidence score from YOLO-Pose actually represent?**

After training, YOLO-Pose outputs a confidence score for each keypoint — a value in [0,1] representing the model's estimated probability that the predicted keypoint location is correct.

In this notebook:
- Skeleton lines drawn only when both endpoint keypoints have conf > 0.3
- Individual keypoint dots drawn when conf > 0.25
- Keypoints with consistently low mean confidence (Section 13) are the model's "weak spots" — typically hooves and tail tips

Important: Confidence is not the same as accuracy. A model can be confidently wrong. But averaged over the validation set, mean confidence correlates well with prediction quality.

---

**Q34. How is training loss different from validation metric? Why do both matter?**

Training loss is computed on mini-batches during backpropagation. It directly drives weight updates. Decreasing train loss means the model is fitting the training data better.

Validation metric (mAP) is computed on the entire held-out validation set after each epoch, using no gradient updates. It measures generalization.

Both matter because:
- Train loss drops but val mAP stagnates → overfitting (memorizing, not generalizing)
- Val mAP drops while train loss still drops → definite overfitting — early stopping kicks in
- Both improve together → healthy training
- Both high → underfitting (model capacity or LR too low)

---

**Q35. What is the significance of the Albumentations KeypointParams in the augmentation pipeline?**

When applying transformations to images, the keypoint coordinates must be transformed consistently with the image. Albumentations handles this through KeypointParams.

> keypoint_params=A.KeypointParams(format="xy", remove_invisible=False, label_fields=["kp_vis"])

- format="xy": Keypoints are provided as (x, y) pixel coordinates
- remove_invisible=False: When a keypoint goes out of frame after transformation (e.g., after ShiftScaleRotate), keep it in the output list as (0,0) instead of removing it. This is critical — YOLO expects exactly 26 keypoints per instance at fixed indices. If some were removed, the remaining keypoints would shift to wrong indices, corrupting the label structure entirely.
- label_fields=["kp_vis"]: The visibility flag list is linked to keypoints so they transform together as pairs.

Without KeypointParams, Albumentations would only transform the image and leave keypoint coordinates unchanged — producing completely wrong annotations.
