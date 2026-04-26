# Hand Tracker Fine-tune

Fine-tuning of **YOLOv8n-pose** on the [Ultralytics Hand Keypoints dataset](https://docs.ultralytics.com/datasets/pose/hand-keypoints/) for real-time 21-keypoint hand pose estimation.
The trained model is exported to ONNX and used by [HandSynth](https://github.com/Aresga/HandSynth) for gesture-driven audio synthesis.

---

## Results

![Training curves](newplot%20(1).png)

| Metric | Epoch 1 (baseline) | Epoch 51 (final) | Improvement |
|---|---|---|---|
| Pose mAP50 | 0.361 | **0.879** | +143% |
| Box mAP50 | 0.970 | **0.991** | +2.2% |
| Train Pose Loss | 8.706 | **2.287** | −73.7% |
| Val Pose Loss | 5.392 | **1.382** | −74.4% |
| Val Box Loss | 0.987 | **0.548** | −44.5% |

Box detection was already strong from epoch 1 — the base `yolov8n-pose.pt` generalises well to hands out of the box. The major gain was in **keypoint localisation quality**: Pose mAP50 more than doubled over 51 epochs and the validation loss curve was still trending downward, meaning further epochs would push it higher.

---

## Dataset

| Property | Value |
|---|---|
| Name | [Hand Keypoints](https://docs.ultralytics.com/datasets/pose/hand-keypoints/) |
| Creator | Rion Dsilva |
| Total images | 26,768 |
| Train split | 18,776 images |
| Val split | 7,992 images |
| Keypoints per hand | 21 |
| Format | YOLO keypoint (normalized, visibility flag) |
| License | CC BY-NC-SA 4.0 |

Annotations were generated using Google MediaPipe, ensuring high accuracy and consistency.

---

## Keypoint Ordering

This model follows the **standard MediaPipe ordering** as documented in the [Ultralytics hand-keypoints dataset](https://docs.ultralytics.com/datasets/pose/hand-keypoints/#introduction).

| Index | Joint | Finger |
|---|---|---|
| 0 | Wrist | — |
| 1 | CMC | Thumb |
| 2 | MCP | Thumb |
| 3 | IP | Thumb |
| 4 | Tip | Thumb |
| 5 | MCP | Index |
| 6 | PIP | Index |
| 7 | DIP | Index |
| 8 | Tip | Index |
| 9 | MCP | Middle |
| 10 | PIP | Middle |
| 11 | DIP | Middle |
| 12 | Tip | Middle |
| 13 | MCP | Ring |
| 14 | PIP | Ring |
| 15 | DIP | Ring |
| 16 | Tip | Ring |
| 17 | MCP | Pinky |
| 18 | PIP | Pinky |
| 19 | DIP | Pinky |
| 20 | Tip | Pinky |

---

## Output Tensor Format

```
ONNX output shape: [1, 300, 69]

Per detection (69 values):
  [0..3]  → bounding box (cx, cy, w, h) — normalised to input size
  [4]     → objectness confidence
  [5]     → class index (0 = hand)
  [6..68] → 21 keypoints × 3  (x, y, confidence)
```

---

## Training

### Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 5050 (Blackwell) |
| VRAM | 8 GB |
| Training time | ~3 hours (51 epochs) |

### Configuration (`args.yaml`)

| Parameter | Value |
|---|---|
| Base model | `yolov8n-pose.pt` (pretrained) |
| Epochs | 100 (stopped at 51 via patience) |
| Patience | 5 |
| Batch size | 16 |
| Image size | 640 × 640 |
| Optimizer | Auto (AdamW) |
| LR₀ / LRf | 0.01 / 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warmup epochs | 3 |
| Pose loss weight | 12.0 |
| Box loss weight | 7.5 |
| DFL loss weight | 1.5 |
| AMP | ✅ enabled |
| Flip LR augment | 0.5 |
| Mosaic augment | 1.0 |
| Close mosaic | last 10 epochs |

### Training Command

```bash
yolo pose train \
  data=datasets/hand-keypoints.yaml \
  model=yolov8n-pose.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  name=hand_synth_v1
```

### Export Command

```bash
yolo export \
  model=runs/pose/hand_synth_v1/weights/best.pt \
  format=onnx \
  imgsz=640 \
  opset=12 \
  dynamic=False \
  simplify=False
```

---

## Model Weights

Weights are distributed via **[GitHub Releases](https://github.com/Aresga/Hand-tracker-finetune/releases)**:

| File | Size | Description |
|---|---|---|
| `best.onnx` | 12 MB | ONNX export — use for inference |
| `best.pt` | 24 MB | PyTorch checkpoint |
| `last.pt` | 24 MB | Last epoch checkpoint |

---

## Repository Structure

```
├── args.yaml                # Full training configuration
├── datasets/
│   └── hand-keypoints.yaml  # Dataset config
├── results.csv              # Per-epoch metrics
├── newplot (1).png          # Training curves
├── labels.jpg               # Dataset label visualisation
├── train_batch*.jpg         # Sample training batches
├── monitor.py               # Training monitor script
└── notes.txt                # Training notes and commands
```

---

## Acknowledgements

- Dataset by **Rion Dsilva** — [Ultralytics Hand Keypoints](https://docs.ultralytics.com/datasets/pose/hand-keypoints/)
- Source images from: [11k Hands](https://sites.google.com/view/11khands), [2000 Hand Gestures](https://www.kaggle.com/), [Gesture Recognition](https://www.kaggle.com/)
- Images distributed under **CC BY-NC-SA 4.0**
- Keypoint annotations generated with **Google MediaPipe**
