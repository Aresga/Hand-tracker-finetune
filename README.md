# Hand Tracker Fine-tune

YOLO-based hand pose estimation model fine-tuned for real-time 21-keypoint tracking.
Exported to ONNX for CPU/CUDA inference via [HandSynth](https://github.com/Aresga/HandSynth).

---

## Keypoint Ordering

> ⚠️ This model uses **index-first ordering** — NOT standard MediaPipe.

| Index | Joint | Finger |
|-------|-------|--------|
| 0 | Wrist | — |
| 1 | MCP | Index |
| 2 | PIP | Index |
| 3 | DIP | Index |
| 4 | Tip | Index |
| 5 | MCP | Middle |
| 6 | PIP | Middle |
| 7 | DIP | Middle |
| 8 | Tip | Middle |
| 9 | MCP | Ring |
| 10 | PIP | Ring |
| 11 | DIP | Ring |
| 12 | Tip | Ring |
| 13 | MCP | Pinky |
| 14 | PIP | Pinky |
| 15 | DIP | Pinky |
| 16 | Tip | Pinky |
| 17 | CMC | Thumb |
| 18 | MCP | Thumb |
| 19 | IP | Thumb |
| 20 | Tip | Thumb |

---

## Output Tensor Format

```
shape: [batch, 300, 69]

Per detection (69 values):
  [0..3]  → bounding box (cx, cy, w, h)  — model input space
  [4]     → objectness confidence
  [5]     → class index (0 = hand)
  [6..68] → 21 keypoints × 3 (x, y, confidence)
```

---

## Repository Structure

```
├── train/
│   ├── train.py           # training entry point
│   ├── val.py             # validation / metric evaluation
│   └── requirements.txt   # pinned dependencies
├── export/
│   └── export_onnx.py     # PT → ONNX export with correct settings
├── configs/
│   ├── hand_pose.yaml     # YOLO hyperparameters
│   └── dataset.yaml       # dataset paths, class names, skeleton
├── results/
│   ├── metrics.json       # mAP + PCK scores
│   └── sample_outputs/    # annotated inference images
└── models/
    └── .gitkeep           # weights distributed via GitHub Releases
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r train/requirements.txt

# 2. Prepare dataset — edit configs/dataset.yaml with your paths

# 3. Train
python train/train.py --config configs/hand_pose.yaml --data configs/dataset.yaml

# 4. Export to ONNX
python export/export_onnx.py --weights runs/train/best.pt --output models/hand_tracker.onnx

# 5. Validate
python train/val.py --weights models/hand_tracker.onnx --data configs/dataset.yaml
```

---

## Model Weights

Weights are distributed via **[GitHub Releases](https://github.com/Aresga/Hand-tracker-finetune/releases)**:

| File | Description |
|------|-------------|
| `hand_tracker.onnx` | ONNX export — use this for inference |
| `hand_tracker.pt` | PyTorch checkpoint |

---

## Performance

| Metric | Value |
|--------|-------|
| mAP@0.5 | see `results/metrics.json` |
| PCK@0.2 | see `results/metrics.json` |
| Input size | 640×640 |
| Inference (CUDA) | ~4ms/frame |
| Inference (CPU) | ~25ms/frame |

---

## License

MIT — see [LICENSE](LICENSE).
