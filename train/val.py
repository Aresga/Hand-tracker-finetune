"""Validation and metric evaluation.

Supports both PyTorch (.pt) and ONNX (.onnx) weights.

Usage:
    python train/val.py --weights runs/train/exp/weights/best.pt --data configs/dataset.yaml
    python train/val.py --weights models/hand_tracker.onnx --data configs/dataset.yaml
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate hand pose model")
    p.add_argument("--weights", type=str, required=True,
                   help="Path to .pt or .onnx weights")
    p.add_argument("--data", type=str, default="configs/dataset.yaml")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--save-metrics", type=str, default="results/metrics.json",
                   help="Write metrics to this JSON file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.weights)

    metrics = model.val(
        data=args.data,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
    )

    # ── extract key numbers ─────────────────────────────────────────────────
    out = {
        "map50":      float(metrics.box.map50),
        "map50_95":   float(metrics.box.map),
        "pose_map50": float(metrics.pose.map50)  if hasattr(metrics, "pose") else None,
        "pose_pck":   float(metrics.pose.map)    if hasattr(metrics, "pose") else None,
    }

    print("\n=== Validation Results ===")
    for k, v in out.items():
        print(f"  {k}: {v}")

    # ── save to file ────────────────────────────────────────────────────────
    save_path = Path(args.save_metrics)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nMetrics saved to {save_path}")


if __name__ == "__main__":
    main()
