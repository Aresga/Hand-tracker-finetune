"""Training entry point for hand pose fine-tuning.

Usage:
    python train/train.py --config configs/hand_pose.yaml --data configs/dataset.yaml
    python train/train.py --config configs/hand_pose.yaml --data configs/dataset.yaml --resume
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLO hand pose model")
    p.add_argument("--config", type=str, default="configs/hand_pose.yaml",
                   help="Hyperparameter config (YOLO .yaml)")
    p.add_argument("--data", type=str, default="configs/dataset.yaml",
                   help="Dataset config")
    p.add_argument("--weights", type=str, default="yolov8n-pose.pt",
                   help="Starting weights (Ultralytics pretrained or local .pt)")
    p.add_argument("--device", type=str, default="0",
                   help="Device: '0' for CUDA GPU 0, 'cpu' for CPU")
    p.add_argument("--resume", action="store_true",
                   help="Resume from last checkpoint")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.weights)

    results = model.train(
        cfg=args.config,
        data=args.data,
        device=args.device,
        resume=args.resume,
    )

    print(f"\nTraining complete. Best weights: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
