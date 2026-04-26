"""Train YOLOv8n-pose on the Ultralytics Hand Keypoints dataset.

Equivalent CLI command:
    yolo pose train \
        data=datasets/hand-keypoints.yaml \
        model=yolov8n-pose.pt \
        epochs=100 \
        imgsz=640 \
        batch=16 \
        device=0 \
        name=hand_synth_v1

Usage:
    python train.py
    python train.py --epochs 200 --batch 32 --device 0
    python train.py --resume runs/pose/hand_synth_v1/weights/last.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8n-pose for hand keypoints")
    p.add_argument("--model",   default="yolov8n-pose.pt",          help="Base weights")
    p.add_argument("--data",    default="datasets/hand-keypoints.yaml", help="Dataset config")
    p.add_argument("--epochs",  type=int,   default=100)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--batch",   type=int,   default=16)
    p.add_argument("--device",  default="0",                         help="'0' for GPU, 'cpu' for CPU")
    p.add_argument("--name",    default="hand_synth_v1",             help="Run name under runs/pose/")
    p.add_argument("--patience",type=int,   default=5,               help="Early stopping patience")
    p.add_argument("--workers", type=int,   default=2)
    p.add_argument("--resume",  default=None,                        help="Path to last.pt to resume from")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    weights = args.resume if args.resume else args.model
    model = YOLO(weights)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        resume=bool(args.resume),
        # --- Optimiser (mirrors args.yaml) ---
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.0,
        # --- Loss weights ---
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        # --- Augmentation ---
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        erasing=0.4,
        auto_augment="randaugment",
        # --- Misc ---
        amp=True,
        seed=0,
        deterministic=True,
        plots=True,
        save=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"Best weights : {best}")
    print(f"To export    : python export.py --weights {best}")


if __name__ == "__main__":
    main()
