"""Export trained YOLOv8n-pose weights to ONNX.

Equivalent CLI command:
    yolo export \
        model=runs/pose/hand_synth_v1/weights/best.pt \
        format=onnx \
        imgsz=640 \
        opset=12 \
        dynamic=False \
        simplify=False

Output tensor shape: [1, 300, 69]
    [0:4]   bounding box  (cx, cy, w, h) normalised
    [4]     objectness confidence
    [5]     class index   (0 = hand)
    [6:69]  21 keypoints x 3  (x, y, confidence)

Keypoint order (standard MediaPipe):
    0=wrist | 1-4=thumb | 5-8=index | 9-12=middle | 13-16=ring | 17-20=pinky

Usage:
    python export.py
    python export.py --weights runs/pose/hand_synth_v1/weights/best.pt
    python export.py --weights best.pt --output models/hand_tracker.onnx --opset 17
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLO pose model to ONNX")
    p.add_argument("--weights",  default="runs/pose/hand_synth_v1/weights/best.pt")
    p.add_argument("--output",   default="models/hand_tracker.onnx",
                   help="Destination path for the .onnx file")
    p.add_argument("--imgsz",    type=int,  default=640)
    p.add_argument("--opset",    type=int,  default=12)
    p.add_argument("--dynamic",  action="store_true", default=False,
                   help="Dynamic batch axis (changes output shape)")
    p.add_argument("--simplify", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.weights)

    print(f"Exporting {args.weights} -> ONNX")
    print(f"  imgsz={args.imgsz}  opset={args.opset}  dynamic={args.dynamic}  simplify={args.simplify}")

    exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
        nms=False,   # keep raw [1,300,69] tensor — NMS handled by consumer
    )

    # Move to requested destination
    src = Path(exported_path)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    print(f"\nSaved to: {dst}")

    # Verify shape
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
        dummy = np.zeros((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: dummy})
        print(f"Verified output shape: {out[0].shape}")
        assert out[0].shape == (1, 300, 69), "Unexpected output shape!"
        print("Shape check passed: [1, 300, 69] ✓")
    except ImportError:
        print("(onnxruntime not installed — skipping shape verification)")


if __name__ == "__main__":
    main()
