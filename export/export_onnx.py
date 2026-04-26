"""Export YOLO hand pose .pt weights to ONNX.

The export settings here produce the exact output tensor shape used by HandSynth:
    [batch, 300, 69]  →  4 bbox + 1 conf + 1 class + 21×3 keypoints

Keypoint ordering in the exported model (index-first, thumb-last):
    0  = wrist
    1-4  = index  MCP/PIP/DIP/tip
    5-8  = middle MCP/PIP/DIP/tip
    9-12 = ring   MCP/PIP/DIP/tip
   13-16 = pinky  MCP/PIP/DIP/tip
   17-20 = thumb  CMC/MCP/IP/tip

Usage:
    python export/export_onnx.py --weights runs/train/exp/weights/best.pt
    python export/export_onnx.py --weights best.pt --output models/hand_tracker.onnx --imgsz 640
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLO pose model to ONNX")
    p.add_argument("--weights", type=str, required=True,
                   help="Path to trained .pt checkpoint")
    p.add_argument("--output", type=str, default="models/hand_tracker.onnx",
                   help="Destination .onnx path")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Input resolution (square)")
    p.add_argument("--dynamic", action="store_true", default=True,
                   help="Dynamic batch axis (recommended for C++ runtime)")
    p.add_argument("--simplify", action="store_true", default=True,
                   help="Run onnx-simplifier to clean the graph")
    p.add_argument("--opset", type=int, default=17,
                   help="ONNX opset version")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.weights)

    print(f"Exporting {args.weights} → ONNX (imgsz={args.imgsz}, opset={args.opset})...")

    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
        # Keep the default Ultralytics post-processing inside the graph OFF
        # so the C++ runtime receives the raw [batch, 300, 69] tensor.
        # If you set nms=True the output shape changes to [batch, num_dets, 6].
        nms=False,
    )

    # Move to requested output path
    src = Path(exported)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)

    print(f"\nExport complete: {dst}")
    print("Expected output tensor: [batch, 300, 69]")
    print("  stride layout: [cx,cy,w,h | conf | class | kpt0_x,kpt0_y,kpt0_c | ... | kpt20_x,kpt20_y,kpt20_c]")

    # ── quick shape verification via onnxruntime ────────────────────────────
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
        dummy = np.zeros((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: dummy})
        print(f"\nVerification — output shape: {out[0].shape}  ✓")
    except ImportError:
        print("(onnxruntime not available for verification — skipping)")


if __name__ == "__main__":
    main()
