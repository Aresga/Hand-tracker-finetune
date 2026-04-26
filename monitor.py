#!/usr/bin/env python3
"""
YOLO Training Monitor
Run: python monitor.py
Or for auto-refresh: watch -n 60 python monitor.py
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys

CSV_PATH = "runs/pose/hand_synth_v1/results.csv"

if not os.path.exists(CSV_PATH):
    print(f"ERROR: Cannot find {CSV_PATH}")
    print("Make sure you run this from ~/GitRepos/yolo/Training/")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
ep = df["epoch"]

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Pose mAP50 (keypoints quality) ↑",
        "Box mAP50 (hand detection) ↑",
        "Pose Loss ↓",
        "Val Pose Loss ↓",
    ],
    vertical_spacing=0.18,
    horizontal_spacing=0.12,
)

fig.add_trace(go.Scatter(x=ep, y=df["metrics/mAP50(P)"],
    mode="lines+markers", name="mAP50 Pose",
    line=dict(width=2.5), marker=dict(size=6),
    fill="tozeroy", fillcolor="rgba(99,110,250,0.08)"), row=1, col=1)

fig.add_trace(go.Scatter(x=ep, y=df["metrics/mAP50(B)"],
    mode="lines+markers", name="mAP50 Box",
    line=dict(width=2.5), marker=dict(size=6),
    fill="tozeroy", fillcolor="rgba(0,204,150,0.08)"), row=1, col=2)

fig.add_trace(go.Scatter(x=ep, y=df["train/pose_loss"],
    mode="lines+markers", name="Train Pose Loss",
    line=dict(width=2.5), marker=dict(size=6)), row=2, col=1)

fig.add_trace(go.Scatter(x=ep, y=df["val/pose_loss"],
    mode="lines+markers", name="Val Pose Loss",
    line=dict(color="#EF553B", width=2.5), marker=dict(size=6),
    fill="tozeroy", fillcolor="rgba(239,85,59,0.08)"), row=2, col=2)

latest = df.iloc[-1]
subtitle = (f"Epoch {int(latest['epoch'])} | "
            f"mAP50(P)={latest['metrics/mAP50(P)']:.3f} | "
            f"mAP50(B)={latest['metrics/mAP50(B)']:.3f} | "
            f"Pose Loss={latest['train/pose_loss']:.4f}")

fig.update_layout(
    title=dict(
        text=f"YOLO Hand Keypoints Training<br>"
             f"<span style='font-size:14px;font-weight:normal'>{subtitle}</span>",
        x=0.5, xanchor="center"
    ),
    height=750,
    showlegend=False,
    font=dict(family="monospace", size=13),
    paper_bgcolor="#0f0f0f",
    plot_bgcolor="#141414",
    font_color="#cccccc",
)

fig.update_xaxes(title_text="Epoch", gridcolor="#222", zerolinecolor="#333")
fig.update_yaxes(gridcolor="#222", zerolinecolor="#333")

out = "training_monitor.html"
fig.write_html(out, auto_open=False)
print(f"\n✅ Saved: {out}")
print(f"   Open in browser: xdg-open {out}")
print(f"   Auto-refresh:    watch -n 60 python monitor.py")
print(f"\n📊 Latest stats (epoch {int(latest['epoch'])}):")
print(f"   mAP50 Pose (keypoints): {latest['metrics/mAP50(P)']:.4f}")
print(f"   mAP50 Box  (detection): {latest['metrics/mAP50(B)']:.4f}")
print(f"   Train pose loss:        {latest['train/pose_loss']:.4f}")
print(f"   Val   pose loss:        {latest['val/pose_loss']:.4f}")

if len(df) >= 5:
    last5 = df["metrics/mAP50(P)"].tail(5).values
    max_gain = max(abs(last5[i] - last5[i-1]) for i in range(1, 5))
    if max_gain < 0.005:
        print("\n⚠️  WARNING: barely moved in last 5 epochs — early stopping soon.")
    else:
        print(f"\n✅ Still improving (max gain last 5 epochs: {max_gain:.4f})")
