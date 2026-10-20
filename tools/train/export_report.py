from pathlib import Path
import pandas as pd
from ultralytics import YOLO
import torch

ROOT = Path("/workspace/PHA_YOLO")
RUN = ROOT / "runs/obb/pha_test_b4_6006"
WEIGHTS = RUN / "weights/best.pt"
CSV = RUN / "results.csv"
REPORT = ROOT / "backup/pha_mvtec_screws_report.txt"

REPORT.parent.mkdir(parents=True, exist_ok=True)

# 1. Training history and best epoch
df = pd.read_csv(CSV)
df.columns = df.columns.str.strip()

metric = "metrics/mAP50-95(B)"
best_row = df.loc[df[metric].idxmax()]

# 2. Load best checkpoint and evaluate it
model = YOLO(str(WEIGHTS), task="obb")

# Use the same dataset configuration as training.
# This evaluates best.pt on the validation split.
results = model.val(
    data=str(ROOT / "ultralytics/cfg/datasets/mvtec.yaml"),
    split="val",
    imgsz=640,
    batch=1,
    device=0,
    plots=True,
    project=str(ROOT / "runs/obb"),
    name="pha_best_report_val",
    exist_ok=True,
)

# 3. Write report
with REPORT.open("w", encoding="utf-8") as f:
    f.write("PHA-YOLO Evaluation Report\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Checkpoint: {WEIGHTS}\n")
    f.write(f"Recorded CSV epochs: {len(df)}\n")
    f.write(f"First recorded epoch: {df['epoch'].iloc[0]}\n")
    f.write(f"Last recorded epoch: {df['epoch'].iloc[-1]}\n")
    f.write(f"Best recorded epoch: {int(best_row['epoch'])}\n")
    f.write(f"Best recorded mAP50: "
            f"{best_row['metrics/mAP50(B)']:.4f}\n")
    f.write(f"Best recorded mAP50-95: {best_row[metric]:.4f}\n\n")

    f.write("Best Checkpoint Validation Metrics\n")
    f.write("-" * 60 + "\n")
    f.write(f"Precision: {results.box.mp:.4f}\n")
    f.write(f"Recall: {results.box.mr:.4f}\n")
    f.write(f"mAP50: {results.box.map50:.4f}\n")
    f.write(f"mAP50-95: {results.box.map:.4f}\n\n")

    f.write("Model Summary\n")
    f.write("-" * 60 + "\n")
    f.write(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}\n")
    f.write(f"GFLOPs: {model.info()[2]:.2f}\n")
    f.write(f"Layers: {len(model.model.model)}\n\n")

    f.write("Per-class metrics\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'Class':<20} {'P':>8} {'R':>8} "
            f"{'mAP50':>8} {'mAP50-95':>10}\n")

    names = results.names
    for i, name in names.items():
        if i >= len(results.box.ap50):
            continue
        f.write(
            f"{name:<20} "
            f"{results.box.p[i]:8.4f} "
            f"{results.box.r[i]:8.4f} "
            f"{results.box.ap50[i]:8.4f} "
            f"{results.box.ap[i]:10.4f}\n"
        )

    f.write("\nSpeed (ms/image)\n")
    f.write("-" * 60 + "\n")
    for key, value in results.speed.items():
        f.write(f"{key}: {value:.3f}\n")

print(f"\nReport saved to: {REPORT}")
print(REPORT.read_text(encoding="utf-8"))

