from pathlib import Path
import pandas as pd
from ultralytics import YOLO

ROOT = Path("/workspace/PHA_YOLO")
RUN = ROOT / "runs/obb/pha_dscov_hrsc2016_b4_600"
WEIGHTS = RUN / "weights/best.pt"
CSV = RUN / "results.csv"
REPORT = ROOT / "backup/pha_hrsc2016_b4_600.txt"

DATA = ROOT / "ultralytics/cfg/datasets/hrsc2016.yaml"

REPORT.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Read training history
# --------------------------------------------------
if not CSV.exists():
    raise FileNotFoundError(f"Training results CSV not found: {CSV}")

if not WEIGHTS.exists():
    raise FileNotFoundError(f"Best checkpoint not found: {WEIGHTS}")

df = pd.read_csv(CSV)
df.columns = df.columns.str.strip()

required_columns = [
    "epoch",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]

missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing required CSV columns: {missing}")

df = df.dropna(subset=["epoch"]).reset_index(drop=True)

if df.empty:
    raise ValueError(f"No recorded training epochs found in {CSV}")

# Convert metric columns to numeric values.
for col in ["metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Best epoch is determined by mAP50-95.
valid_rows = df.dropna(subset=["metrics/mAP50-95(B)"])
if valid_rows.empty:
    raise ValueError("No valid mAP50-95 values found in results.csv")

best_row = valid_rows.loc[valid_rows["metrics/mAP50-95(B)"].idxmax()]

# Last recorded epoch and its training-time validation metrics.
last_row = df.iloc[-1]

def write_epoch_metrics(f, label, row):
    f.write(f"{label}\n")
    f.write("-" * 60 + "\n")
    f.write(f"Epoch: {int(row['epoch'])}\n")

    map50 = row.get("metrics/mAP50(B)", float("nan"))
    map5095 = row.get("metrics/mAP50-95(B)", float("nan"))

    f.write(f"mAP50: {map50:.4f} ({map50 * 100:.2f}%)\n")
    f.write(f"mAP50-95: {map5095:.4f} ({map5095 * 100:.2f}%)\n\n")


# --------------------------------------------------
# 2. Validate the BEST checkpoint
# --------------------------------------------------
model = YOLO(str(WEIGHTS), task="obb")

results = model.val(
    data=str(DATA),
    split="val",
    imgsz=640,
    batch=1,
    device=0,
    plots=True,
    project=str(ROOT / "runs/obb"),
    name="pha_dsconv_best_report_val",
    exist_ok=True,
)

# --------------------------------------------------
# 3. Write report
# --------------------------------------------------
with REPORT.open("w", encoding="utf-8") as f:
    f.write("PHA-YOLO DSConv Evaluation Report\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Run directory: {RUN}\n")
    f.write(f"Best checkpoint: {WEIGHTS}\n")
    f.write(f"Recorded CSV epochs: {len(df)}\n")
    f.write(f"First recorded epoch: {int(df['epoch'].iloc[0])}\n")
    f.write(f"Last recorded epoch: {int(last_row['epoch'])}\n\n")

    # Best and last epoch metrics from the training CSV.
    write_epoch_metrics(
        f,
        "Best Recorded Training Epoch (by mAP50-95)",
        best_row,
    )

    write_epoch_metrics(
        f,
        "Last Recorded Training Epoch",
        last_row,
    )

    # Actual validation of best.pt.
    f.write("Best Checkpoint Validation Metrics\n")
    f.write("-" * 60 + "\n")
    f.write(f"Precision: {results.box.mp:.4f}\n")
    f.write(f"Recall: {results.box.mr:.4f}\n")
    f.write(f"mAP50: {results.box.map50:.4f}\n")
    f.write(f"mAP50-95: {results.box.map:.4f}\n\n")

    # Model complexity.
    f.write("Model Summary (best.pt)\n")
    f.write("-" * 60 + "\n")
    params = sum(p.numel() for p in model.model.parameters())
    info = model.info()

    f.write(f"Parameters: {params:,}\n")
    f.write(f"GFLOPs: {info[2]:.2f}\n")
    f.write(f"Layers: {len(model.model.model)}\n\n")

    # Per-class metrics for best.pt validation.
    f.write("Per-class Validation Metrics (best.pt)\n")
    f.write("-" * 60 + "\n")
    f.write(
        f"{'Class':<20} {'P':>8} {'R':>8} "
        f"{'mAP50':>8} {'mAP50-95':>10}\n"
    )

    for i, name in results.names.items():
        if i >= len(results.box.ap50):
            continue

        f.write(
            f"{name:<20} "
            f"{results.box.p[i]:8.4f} "
            f"{results.box.r[i]:8.4f} "
            f"{results.box.ap50[i]:8.4f} "
            f"{results.box.ap[i]:10.4f}\n"
        )

    f.write("\nBest Checkpoint Speed (ms/image)\n")
    f.write("-" * 60 + "\n")

    for key, value in results.speed.items():
        f.write(f"{key}: {value:.3f}\n")

print(f"\nReport saved to: {REPORT}")
print(REPORT.read_text(encoding="utf-8"))
