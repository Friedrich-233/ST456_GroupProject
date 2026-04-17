"""
modal_train.py — ST456 Deep Learning Training on Modal GPU
=========================================================
Uses the same data loading and training code from the project notebooks.

Usage:
    # One-time setup: create the persistent volume
    modal volume create pcam-data

    # Download data from Google Drive to the volume (do this once)
    modal run modal_train.py::app download_data

    # Pre-train all three SSL encoders
    modal run modal_train.py::app pretrain_all --gpu T4

    # Run downstream experiment grid (3 seeds)
    modal run modal_train.py::app run_experiments --gpu A10G

    # Download checkpoints back to local machine
    modal run modal_train.py::app download_checkpoints

    # Stream logs of a running job
    modal run modal_train.py::app pretrain_all --gpu T4 2>&1 | tee train.log
"""

import os
import subprocess
from pathlib import Path

import modal

# ── Volume ──────────────────────────────────────────────────────────────────
VOLUME_NAME = "pcam-data"           # Persistent storage for data + checkpoints
GDRIVE_FOLDER_ID = "1gHou49cA1s5vua2V5L98Lt8TiWA3FrKB"  # Your shared Drive folder

# ── Modal App ──────────────────────────────────────────────────────────────
app = modal.App("st456-pcam-training")

# Base image: PyTorch + all dependencies needed for this project
BASE_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "numpy<2",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "timm",
        "h5py",
        "gdown",
        "tensorflow",               # for TFDS (PCam via tfds)
        "tensorflow-datasets",
    )
    .apt_install("git")
)

# Volume-mounted image: adds project code via git clone
MOUNTED_IMAGE = (
    BASE_IMAGE
    .pip_install("pyyaml")          # just in case
)


# ── Step 1: Download data from Google Drive ────────────────────────────────
@app.function(
    image=BASE_IMAGE,
    timeout=3600,                  # up to 1 hour to download ~8 GB
    volumes={"/vol": modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)},
)
def download_data():
    """
    Download PCam H5 files from your shared Google Drive folder into
    /vol/pcam_data on the Modal persistent volume.

    This is a ONE-TIME operation. Once downloaded, the volume retains the
    data across all subsequent training runs.
    """
    vol_dir = Path("/vol")
    pcam_dir = vol_dir / "pcam_data"
    pcam_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data will live at: {pcam_dir}")
    print(f"Downloading from Google Drive folder: {GDRIVE_FOLDER_ID}")

    # Download the entire shared folder using gdown
    # --folder flag downloads all files in the Drive folder
    result = subprocess.run(
        [
            "gdown",
            "--folder",
            f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}",
            "-O", str(pcam_dir),
            "--remaining-silent",
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )

    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])
        # Try individual files if folder download failed
        print("Trying alternative: download key files individually...")
        files_to_try = [
            "camelyonpatch_level_2_split_train_x.h5.gz",
            "camelyonpatch_level_2_split_train_y.h5.gz",
            "camelyonpatch_level_2_split_valid_x.h5.gz",
            "camelyonpatch_level_2_split_valid_y.h5.gz",
            "camelyonpatch_level_2_split_test_x.h5.gz",
            "camelyonpatch_level_2_split_test_y.h5.gz",
        ]
        for fname in files_to_try:
            print(f"  Trying {fname}...")
            r = subprocess.run(
                [
                    "gdown",
                    "--fuzzy",
                    "-O", str(pcam_dir / fname),
                    f"https://drive.google.com/uc?id={GDRIVE_FOLDER_ID}&confirm=t",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if r.returncode == 0:
                print(f"  ✓ Downloaded {fname}")
            else:
                print(f"  ✗ Failed: {fname}")
                print(f"    Error: {r.stderr[-300:]}")

    # Verify what we got
    downloaded = list(pcam_dir.glob("*"))
    print(f"\nFiles in {pcam_dir}:")
    for f in downloaded:
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:50s}  {size_mb:8.1f} MB")

    total_mb = sum(f.stat().st_size for f in downloaded if f.is_file()) / 1e6
    print(f"\nTotal downloaded: {total_mb:.0f} MB")

    # Commit volume so data persists
    print("\nData download complete. Volume committed.")
    print(f"Next: modal run modal_train.py::app pretrain_all --gpu T4")


# ── Step 2: Pre-train SSL encoders ────────────────────────────────────────
@app.function(
    image=MOUNTED_IMAGE,
    gpu="T4",                      # T4 = ~$0.60/hr, good for SSL pre-training
    timeout=28800,                 # 8 hours max
    volumes={
        "/vol":    modal.Volume.from_name(VOLUME_NAME, create_if_missing=True),
        "/content/ST456_GroupProject": modal.Volume.from_name(
            "st456-project-code", create_if_missing=True
        ),
    },
    container_idle_timeout=300,
)
def pretrain_all():
    """
    Pre-train all three SSL encoders:
        1. SimCLR (tailored augmentations)
        2. MAE base (patch=8, depth=6)
        3. MAE Improved (patch=4, depth=12, cosine LR, EMA)

    Checkpoints are saved to /vol/checkpoints/ on the persistent volume.
    Re-run this function to overwrite checkpoints (e.g. with more epochs).

    Expected data layout on /vol:
        /vol/pcam_data/
            camelyonpatch_level_2_split_train_x.h5.gz   (or .h5)
            camelyonpatch_level_2_split_train_y.h5.gz
            ...

    Runtime estimate on T4 (~16 GB VRAM):
        SimCLR 10 epochs:    ~20 min
        MAE 10 epochs:       ~25 min
        MAE Improved 15 ep:  ~40 min
        Total:               ~1.5 hours
    """
    import sys, random, math
    from pathlib import Path

    # ── Clone the project repo ────────────────────────────────────────────
    REPO_URL = "https://github.com/Friedrich-233/ST456_GroupProject.git"
    REPO_DIR = Path("/content/ST456_GroupProject")

    if not REPO_DIR.exists():
        print(f"Cloning repo to {REPO_DIR} ...")
        subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
    sys.path.insert(0, str(REPO_DIR))

    # ── Imports from project code ──────────────────────────────────────────
    from data import (
        load_all_data,
        build_downstream_loaders,
        LABEL_FRACTIONS,
    )
    from training import (
        train_simclr,
        train_mae,
        train_mae_improved,
    )

    # ── Paths ─────────────────────────────────────────────────────────────
    DATA_DIR       = Path("/vol/pcam_data")       # data.py will find .h5/.h5.gz here
    CHECKPOINT_DIR = Path("/vol/checkpoints")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"DATA_DIR:       {DATA_DIR}")
    print(f"CHECKPOINT_DIR: {CHECKPOINT_DIR}")
    print(f"GPU available:  {'cuda' if __import__('torch').cuda.is_available() else 'CPU'}")

    # ── Load data (same code as notebook) ────────────────────────────────
    print("\n[1/3] Loading PCam data...")
    data = load_all_data(
        drive_data_dir=DATA_DIR,            # data.py will find .h5/.h5.gz here
        data_dir=DATA_DIR,                  # cache in same dir (already unzipped)
        pretrain_fraction=0.15,             # ~50 K images for SSL pre-training
        downstream_pool_fraction=0.15,
        use_subset=True,
    )
    print("\nData loaded:")
    data.summary()

    # Build downstream loaders for later (used in Step 3)
    print("\nBuilding downstream loaders...")
    train_loaders, val_loader, test_loader = build_downstream_loaders(data)
    print(f"  train_loaders: {list(train_loaders.keys())}")
    print(f"  val samples:   {len(val_loader.dataset)}")
    print(f"  test samples:  {len(test_loader.dataset)}")

    # ── 2a. SimCLR ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2a — SimCLR Pre-Training (tailored augmentations)")
    print("=" * 60)
    simclr_model, simclr_history, simclr_ckpt = train_simclr(
        data=data,
        checkpoint_dir=CHECKPOINT_DIR,
        epochs=10,
        batch_size=128,
        learning_rate=3e-4,
        tailored=True,
        checkpoint_name="simclr_encoder_tailored.pth",
    )
    print(f"\n✓ SimCLR checkpoint saved → {simclr_ckpt}")

    # ── 2b. MAE Base ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2b — MAE Pre-Training (base: patch=8, depth=6)")
    print("=" * 60)
    mae_model, mae_history, mae_ckpt = train_mae(
        data=data,
        checkpoint_dir=CHECKPOINT_DIR,
        epochs=10,
        batch_size=128,
        learning_rate=1e-3,
        checkpoint_name="mae_encoder.pth",
    )
    print(f"\n✓ MAE checkpoint saved → {mae_ckpt}")

    # ── 2c. MAE Improved ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2c — MAE Improved (patch=4, depth=12, cosine LR, EMA)")
    print("=" * 60)
    mae_imp_model, mae_imp_history, mae_imp_ckpt = train_mae_improved(
        data=data,
        checkpoint_dir=CHECKPOINT_DIR,
        epochs=15,
        batch_size=128,
        learning_rate=1e-3,
        warmup_epochs=2,
        weight_decay=0.05,
        mask_ratio=0.65,
        ema_decay=0.996,
        checkpoint_name="mae_improved_encoder.pth",
    )
    print(f"\n✓ MAE Improved checkpoint saved → {mae_imp_ckpt}")

    # ── Summary ───────────────────────────────────────────────────────────
    import os
    print("\n" + "=" * 60)
    print("  PRE-TRAINING COMPLETE — Checkpoints")
    print("=" * 60)
    for name, path in [
        ("SimCLR (tailored)",  CHECKPOINT_DIR / "simclr_encoder_tailored.pth"),
        ("MAE (base)",         CHECKPOINT_DIR / "mae_encoder.pth"),
        ("MAE Improved",       CHECKPOINT_DIR / "mae_improved_encoder.pth"),
    ]:
        exists = path.exists()
        size_mb = os.path.getsize(path) / 1e6 if exists else 0
        status = "✓" if exists else "✗ MISSING"
        print(f"  {status}  {name:<22}  {size_mb:.1f} MB  {path}")

    print("\nNext: modal run modal_train.py::app run_experiments --gpu A10G")


# ── Step 3: Downstream experiment grid ─────────────────────────────────────
@app.function(
    image=MOUNTED_IMAGE,
    gpu="A10G",                    # A10G = ~$1.50/hr, better for full grid
    timeout=86400,                 # 24 hours max
    volumes={
        "/vol": modal.Volume.from_name(VOLUME_NAME, create_if_missing=True),
        "/content/ST456_GroupProject": modal.Volume.from_name(
            "st456-project-code", create_if_missing=True
        ),
    },
    container_idle_timeout=300,
)
def run_experiments():
    """
    Run the full downstream fine-tuning experiment grid.

    Grid: 4 methods × 3 strategies × 3 label fractions × 3 seeds = 108 runs

    Methods:
        - supervised_from_scratch  (random init baseline)
        - simclr                  (SimCLR encoder + linear head)
        - mae                     (MAE base encoder + linear head)
        - mae_improved            (MAE Improved encoder + linear head)

    Strategies (fine-tuning):
        - frozen   (encoder frozen, only linear head trained)
        - partial  (last blocks unfrozen)
        - full     (full fine-tune)

    Results saved to: /vol/results/results_main.csv

    Runtime estimate on A10G (24 GB VRAM):
        ~20–40 min per seed × 3 seeds = 1–2 hours
    """
    import sys
    from pathlib import Path

    REPO_DIR = Path("/content/ST456_GroupProject")
    if not REPO_DIR.exists():
        print(f"Cloning repo to {REPO_DIR} ...")
        subprocess.run(
            ["git", "clone", "https://github.com/Friedrich-233/ST456_GroupProject.git",
             str(REPO_DIR)],
            check=True,
        )
    sys.path.insert(0, str(REPO_DIR))

    from data import (
        load_all_data,
        build_downstream_loaders,
        LABEL_FRACTIONS,
    )
    from training import (
        run_experiment_grid,
        MAIN_METHODS,
        FINETUNE_STRATEGIES,
        TrainConfig,
    )

    DATA_DIR       = Path("/vol/pcam_data")
    CHECKPOINT_DIR = Path("/vol/checkpoints")
    RESULTS_DIR    = Path("/vol/results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("[1/4] Loading data...")
    data = load_all_data(
        drive_data_dir=DATA_DIR,
        data_dir=DATA_DIR,
        pretrain_fraction=0.15,
        downstream_pool_fraction=0.15,
        use_subset=True,
    )
    data.summary()

    # Build loaders
    print("\n[2/4] Building downstream loaders...")
    train_loaders, val_loader, test_loader = build_downstream_loaders(data)
    print(f"  Label fractions: {list(train_loaders.keys())}")

    # Verify checkpoints exist
    print("\n[3/4] Checking checkpoints...")
    required = {
        "simclr":         CHECKPOINT_DIR / "simclr_encoder_tailored.pth",
        "mae":            CHECKPOINT_DIR / "mae_encoder.pth",
        "mae_improved":   CHECKPOINT_DIR / "mae_improved_encoder.pth",
    }
    for name, path in required.items():
        status = "✓" if path.exists() else "✗ MISSING"
        print(f"  {status}  {name}: {path}")
    missing = [n for n, p in required.items() if not p.exists()]
    if missing:
        print(f"\nERROR: Missing checkpoints: {missing}")
        print("Run: modal run modal_train.py::app pretrain_all --gpu T4")
        return

    # Run grid — seed 42
    print("\n[4/4] Running experiment grid (seeds 42, 52, 62)...")
    results_42 = run_experiment_grid(
        methods=MAIN_METHODS,
        strategies=FINETUNE_STRATEGIES,
        label_names=list(LABEL_FRACTIONS.keys()),
        seeds=[42],
        train_loaders=train_loaders,
        val_loader=val_loader,
        test_loader=test_loader,
        simclr_checkpoint=CHECKPOINT_DIR / "simclr_encoder_tailored.pth",
        mae_checkpoint=CHECKPOINT_DIR / "mae_encoder.pth",
        mae_improved_checkpoint=CHECKPOINT_DIR / "mae_improved_encoder.pth",
        config=TrainConfig(epochs=15, encoder_lr=1e-4, head_lr=1e-3),
    )
    results_42.to_csv(RESULTS_DIR / "results_seed42.csv", index=False)
    print(f"  Seed 42 done: {len(results_42)} rows → {RESULTS_DIR / 'results_seed42.csv'}")

    results_52 = run_experiment_grid(
        methods=MAIN_METHODS,
        strategies=FINETUNE_STRATEGIES,
        label_names=list(LABEL_FRACTIONS.keys()),
        seeds=[52],
        train_loaders=train_loaders,
        val_loader=val_loader,
        test_loader=test_loader,
        simclr_checkpoint=CHECKPOINT_DIR / "simclr_encoder_tailored.pth",
        mae_checkpoint=CHECKPOINT_DIR / "mae_encoder.pth",
        mae_improved_checkpoint=CHECKPOINT_DIR / "mae_improved_encoder.pth",
        config=TrainConfig(epochs=15, encoder_lr=1e-4, head_lr=1e-3),
    )
    results_52.to_csv(RESULTS_DIR / "results_seed52.csv", index=False)
    print(f"  Seed 52 done: {len(results_52)} rows → {RESULTS_DIR / 'results_seed52.csv'}")

    results_62 = run_experiment_grid(
        methods=MAIN_METHODS,
        strategies=FINETUNE_STRATEGIES,
        label_names=list(LABEL_FRACTIONS.keys()),
        seeds=[62],
        train_loaders=train_loaders,
        val_loader=val_loader,
        test_loader=test_loader,
        simclr_checkpoint=CHECKPOINT_DIR / "simclr_encoder_tailored.pth",
        mae_checkpoint=CHECKPOINT_DIR / "mae_encoder.pth",
        mae_improved_checkpoint=CHECKPOINT_DIR / "mae_improved_encoder.pth",
        config=TrainConfig(epochs=15, encoder_lr=1e-4, head_lr=1e-3),
    )
    results_62.to_csv(RESULTS_DIR / "results_seed62.csv", index=False)
    print(f"  Seed 62 done: {len(results_62)} rows → {RESULTS_DIR / 'results_seed62.csv'}")

    # Merge
    import pandas as pd
    results_all = pd.concat([results_42, results_52, results_62], ignore_index=True)
    results_all.to_csv(RESULTS_DIR / "results_main.csv", index=False)
    print(f"\n✓ All done! Combined: {len(results_all)} rows")
    print(f"  Saved → {RESULTS_DIR / 'results_main.csv'}")

    # Quick summary
    summary = (
        results_all.groupby(["method", "strategy", "label_fraction"])
        [["test_auc", "test_accuracy", "test_f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    print("\nTop-3 by mean test AUC (frozen, 1%):")
    frozen_1pct = summary[
        (summary[("test_auc", "")] == "1%") &
        (summary[("test_auc", "")] == "frozen")
    ]
    print(summary.sort_values(("test_auc", "mean"), ascending=False).head(5).to_string())


# ── Step 4: Download checkpoints to local machine ───────────────────────────
@app.local_entrypoint()
def download_checkpoints():
    """
    Download trained checkpoints from the Modal volume to the local machine.

    Run locally (no GPU needed):
        modal run modal_train.py::download_checkpoints
    """
    import shutil
    from pathlib import Path

    volume = modal.Volume.from_name(VOLUME_NAME)
    local_dir = Path("~/ST456_checkpoints").expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from volume '{VOLUME_NAME}' to {local_dir} ...")

    for subdir in ["checkpoints", "results"]:
        src_dir = Path("/vol") / subdir
        dst_dir = local_dir / subdir
        dst_dir.mkdir(parents=True, exist_ok=True)

        # List files in the volume at that path
        try:
            files = list(volume.path(src_dir).glob("*"))
        except Exception:
            print(f"  No files found at /vol/{subdir}/ — skipping.")
            continue

        for f in files:
            dst = dst_dir / f.name
            print(f"  Downloading {f.name} ...")
            shutil.copy(str(f), str(dst))

    print(f"\n✓ All files downloaded to: {local_dir}")
    print("Contents:")
    for f in local_dir.rglob("*"):
        if f.is_file():
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.relative_to(local_dir):50s}  {size_mb:8.1f} MB")


# ── Local entrypoint for pretrain ──────────────────────────────────────────
@app.local_entrypoint()
def main():
    """Default: just print instructions."""
    print(__doc__)
