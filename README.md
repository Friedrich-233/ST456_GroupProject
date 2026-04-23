# ST456 Group Project — PCam Self-Supervised Learning
SimCLR vs MAE on PatchCamelyon histopathology classification, with PEFT (LoRA/VPT) ablations.
---

## Quick Start

1. Open `main.ipynb` in **Google Colab** with a **GPU runtime** (L4 is fine).

2. And set the data's directory to /MyDrive/ST456 Group project/pcamv1/

3. **If your Drive path is different**, edit `DRIVE_DATA_DIR` in **Section 0** of `main.ipynb`:
```python
   DRIVE_DATA_DIR = Path('/content/drive/MyDrive/YOUR/PATH/HERE/pcamv1')
```

4. Run **Section 0** first (~5 minutes — mounts Drive, clones repo, loads PCam data).

5. After Section 0 finishes, any of Section 1 / 2 / 3 can be run independently.

## Notebook Structure

| Section | What it does | Runtime (T4) | Depends on |
|---|---|---|---|
| **Section 0** — Setup | Mounts Drive, clones repo, loads PCam data | ~5 min | — |
| **Section 1** — SSL Pre-training | Trains SimCLR (tailored + generic) and MAE encoders | ~3 hours | Section 0 |
| **Section 2** — Downstream Grid | Runs the full (method × strategy × label × seed) experiment grid | ~4–5 hours per seed | Section 0 + pretrained checkpoints |
| **Section 3** — Analysis | Generates all report figures (heatmaps, ROC, t-SNE, Grad-CAM, Pareto) | ~15 min | Section 0 + results CSVs |

**Key point**: Once Section 0 has been run, you can pick up any later section in the same session without re-running earlier ones, **as long as the required checkpoints / CSVs already exist** (either from your own previous runs or pulled from the repo).

## Code structure
- `code/data.py` — PCam loading, augmentation, datasets, DataLoaders
- `code/models.py` — SimCLR, MAE architectures and downstream classifiers
- `code/training.py` — SSL pre-training, fine-tuning, experiment orchestration
- `code/evaluation.py` — Metrics, result tables, t-SNE, Grad-CAM

## Key path
- `checkpoint` store all the trained models
- `result` store all the experiment results

## Common Workflows

### Just want to regenerate the report plots
→ Run Section 0, then jump straight to Section 3. The CSVs in `results/` are already committed.

### Want to re-train only the downstream classifiers
→ Run Section 0, then Section 2. Section 2 uses the pretrained encoders from `checkpoints/`.

### Running the generic SimCLR ablation
→ Section 2 has an optional cell near the end (`results_simclr_generic`). Run Section 0 + Section 1's SimCLR-generic cell (Cell 8) first if the `simclr_encoder_generic.pth` checkpoint is missing.
