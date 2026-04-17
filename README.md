## Code structure
- `code/data.py` — PCam loading, augmentation, datasets, DataLoaders
- `code/models.py` — SimCLR, MAE architectures and downstream classifiers
- `code/training.py` — SSL pre-training, fine-tuning, experiment orchestration
- `code/evaluation.py` — Metrics, result tables, t-SNE, Grad-CAM

## Key path
- `checkpoint` store all the trained models
- `result` store all the experiment results


## How to run
1. `01_pretrain_ssl.ipynb` — train the 3 SSL encoders and save checkpoints
2. `02_main_experiments.ipynb` — run the full downstream experiment grid
3. `03_analysis.ipynb` — load results and produce report figures