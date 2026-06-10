# Self-Supervised Pretraining for Chest X-ray Classification

Does self-supervised pretraining on unlabeled medical images beat training from
scratch or transfer learning from ImageNet? This project runs that comparison on
the **CheXpert** chest X-ray dataset using a **Masked Autoencoder (MAE)**-style
pretext task and a Vision Transformer (ViT) backbone.

> **Motivation.** Labeled medical data is scarce and expensive (each label needs a
> radiologist). Self-supervised learning (SSL) lets a model learn useful visual
> representations from *unlabeled* scans first, so the downstream classifier needs
> far fewer labels. This repo tests whether that holds on CheXpert.

---

## Approach

1. **Pretrain (self-supervised).** A ViT-Base encoder is trained on unlabeled
   chest X-rays with a masked-image reconstruction objective: parts of the image
   are masked and the model learns to reconstruct them. No labels used.
2. **Finetune (supervised).** The pretrained encoder is attached to a linear head
   and trained for multi-label classification of 5 findings: *Atelectasis,
   Cardiomegaly, Consolidation, Edema, Pleural Effusion*.
3. **Compare.** The same architecture is finetuned under three initializations:

   | Mode          | Encoder initialization                    |
   |---------------|-------------------------------------------|
   | `scratch`     | Random weights                            |
   | `imagenet`    | ImageNet-pretrained (transfer learning)   |
   | `medical_ssl` | This project's SSL-pretrained encoder     |

Evaluation metric: **macro-averaged ROC-AUC** across the 5 findings.

---

## Results

> Replace the values below with your actual runs. Do not report numbers you have
> not measured.

| Mode          | Macro AUC | Notes                          |
|---------------|:---------:|--------------------------------|
| `scratch`     |   _TBD_   | baseline, no pretraining       |
| `imagenet`    |   _TBD_   | transfer learning baseline     |
| `medical_ssl` |   _TBD_   | SSL pretraining (this project) |

_Key takeaway: (fill in once you have numbers — e.g. "SSL pretraining improved
macro AUC by X points over scratch under a limited-label regime")._

---

## Dataset

[CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) — a public
dataset of chest radiographs with multi-label findings. The small variant
(`CheXpert-v1.0-small`) is used here. Uncertain labels (`-1`) and missing labels
are mapped to `0`.

The dataset is **not** included in this repo and must be downloaded separately
(see the link above; access requires agreeing to Stanford's terms).

---

## Repository structure

```
medical_ssl/
├── models/
│   ├── vit.py            # ViT backbone (timm)
│   └── mae.py            # Masked-autoencoder pretext model
├── notebooks/            # Exploratory analysis
├── dataset.py            # CheXpert Dataset + DataLoader
├── pretrain_mae.py       # Self-supervised pretraining entry point
├── finetune.py           # Supervised finetuning + evaluation
├── run_kaggle.py         # Single-file end-to-end script for Kaggle GPUs
├── utils.py              # Checkpoint + metric helpers
├── train_config.yaml     # Hyperparameters
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/ctrlc0704/medical_ssl.git
cd medical_ssl
pip install -r requirements.txt
```

## Usage

**1. Self-supervised pretraining**

```bash
python pretrain_mae.py \
    --csv  path/to/train.csv \
    --root_dir path/to/images \
    --epochs 50
```

Produces `mae_encoder.pth`.

**2. Finetune and evaluate**

```bash
python finetune.py \
    --train_csv path/to/train.csv \
    --val_csv   path/to/valid.csv \
    --root_dir  path/to/images \
    --mode      medical_ssl     # or: scratch | imagenet
```

Prints the macro AUC for the chosen initialization.

**Kaggle (single GPU).** `run_kaggle.py` runs the full pretrain → finetune →
evaluate loop in one file, pathed for the Kaggle CheXpert dataset mount.

---

## Tech stack

PyTorch · timm · scikit-learn · pandas · Vision Transformer (ViT-Base/16) ·
Masked image modeling · Multi-label classification

## Roadmap / known limitations

- The masking is currently pixel-level with a pooled latent + MLP decoder; a
  patch-level MAE (encoder sees only visible patches, decoder reconstructs masked
  patches) would be closer to the original method and is the next step.
- Report results under a **limited-label** setting (e.g. 1% / 10% of labels), which
  is where SSL is expected to help most.
- Add data augmentation and longer pretraining schedules.

## License

MIT (add a `LICENSE` file).
