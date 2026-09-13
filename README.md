# Camel Beauty Assessor (Computer Vision)

![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red)

Detects the beauty traits judges look for in a camel photo and returns each one with a confidence,
ready for a language model to write up.

> **Developed as part of the Tuwaiq Bootcamp.**

This repo is the computer vision half of the project. The platform and language model live in
[Camel-Beauty-Assessor-Platform](https://github.com/Renadyousef/Camel-Beauty-Assessor-Platform).

* [Overview](https://github.com/Rmsaah/camel-beauty-assessor?tab=readme-ov-file#overview)
* [How it Works](https://github.com/Rmsaah/camel-beauty-assessor?tab=readme-ov-file#how-it-works)
* [Results](https://github.com/Rmsaah/camel-beauty-assessor?tab=readme-ov-file#results)
* [Models](https://github.com/Rmsaah/camel-beauty-assessor?tab=readme-ov-file#models)
* [Setup Instructions](https://github.com/Rmsaah/camel-beauty-assessor?tab=readme-ov-file#setup-instructions)
* [Limitations](https://github.com/Rmsaah/camel-beauty-assessor?tab=readme-ov-file#limitations)
* [Team Members](https://github.com/Rmsaah/camel-beauty-assessor?tab=readme-ov-file#team-members)
* [License](https://github.com/Rmsaah/camel-beauty-assessor?tab=readme-ov-file#license)

## Overview

Camel beauty judging is qualitative. Judges weigh named physical traits like the curve of the neck
or the size of the hump, and nothing gets written down.

This model reads a photo and reports which traits it can see and how sure it is about each one. It
does not rank camels or predict a placing.

Nine classes: `Camel`, `High_withers`, `Large-head`, `Large-lips`, `Large-nose`, `Large_hump`,
`Long-legs`, `Long-neck`, `Wide_body`.

Dataset: 195 annotated photos, split 137 train / 30 val / 28 test. The images are not published here.

## How it Works

1. **EDA** (`dataset_eda.ipynb`) checks the class schema, finds near duplicates, merges the export.
2. **Preprocessing** (`preprocessing.ipynb`) draws the split and writes RGB, grayscale and rotated copies.
3. **Training** across `baseline`, `fine-tuning_experiments`, `freezing_experiments` and
   `hyperparameter_tuning`. 69 runs, each changing one variable so the result can be attributed.
4. **Analysis** (`trait_analysis.ipynb`, `version_comparison.ipynb`) scores the best model per trait
   and per group, and compares the two dataset versions.

Every run is tracked in MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Results

| model | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|
| `e1-gray--aug+` | 0.8113 | 0.4501 | 0.7658 | 0.8271 |
| `b1-rgb--aug+` | 0.8323 | 0.4484 | 0.7757 | 0.7625 |

Two things mattered.

Re-annotating the dataset beat every hyperparameter. All 29 models improved on the second annotation
pass, with a median gain of +0.087 mAP50-95.

Stronger augmentation was the only training change that helped. Rotating images on disk, freezing the
backbone, raising the image size and training longer all made things worse.

## Models

| file | input | mAP50-95 | use for |
|---|---|---|---|
| `export/model_gray.pt` | grayscale | 0.4501 | ranking, higher recall |
| `export/model_rgb.pt` | rgb | 0.4484 | written claims, higher precision |

Output is one record per image: the camel's confidence, then all eight traits with a confidence and a
box. A trait that was not found scores 0.0, so the shape never changes and there is no threshold.

`export/model_card.json` holds the settings and metrics. `export/usage.py` is a working example.

## Setup Instructions

1. Create the environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. On Windows with an NVIDIA GPU, install CUDA torch first, otherwise pip gives you the CPU build:

```bash
pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision
```

3. Run a prediction:

```python
from PIL import Image, ImageOps
from ultralytics import YOLO

model = YOLO("export/model_rgb.pt")
img = ImageOps.exif_transpose(Image.open("camel.jpg")).convert("RGB")
r = model.predict(img, conf=0.01, imgsz=640, verbose=False)[0]
```

Ultralytics handles the resizing and normalisation. For `model_gray.pt` use
`.convert("L").convert("RGB")` instead.

## Limitations

- 195 photos. With 28 test images, small differences between models are noise.
- `Wide_body` and `Large-lips` sit near 28 pixels at 640, where detectors start to fail.
- It reports traits, not a verdict. It does not rank camels or replace a judge.

## Team Members

- [Joud](https://github.com/joudterad-spec)
- [Reema Al Jbreen](https://github.com/Rmsaah)
- [Renad Yousef](https://github.com/Renadyousef)
- [Shahad](https://github.com/shahad-abdullah-d)

## License

**All Rights Reserved.**

This project and its source code are proprietary to the project team.

The code, models, datasets, documentation, and other project materials may **not be copied, modified,
distributed, published, or used for commercial or personal projects without explicit written
permission from the project team.**

© 2026 Joud, Reema, Renad, and Shahad.
