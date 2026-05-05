# U-Net for Microscopic Cell Segmentation

**Project — University of Milan**  
Machine Learning pipeline for biomedical image segmentation using a U-Net convolutional neural network.

---

## Notebook

▶️ [Open in Google Colab](https://colab.research.google.com/drive/12FByn6LMJ76dZ9WJA7XSiqmouxp8G4QX?usp=sharing)

The notebook includes the full training pipeline with outputs and visual predictions on the test set.

---

## Overview

This project implements a **U-Net** architecture for binary segmentation of microscopic cell images (TIFF format). The model learns to produce pixel-wise masks that identify cell regions in grayscale microscopy images.

The pipeline covers:
- Grayscale image loading and preprocessing
- Binary mask loading and alignment
- U-Net model definition and training
- Prediction visualization (input image, predicted mask, ground truth)
- Output saving as JPEG images

---

## Architecture

The model follows the classic **U-Net** encoder-decoder structure with skip connections:

```
Input (256×256×1)
    │
    ├── Encoder
    │   ├── Block 1: Conv2D(64)  → MaxPool
    │   ├── Block 2: Conv2D(128) → MaxPool
    │   ├── Block 3: Conv2D(256) → MaxPool
    │   └── Block 4: Conv2D(512) → MaxPool
    │
    ├── Bottleneck: Conv2D(1024)
    │
    └── Decoder
        ├── Block 6: ConvTranspose(512) + skip from Block 4
        ├── Block 7: ConvTranspose(256) + skip from Block 3
        ├── Block 8: ConvTranspose(128) + skip from Block 2
        └── Block 9: ConvTranspose(64)  + skip from Block 1
            │
            └── Output: Conv2D(1, sigmoid) → binary mask
```

Each encoder/decoder block includes **BatchNormalization** for training stability.

---

## Training Details

| Parameter | Value |
|---|---|
| Input size | 256 × 256 × 1 (grayscale) |
| Loss | Binary crossentropy |
| Optimizer | Adam |
| Epochs | 10 |
| Batch size | 32 |
| Output activation | Sigmoid |

---

## Sample Results

The notebook contains visual comparisons of:
- **Original Image** — raw grayscale microscopy input
- **Ground Truth Mask** — manually annotated binary segmentation
- **Predicted Mask** — model output after thresholding at 0.5

See the full outputs in the [Colab notebook](https://colab.research.google.com/drive/12FByn6LMJ76dZ9WJA7XSiqmouxp8G4QX?usp=sharing).

---

## Repository Structure

```
├── unet_segmentation.py       # Full pipeline: model, data loading, training, prediction
└── README.md
```

---

## Requirements

```
tensorflow >= 2.x
Pillow
numpy
matplotlib
python-dotenv
```

Install with:
```bash
pip install tensorflow pillow numpy matplotlib python-dotenv
```

---

## Environment Variables

The script uses a `.env` file to configure data paths. Create a `.env` file in the root with:

```
PERCORSO_TRAINING_LABELED=/path/to/training/images
PERCORSO_TRAINING_LABELED_LABELS=/path/to/training/labels
PERCORSO_TUNING=/path/to/tuning/images
PERCORSO_TUNING_LABELS=/path/to/tuning/labels
PERCORSO_OUTPUT=/path/to/output
```

---

## Notes

The dataset used for training is not included in this repository. The Colab notebook contains the full execution with outputs and visual results from the original training run.
