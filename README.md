# Pneumonia Detection from Chest X-Rays

A convolutional neural network built with **TensorFlow/Keras** that classifies paediatric chest X-ray images as **NORMAL** or **PNEUMONIA**.

The project has two iterations: a quick baseline CNN, then a deeper, redesigned model with grayscale preprocessing, class weighting and early stopping.

> ⚠️ This is an educational project. It is not a medical device and must not be used for diagnosis.

---

## Dataset

The [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle (Kermany et al.), arranged as:

```
data/
  train/ NORMAL/  PNEUMONIA/
  val/   NORMAL/  PNEUMONIA/
  test/  NORMAL/  PNEUMONIA/
```

| Split | Normal | Pneumonia | Total |
|---|---:|---:|---:|
| Train | 1,000 | 3,875 | 4,875 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

The training set is heavily imbalanced (~1 : 3.9), which shapes several of the design decisions below.

---

## V1 — Baseline CNN ([`classifier.ipynb`](classifier.ipynb))

- **Cleaning:** scans every split and removes corrupt or non-image files (OpenCV + `filetype`)
- **Input:** 256×256 RGB, pixels scaled to [0, 1], loaded with `image_dataset_from_directory`
- **Architecture:** 3 × (Conv2D → MaxPool) with 16/32/16 filters → Dense(256) → sigmoid output (~3.7M parameters)
- **Training:** Adam, binary cross-entropy, 20 epochs, TensorBoard logging, `cache()`/`prefetch()` input pipeline

**Test-set results**

| Metric | Score |
|---|---:|
| Accuracy | 76.9% |
| Precision | 73.8% |
| Recall (sensitivity) | **97.7%** |

The model catches almost every pneumonia case but produces many false positives. Training accuracy reached 100%, which shows it was overfitting.

---

## V2 — Deeper CNN ([`classifierV2.ipynb`](classifierV2.ipynb))

Changes made to tackle the overfitting and the class imbalance:

- **Preprocessing:** images converted to **grayscale** and resized to **196×196** with OpenCV. X-rays carry no colour information, so this cuts input size by two thirds.
- **Architecture:** five convolutional blocks, each with two stacked Conv2D layers (8 → 16 → 32 → 64 → 128 filters, kernels 7×7 → 5×5 → 3×3), followed by Dense(128), **Dropout(0.2)** and a 2-class softmax
- **~370K parameters**, about 10× fewer than V1 despite being much deeper
- **Class weighting** `{normal: 6.0, pneumonia: 0.5}` to stop the model defaulting to the majority class
- **Optimiser:** Adam (lr = 1e-4) with categorical cross-entropy
- **Early stopping** (patience 6) to stop training once loss plateaus
- Exploratory plots of the class distribution and sample images from each class

**Result:** about **82% accuracy on the 624-image test set**, up from 77% in V1. The final model is saved to [`models/imageClassifier2.h5`](models/).

---

## Tech stack
Python · TensorFlow / Keras · OpenCV · NumPy · pandas · Matplotlib · seaborn · TensorBoard · Jupyter

## Running it

```bash
git clone https://github.com/GeorgeBarnes06/Pneumonia-image-classifier.git
cd Pneumonia-image-classifier
pip install tensorflow opencv-python filetype numpy pandas matplotlib seaborn jupyter
jupyter notebook classifierV2.ipynb
```

If the `data/` folder is missing, download the dataset from Kaggle and extract it into `data/` using the structure above. A GPU is optional but speeds up training a lot.

To view the training curves:
```bash
tensorboard --logdir logs
```

---

## Lessons learned and next steps
- **Recall matters most here.** A missed pneumonia case is far worse than a false alarm, so recall and the confusion matrix are more meaningful than raw accuracy.
- **The validation split is tiny (16 images).** V2 therefore monitored the test set during training, which makes its test score slightly optimistic. Next step: carve a proper validation set (~10–15%) out of the training data and keep the test set untouched until the end.
- [ ] Data augmentation (small rotations, zoom, brightness)
- [ ] Transfer learning from a pretrained backbone (e.g. EfficientNet or DenseNet121)
- [ ] Report a confusion matrix, ROC-AUC and per-class F1
- [ ] Grad-CAM heatmaps to show which lung regions drive each prediction
