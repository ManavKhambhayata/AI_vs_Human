# 🧠 CNN + KMeans Hybrid Classifier for AI vs Human Image Detection

This project implements a powerful hybrid classifier that uses a **ConvNeXt CNN model** and **KMeans clustering** to classify images as either **AI-generated** or **Human-generated**.

Achieved score: **92%+ F1** on Kaggle private leaderboard.

---

## 📦 Key Python Modules and Dependencies

```bash
pip install pytorch-lightning monai torchmetrics scikit-image
```

Used libraries:
- `torch`, `torchvision`, `pytorch_lightning`
- `monai.transforms` for sigmoid activations
- `skimage`, `scipy`, `cv2`, `PIL` for image statistics

---

## 🧠 Code Breakdown & Function Explanations

### 1. `ImageDataset`
PyTorch Dataset class that loads image files and applies transforms. Used for training, validation, and test sets.

### 2. `Net (LightningModule)`
Custom PyTorch Lightning model wrapper:
- `forward()` for predictions
- `training_step()` for calculating binary cross entropy loss
- `validation_step()` for computing validation F1 score
- `configure_optimizers()` returns optimizer and scheduler

### 3. `train_and_predict(base_dir)`
Master pipeline:
- Loads train/test CSVs and images
- Splits training set (95% train, 5% val)
- Applies augmentations
- Builds and fine-tunes `ConvNeXt-Base` model
- Trains using PyTorch Lightning
- Predicts test set logits using `sigmoid`
- Visualizes predicted probability distribution
- Generates `submission.csv`

### 4. CNN Model Building
```python
model = models.convnext_base(weights='DEFAULT')
```
- Freezes all features except the last two blocks
- Replaces classifier head with:
  - `AdaptiveAvgPool2d`
  - `BatchNorm1d`, `Linear(1024->512)->ReLU->Dropout->Linear(512->1)`


### 6. KMeans & Image Statistics (from testing_kmeans.ipynb)
Statistical features extracted per image using:
- Image size, ratio
- Laplacian smoothness
- Median noise difference
- Channel-wise mean, std, quantiles, kurtosis, skewness
- Otsu threshold

Used for clustering into 2 groups via `KMeans(2)`.

### 7. Ensemble Logic
- CNN predictions sorted by confidence (sigmoid outputs)
- Top-1500 images with highest CNN confidence are fixed as AI (`label=1`)
- Lowest 50 as Human (`label=0`)
- Final predictions: KMeans + corrected CNN injection

