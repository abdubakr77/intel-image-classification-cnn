# 🏔️ Intel Image Classification — Transfer Learning with InceptionV3

> Classifying natural scenes into 6 categories using Transfer Learning on top of a pre-trained InceptionV3 architecture.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)

---

## 🧠 Overview

This project tackles a **multi-class image classification** problem using the Intel Image Classification dataset. Rather than training a CNN from scratch — which would require far more data and time — I chose to leverage **Transfer Learning** with **InceptionV3** pretrained on ImageNet.

The motivation was simple: the dataset (~24K images) is decent but not massive. InceptionV3 brings powerful, battle-tested feature extraction that allows the model to generalize well even with limited fine-tuning.

---

## 📂 Dataset

The dataset contains approximately **24,335 images** across 3 splits:

| Split | Structure | Purpose |
|-------|-----------|---------|
| `seg_train` | Organized by class subfolders | Training |
| `seg_test` | Organized by class subfolders | Validation during training |
| `seg_pred` | Mixed images (no subfolders) | Final real-world prediction |

### 🏷️ Classes (6 Categories)

```
0 → buildings
1 → forest
2 → glacier
3 → mountain
4 → sea
5 → street
```

> **Note:** The `seg_pred` folder contains mixed, unlabeled images — simulating a real deployment scenario where we don't know the ground truth.

---

## 📁 Project Structure

```
📦 Intel-Image-Classification/
├── 📓 intel_image_classification.ipynb   ← Main notebook
├── 📄 README.md
└── 📂 Data/
    ├── seg_train/
    │   ├── buildings/
    │   ├── forest/
    │   ├── glacier/
    │   ├── mountain/
    │   ├── sea/
    │   └── street/
    ├── seg_test/
    │   └── (same structure as seg_train)
    └── seg_pred/
        └── (flat folder — mixed images)
```

---

## 🏗️ Model Architecture

The model is built with **Keras / TensorFlow** using a clean Sequential pipeline:

```
Input (299×299×3)
       ↓
InceptionV3 (pretrained on ImageNet, frozen)
       ↓
GlobalAveragePooling2D  ← built into InceptionV3 via pooling="avg"
       ↓
Dense(6, activation="softmax")
       ↓
Output: probability distribution over 6 classes
```

### 🔒 Why freeze InceptionV3?

I kept `model.layers[0].trainable = False` intentionally. Since the dataset is not huge, unfreezing InceptionV3 at this stage would risk **catastrophic forgetting** — where the pretrained weights get overwritten with noise. The frozen base acts as a powerful fixed feature extractor, while only the classification head is trained from scratch.

> In a future iteration, **fine-tuning** (gradually unfreezing the last N layers) can squeeze out extra performance.

---

## ⚙️ Training Strategy

### Data Preprocessing

```python
ImageDataGenerator(preprocessing_function=preprocess_input)
```

Used InceptionV3's built-in `preprocess_input` (scales pixels to `[-1, 1]`) — the exact preprocessing the model was originally trained with. This is critical for Transfer Learning to work correctly.

> I initially experimented with augmentation (`zoom_range`, `shear_range`, `brightness_range`, `horizontal_flip`), but found it wasn't necessary to achieve strong results with the frozen base. It remains commented out for easy re-enabling.

### Key Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Image Size | `299×299` | Required by InceptionV3 |
| Batch Size (Train) | `100` | Balance between speed and stability |
| Batch Size (Validation) | `16` | Lighter memory footprint |
| Optimizer | `Adam` | Adaptive LR, works well out of the box |
| Loss | `categorical_crossentropy` | Multi-class classification standard |
| Epochs | `10` (max) | EarlyStopping handles the rest |
| Patience | `3` | Stop if val_loss doesn't improve for 3 epochs |

### Callbacks

```python
EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
```

The `restore_best_weights=True` flag ensures we always keep the best checkpoint automatically — no manual saving needed.

---

## 📊 Results

| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | ~94% | ~92.5% |
| Loss | ~0.15 | ~0.20 |

### 📉 Loss Curve
The training loss steadily decreased from ~0.42 → ~0.15, while validation loss stabilized around 0.20. A small but consistent gap appeared after epoch 4.

### 📈 Accuracy Curve
Both curves climbed quickly in the first 2 epochs and plateaued — train at 94%, validation at 92.5%.

### ⚠️ Slight Overfitting Observed

There is a **mild overfitting** present:
- The training accuracy kept improving after epoch 4, while validation accuracy plateaued.
- The gap (~1.5%) is not alarming, but it's a signal.

The model is still generalizing well — 92.5% validation accuracy on a 6-class problem is solid. But this is something to address in future iterations.

---

## 🚀 Future Improvements

- [ ] **Add Dropout** before the final Dense layer to reduce overfitting
- [ ] **Enable Data Augmentation** (already scaffolded in the code, just uncomment)
- [ ] **Experiment with other architectures** — EfficientNetV2, VGG, ConvNeXt
- [ ] **Use ReduceLROnPlateau** callback alongside EarlyStopping
- [ ] **Use CheckPoint** callback

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| TensorFlow / Keras | Model building & training |
| InceptionV3 | Pretrained base model |
| NumPy | Array operations |
| Pandas | Prediction dataframe |
| Matplotlib | Visualization |

---

## ▶️ How to Run

1. **Clone the repo and set up your data:**
```
Data/
├── seg_train/
├── seg_test/
└── seg_pred/
```

2. **Install dependencies:**
```bash
pip install tensorflow numpy pandas matplotlib
```

3. **Run the notebook:**
```bash
jupyter notebook intel_image_classification.ipynb
```

---

## 📬 Notes

- The `seg_pred` folder is handled differently from train/test — it uses `flow_from_dataframe` instead of `flow_from_directory` since it has no class subfolders.
- Labels are automatically inferred from the `seg_train` folder structure using `os.listdir`.
- Predictions include a **confidence score** (the max softmax probability) displayed alongside each image.

---

*Built as part of a deep learning portfolio project exploring Transfer Learning on real-world image data.*
=======
