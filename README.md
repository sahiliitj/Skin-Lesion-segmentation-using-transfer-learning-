# 🧠 Predicting Segmentation Masks for ISIC 2016 Dataset using MobileNet Encoder

## 📘 Overview

This project is part of the *CSL7590: Deep Learning* course at the Indian Institute of Technology, Jodhpur. The objective is to build a segmentation model that can accurately predict lesion masks for dermoscopic images from the ISIC 2016 dataset. The model architecture is based on **Transfer Learning**, utilizing a **pre-trained MobileNet encoder** with a **custom decoder** for semantic segmentation.

We explore two approaches:
- **Task 1**: Using a frozen MobileNet encoder for feature extraction.
- **Task 2**: Fine-tuning the MobileNet encoder for improved segmentation performance.

---

## 📁 Dataset: ISIC 2016

The **ISIC (International Skin Imaging Collaboration) 2016** dataset contains dermoscopic images annotated for skin lesion segmentation and melanoma detection.

- **Training Images**: 900
- **Test Images**: 379
- **Image Format**: JPEG (images), PNG (masks)
- **Preprocessed Image Size**: `128 × 128`
- **Masks**: Binary (1-channel grayscale)

---

## ⚙️ Pre-processing Pipeline

1. **Resizing** images and masks to a fixed size of `128x128`.
2. **Custom Dataloader** to read images and corresponding masks.
3. **Data Splitting**: 80% training, 20% validation.
4. **Data Augmentation**: Horizontal flip, vertical flip, and color jitter to boost generalization.
5. **Image-Mask Matching Validation** to ensure correctness.

---

## 🔍 Data Analysis

- **Mean Pixel Value**: `0.6366`
- **Pixel Standard Deviation**: `0.1514`
- **Image Size Distribution** visualized.
- Random sample visualizations to validate augmentation pipeline.

---

## 🏗️ Network Architecture

The architecture consists of:
- **Encoder**: Pre-trained **MobileNet** (ImageNetV1 weights).
- **Decoder**: Custom-built using 5 layers of `ConvTranspose2D` followed by a `1x1 Convolution` layer.

### Decoder Architecture
Each decoder layer:
- Transposed Convolution (stride=2, padding=1, kernel=4x4)
- Channels: `[1280 → 512 → 256 → 128 → 64 → 32 → 1]`

### Regularization & Optimizer
- **Dropout**: To reduce overfitting.
- **Loss Function**: Binary Cross Entropy with Logits Loss.
- **Optimizer**: AdamW
  - Task 1: LR = 0.001
  - Task 2: LR = 0.0001 (for fine-tuning encoder)

---

## 🔄 Methodology

### Task 1: Feature Extraction
- Freeze encoder weights.
- Train only the decoder.

### Task 2: Fine-Tuning
- Allow encoder weights to update with a small learning rate.
- Enhances feature learning for segmentation-specific task.

### Training Details
- **Epochs**: 20
- **Metrics**:
  - Intersection Over Union (IoU)
  - Dice Score
  - Pixel-wise Accuracy

### Workflow
1. Forward pass through encoder and decoder.
2. Calculate loss and backpropagate.
3. Track metrics per epoch.
4. Visualize performance with actual vs predicted masks.

---

## 📈 Results & Observations

### 📌 Task 1: Feature Extraction

| Metric | Value |
|--------|-------|
| Mean Train Loss | 0.3031 |
| Mean Validation Loss | 0.3639 |
| Mean Test Loss | 0.4027 |
| Max IoU | 0.5585 |
| Max Dice Score | 0.7167 |
| Pixel-wise Accuracy | 81.5% |

### 📌 Task 2: Fine-Tuning Encoder

| Metric | Value |
|--------|-------|
| Mean Train Loss | 0.1482 |
| Mean Validation Loss | 0.2054 |
| Mean Test Loss | 0.2370 |
| Max IoU | 0.7165 |
| Max Dice Score | 0.8348 |
| Pixel-wise Accuracy | 89.7% |

---

## 📊 Comparative Analysis

| Metric | Task 1 | Task 2 |
|--------|--------|--------|
| IoU Score | 0.5585 | **0.7165** |
| Dice Score | 0.7167 | **0.8348** |
| Accuracy | 81.5% | **89.7%** |

- Fine-tuning the encoder drastically improves segmentation performance.
- Demonstrates the importance of domain adaptation in transfer learning.
- Visualizations reveal improved boundary detection in fine-tuned model.

---

## 📌 Key Takeaways

- **Transfer Learning** enables quick convergence and effective use of limited medical data.
- **Fine-tuning** improves results significantly over frozen encoder.
- **MobileNet** offers a lightweight and efficient encoder backbone for segmentation tasks.
- Use of augmentation and correct mask pairing is crucial for effective model training.

---

## 📚 Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/beginner/nlp/pytorch_tutorial.html)
- [ConvTranspose2D Documentation](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)
- [Segmentation Models PyTorch](https://segmentation-models-pytorch.readthedocs.io/en/latest/models.html)
- Instructor’s lecture slides

---

## 🧑‍🎓 Author

**Name:** Sahil  
**Program:** M.Sc - M.Tech (Data and Computational Sciences)  
**Institution:** Indian Institute of Technology, Jodhpur  

---

> *“Efficient segmentation for clinical diagnostics can be vastly improved with targeted transfer learning — this project demonstrates one such successful attempt.”*
