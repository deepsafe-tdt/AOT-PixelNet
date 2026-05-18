# AOT-PixelNet: Lightweight and Interpretable Detection of Forged Images via Adaptive Orthogonal Transform

This is the official implementation of **AOT-PixelNet: Lightweight and interpretable detection of forged images via adaptive orthogonal transform**.
**Paper DOI**: [https://doi.org/10.1016/j.asoc.2025.113873](https://doi.org/10.1016/j.asoc.2025.113873)
---


## ✨ Highlights


- Proposed the Adaptive Orthogonal Transform (AOT) module.
- Extended the AOT to random transformations.
- Introduced a minimalist PixelNet using only 1 × 1 convolutions.
- Achieved a 2.96% accuracy improvement on benchmark datasets.
- AOT reveals frequency-domain artifacts, enhancing interpretability.

---

## 📂 Project Structure

```text
.
├── configs.py                 # Global configurations
├── main.py                    # Training and validation entry
├── test_model.py              # Single test-set evaluation
│
├── dataloader/
│   └── data_loader.py         # Dataset loader
│
├── models/
│   └── PixelNet.py            # PixelNet definition
│
├── utils/
│   ├── gen_matrix.py          # Orthogonal transform matrix generation
│   └── utils.py               # Training/testing utilities
│
└── results/                   # Logs, weights, figures
```

---

## 🛠️ Requirements

- Python >= 3.7
- PyTorch >= 1.7.0
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn

Install dependencies:

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

---

## ⚙️ Dataset Preparation

Please configure dataset paths in `configs.py`:

- `train_dir`
- `val_dir`
- `test_dir`

Recommended dataset structure:

```text
Dataset_Root/
├── train/
│   ├── car/
│   │   ├── 0_real/    # Real images. Label: 0
│   │   └── 1_fake/    # Generated images. Label: 1
│   ├── cat/
│   │   ├── 0_real/
│   │   └── 1_fake/
│   ├── chair/
│   │   ├── 0_real/
│   │   └── 1_fake/
│   └── horse/
│       ├── 0_real/
│       └── 1_fake/
└── val/
    └── ... (similar structure)
└── test/
    └── ... (similar structure)
```



The experiments in this project are mainly based on the following public datasets:
- CNNDetection Dataset. GitHub Repository: https://github.com/PeterWang512/CNNDetection
- GenImage Dataset. GitHub Repository: https://github.com/GenImage-Dataset/GenImage


---

## 🚀 Usage

### 1. Configure Parameters

Edit hyperparameters and dataset paths in:

```bash
configs.py
```

---

### 2. Train the Model

```bash
python main.py
```

Best model weights will be saved to:

```text
results/parameter.pth
```

> Note: The `results/` directory will be automatically cleared before training starts.

---

### 3. Evaluate the Model

```bash
python test_model.py
```

The evaluation script supports:

- Accuracy
- Confusion Matrix
- Average Precision (AP)

---

## 📊 Interpretability

This project supports multiple frequency-domain interpretability analyses, including:

- Frequency energy visualization
- Orthogonal transform spectrum analysis
- SHAP-based frequency contribution analysis
- Grad-CAM visualization
- Frequency-band importance analysis

These analyses help reveal the intrinsic differences between real and generated images in the transform domain.

---

## 📌 Citation

If you find this project useful, please cite:

```bibtex
@article{TAN2025113873,
  title = {AOT-PixelNet: Lightweight and interpretable detection of forged images via adaptive orthogonal transform},
  journal = {Applied Soft Computing},
  volume = {185},
  pages = {113873},
  year = {2025},
  issn = {1568-4946},
  doi = {https://doi.org/10.1016/j.asoc.2025.113873},
  url = {https://www.sciencedirect.com/science/article/pii/S156849462501186X},
  author = {Dengtai Tan and Chengyu Niu and Yang Yang and Deyi Yang and Boao Tan},
}
```

---

## 📧 Contact

For questions or collaborations, please open an issue or contact the authors.

---

## ⭐ Acknowledgement

We sincerely thank the authors of:

- CNNDetection
- GenImage
- PyTorch

for providing valuable open-source resources to the research community.
