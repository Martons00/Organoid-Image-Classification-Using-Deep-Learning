# Organoid Image Classification Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-latest-green.svg)](https://monai.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive deep learning framework for **3D medical image classification** of organoids using state-of-the-art transformer and CNN architectures. This project is part of a master's thesis research conducted at **Inria Sophia Antipolis** and implements multiple architectures with flexible configuration for medical image analysis.

---

## 🔬 Overview

This repository implements a complete pipeline for classifying 3D organoid images using various deep learning architectures, including:

- **SwinUNETR** (Swin Transformer-based UNETR)
- **ResNet50/ResNet18** (3D implementations)
- **DenseNet201** (3D implementation)

The framework supports multiple classification heads (Linear, ML-Decoder, NOAH), advanced loss functions, data augmentation techniques, and distributed training capabilities.

### Key Features

- ✅ **Multiple 3D Architectures**: SwinUNETR, ResNet, DenseNet with pretrained weights
- ✅ **Advanced Classification Heads**: Linear, ML-Decoder, NOAH
- ✅ **Flexible Loss Functions**: Cross-Entropy, Focal Loss, Label Smoothing, Diversity Loss, Center Loss
- ✅ **Class Imbalance Handling**: Balanced sampling, class-weighted losses, data augmentation
- ✅ **Distributed Training**: Multi-GPU support with PyTorch DDP
- ✅ **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, Specificity metrics
- ✅ **Experiment Tracking**: TensorBoard integration, automatic logging
- ✅ **Grid5000 Cluster Support**: OAR job scheduling, Telegram notifications

---

## 📁 Project Structure

```
Organoid-Image-Classification-Using-Deep-Learning/
│
├── config/                     # Configuration files
│   ├── default.py             # Default configuration parameters
│   ├── template.yaml          # Configuration template
│   └── training/              # Training-specific configs
│
├── dataset/                    # Dataset implementations
│   └── organoidINRIA_custom.py # Custom organoid dataset class
│
├── models/                     # Model architectures
│   ├── SwinUNETREncoder_3D.py # SwinUNETR encoder wrapper
│   ├── ResNet50_3D.py         # ResNet 3D wrapper
│   ├── DenseNet_3D.py         # DenseNet 3D wrapper
│   ├── ML_Decoder_main/       # ML-Decoder implementation
│   └── NOAH_main/             # NOAH classification head
│
├── utils/                      # Utility functions
│   ├── trainer.py             # Training and testing loops
│   ├── data_utils.py          # Data loading and preprocessing
│   └── utils.py               # General utilities
│
├── tools/                      # Additional tools
│   └── loss.py                # Custom loss functions
│
├── optimizers/                 # Custom optimizers and schedulers
│   └── lr_scheduler.py        # Learning rate schedulers
│
├── preprocessing/              # Data preprocessing scripts
│
├── train.py                    # Main training script
├── testing.py                  # Testing script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA 12.x (for GPU support)
- 32GB+ RAM recommended
- GPU with 16GB+ VRAM (for training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Martons00/Organoid-Image-Classification-Using-Deep-Learning.git
cd Organoid-Image-Classification-Using-Deep-Learning
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install MONAI from source** (if not already installed)
```bash
pip install git+https://github.com/Project-MONAI/MONAI.git
```

### Dataset Structure

Organize your dataset in the following structure:

```
data/
├── train_set/
│   ├── class_0/
│   │   ├── sample_001.nii.gz
│   │   ├── sample_002.nii.gz
│   │   └── ...
│   ├── class_1/
│   └── class_2/
│
└── test_set/
    ├── class_0/
    ├── class_1/
    └── class_2/
```

---

## 💻 Usage

### Training

#### Basic Training

```bash
python train.py \
    --cfg config/training/your_config.yaml \
    --model_name swinunetr \
    --batch_size 4 \
    --max_epochs 100 \
    --optim_lr 1e-4
```

#### Training with Configuration File

Create a YAML configuration file (see `config/template.yaml`):

```yaml
DATASET:
  DATA_DIR: /path/to/dataset
  ROI_X: 96
  ROI_Y: 96
  ROI_Z: 96
  IN_CHANNELS: 1

MODEL:
  NAME: swinunetr
  NUM_CLASSES: 3
  PRETRAINED: true
  PRETRAINED_DIR: ./pretrained_models

TRAIN:
  BATCH_SIZE: 4
  MAX_EPOCHS: 100
  OPTIMIZER: adamw
  LR: 1e-4
  WEIGHT_DECAY: 1e-5
  LOSS: FocalLoss
```

Then run:
```bash
python train.py --cfg config/training/your_config.yaml
```

#### Distributed Training (Multi-GPU)

```bash
python train.py \
    --cfg config/training/your_config.yaml \
    --distributed \
    --world_size 4 \
    --dist_url 'tcp://127.0.0.1:23456'
```

### Testing

```bash
python testing.py \
    --cfg config/testing/your_config.yaml \
    --checkpoint_path outputs/training/best_model.pt \
    --model_name swinunetr
```

### Grid5000 Cluster Usage

Submit a job using OAR:

```bash
oarsub -l gpu=1,walltime=10:00:00 ./run.sh
```

The `run.sh` script handles:
- Environment setup
- GPU allocation
- Training execution
- Telegram notifications (optional)

---

## 🏗️ Model Architectures

### SwinUNETR

Swin Transformer-based U-Net architecture for 3D medical image analysis.

```bash
python train.py --model_name swinunetr --batch_size 4
```

**Supported heads:**
- `swinunetr`: Standard linear classification head
- `swinunetr+ml_decoder`: ML-Decoder head
- `swinunetr+noah`: NOAH attention head

### ResNet50/ResNet18

3D ResNet architectures adapted from MedicalNet.

```bash
python train.py --model_name resnet50 --batch_size 8
python train.py --model_name resnet18 --batch_size 16
```

### DenseNet201

3D DenseNet architecture from MONAI.

```bash
python train.py --model_name densenet --batch_size 6
```

---

## 📊 Loss Functions

The framework supports multiple loss functions for handling class imbalance:

- **CrossEntropyLoss** (`CE`): Standard cross-entropy with class weights
- **FocalLoss** (`FocalLoss`): Focuses on hard-to-classify samples
- **LabelSmoothingLoss**: Prevents overconfidence
- **DiversityLoss**: Encourages feature diversity
- **CombinedLoss**: Combines Focal + Diversity losses
- **CenterLoss**: Learns discriminative features

Example:
```bash
python train.py --loss_name FocalLoss
```

---

## 🔧 Training Configuration

### Learning Rate Schedulers

- `warmup_cosine`: Warmup + Cosine annealing
- `cosine_anneal`: Standard cosine annealing
- `cosine_restarts`: SGDR (Stochastic Gradient Descent with Warm Restarts)
- `warmup_cosine_restarts`: Warmup + SGDR

### Data Splitting Methods

- `random`: Random train/val split
- `stratified`: Maintains class distribution
- `balanced`: Ensures equal class representation
- `percentage`: Percentage-based split

### Data Augmentation

The framework uses MONAI transforms for 3D augmentation:
- Random rotation, flipping, scaling
- Intensity normalization
- Elastic deformation
- Gaussian noise

---

## 📈 Monitoring and Logging

### TensorBoard

Monitor training in real-time:

```bash
tensorboard --logdir outputs/
```

### Metrics Tracked

- Training/Validation Loss
- Accuracy (per-class and macro-averaged)
- Precision, Recall, F1-Score
- Specificity
- Confusion matrices

### Telegram Notifications (Optional)

Configure Telegram bot for training updates:

```bash
python train.py --telegram_log --oar_id YOUR_JOB_ID --token path/to/token.txt
```

---

## 🧪 Experiment Management

### Output Structure

```
outputs/
├── training/
│   ├── best_model.pt          # Best model checkpoint
│   ├── checkpoint.pt          # Latest checkpoint
│   ├── metrics.json           # Training metrics
│   └── confusion_matrix.png   # Confusion matrix
│
├── validation/
│   └── predictions/           # Validation predictions
│
└── testing/
    ├── test_metrics.json      # Test set metrics
    └── predictions/           # Test predictions
```

### Checkpoint Format

Checkpoints contain:
- Model state dictionary
- Optimizer state
- Scheduler state
- Current epoch
- Best validation accuracy
- Training configuration

---

## 📝 Configuration Parameters

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `batch_size` | Batch size for training | 4 |
| `max_epochs` | Maximum training epochs | 100 |
| `optim_lr` | Learning rate | 1e-4 |
| `reg_weight` | Weight decay | 1e-5 |
| `roi_x`, `roi_y`, `roi_z` | Input volume size | 96x96x96 |
| `workers` | Number of data loading workers | 4 |
| `debug` | Enable debug mode (small subset) | False |

See `config/default.py` for all available parameters.

---

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce batch size: `--batch_size 2`
- Enable mixed precision: `--amp`
- Reduce input size: `--roi_x 64 --roi_y 64 --roi_z 64`

**Class Imbalance**
- Use balanced sampling: `--split_method balanced`
- Try Focal Loss: `--loss_name FocalLoss`
- Enable class-weighted loss (automatic)

**Slow Training**
- Increase num_workers: `--workers 8`
- Enable pin_memory (automatic on GPU)
- Use distributed training for multi-GPU

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{martone2025organoid,
  author = {Raffaele Martone},
  title = {Organoid Image Classification Using Deep Learning},
  school = {Politecnico di Milano},
  year = {2025},
  institution = {Inria Sophia Antipolis}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- [MONAI](https://monai.io/) for medical imaging tools
- [MedicalNet](https://github.com/Tencent/MedicalNet) for pretrained 3D models
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer) for the base architecture
- Inria Sophia Antipolis for computational resources
- Grid5000 platform for distributed computing infrastructure

---

## 📧 Contact

**Raffaele Martone**
- GitHub: [@Martons00](https://github.com/Martons00)
- Email: raffaele.martone@mail.polimi.it

---

## 🔗 Related Projects

- [MONAI](https://github.com/Project-MONAI/MONAI)
- [SwinUNETR](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR)
- [MedicalNet](https://github.com/Tencent/MedicalNet)
- [ML-Decoder](https://github.com/Alibaba-MIIL/ML-Decoder)

---

**Last Updated:** November 2025