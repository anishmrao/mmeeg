# MMEEG: Modular EEG Decoding Framework

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange)](https://pytorch.org/)

MMEEG is an open-source, modular framework built on top of [MMEngine](https://github.com/open-mmlab/mmengine) for fast and reproducible EEG decoding experiments, with a focus on Motor Imagery tasks. The framework provides a config-driven approach to train and evaluate deep learning models on EEG data with minimal boilerplate code.

## ✨ Features

- **Modular Design**: Easily extendable architecture for custom datasets and models
- **State-of-the-Art Models**: Includes implementations of EEGConformer and EEGNet
- **Config-Driven**: Reproducible experiments through configuration files
- **Built on MMEngine**: Leverages the powerful MMEngine training framework
- **Easy Experimentation**: Quickly prototype and test new ideas with minimal code changes

## 📦 Installation

1. **Prerequisites**
   - Python 3.8+
   - PyTorch 1.8+
   - CUDA 11.3+ (for GPU acceleration)

2. **Install Dependencies**
   ```bash
   # Install MMEngine (follow their [installation guide](https://github.com/open-mmlab/mmengine#installation))
   pip install -U openmim
   mim install mmengine

   # Install MMEEG dependencies
   pip install -r requirements.txt  # or install manually:
   # pip install numpy scipy einops
   ```

## 🚀 Quick Start

1. **Explore the Tutorial**
   Open and run `MMEEG_Tutorial.ipynb` for a hands-on introduction to the framework.

2. **Train a Model**
   ```python
   from mmengine import Config
   from mmengine.runner import Runner
   from models import MMEEGConformer
   
   # Load configuration
   cfg = Config.fromfile('configs/eeg_conformer_config.py')
   
   # Initialize model
   model = MMEEGConformer()
   
   # Create and run training
   runner = Runner(model=model, work_dir='./work_dir', **cfg)
   runner.train()
   ```

## 🧩 Project Structure

```
mmeeg/
├── configs/               # Configuration files for experiments
│   ├── eeg_conformer_config.py
│   └── eegnet_config.py
├── datasets/             # Dataset implementations and registry
│   ├── eeg_dataset.py    # Example EEG dataset implementation
│   └── registry.py       # Dataset registry for modular design
├── models/               # Model architectures and registry
│   ├── eeg_conformer.py  # Transformer-based EEG model
│   ├── eegnet.py         # Compact CNN for EEG
│   └── registry.py       # Model registry for modular design
├── utils/                # Utility functions
├── MMEEG_Tutorial.ipynb  # Getting started tutorial
└── requirements.txt      # Project dependencies
```

## 🏗️ Modular Architecture

MMEEG is built with extensibility in mind. The framework uses a registry pattern that allows easy addition of new models, datasets, and components.

### 🔄 Registry System

The framework includes registries for different components:
- **Models**: Register new model architectures
- **Datasets**: Add support for new datasets
- **Pipelines**: Custom data processing pipelines

### ➕ Adding New Components

#### 1. Adding a New Model

1. Create your model class in `models/your_model.py`
2. Decorate it with `@MODELS.register_module()`
3. Import the model in `models/__init__.py`

```python
# models/your_model.py
from .registry import MODELS

@MODELS.register_module()
class YourModel(nn.Module):
    def __init__(self, param1=default1, param2=default2):
        super().__init__()
        # Your model implementation
```

#### 2. Adding a New Dataset

1. Create your dataset class in `datasets/your_dataset.py`
2. Decorate it with `@DATASETS.register_module()`
3. Import the dataset in `datasets/__init__.py`

```python
# datasets/your_dataset.py
from .registry import DATASETS

@DATASETS.register_module()
class YourDataset(Dataset):
    def __init__(self, data_root, split='train', **kwargs):
        # Your dataset implementation
```

### ⚙️ Configuration System

MMEEG uses a flexible configuration system based on Python files. Here's how to configure different components:

#### Example Configuration (`configs/example_config.py`)

```python
# Model configuration
model = dict(
    type='YourModel',  # Must match registered model name
    param1=value1,
    param2=value2
)

# Training dataset
dataset = dict(
    type='YourDataset',
    data_root='path/to/data',
    split='train',
    # Additional dataset parameters
)

# Validation dataset
val_dataset = dict(
    type='YourDataset',
    data_root='path/to/data',
    split='val',
    # Additional dataset parameters
)

# Training configuration
train_cfg = dict(
    max_epochs=100,
    optimizer=dict(type='Adam', lr=0.001),
    # Additional training parameters
)
```

#### Using the Configuration

```python
from mmengine import Config
from mmengine.runner import Runner

# Load configuration
cfg = Config.fromfile('configs/example_config.py')

# The framework will automatically instantiate the specified components
runner = Runner(**cfg)
runner.train()
```

## 🧠 Available Models

- **EEGConformer**: A transformer-based architecture for EEG signal processing
- **EEGNet**: Compact convolutional neural network for EEG classification

## 🤝 Contributing

We welcome contributions! Please open an issue or submit a pull request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

If you find this project useful for your research, please consider citing:

```bibtex
@misc{mmeeg2023,
  author = {Anish M Rao},
  title = {MMEEG: Modular EEG Decoding Framework},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/anishmrao/mmeeg}}
}
```

## 💡 Support

For questions and support, please open an issue in the repository.