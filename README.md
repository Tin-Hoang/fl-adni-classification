# ADNI Classification

This project implements a deep learning-based classification system for ADNI (Alzheimer's Disease Neuroimaging Initiative) MRI data.

## Project Structure

```
adni_classification/
├── config/              # Configuration management code
├── configs/             # Configuration YAML files
│   ├── default.yaml     # Default configuration
│   └── densenet.yaml    # DenseNet-specific configuration
├── data/                # Data loading and preprocessing
├── models/              # Model implementations
├── scripts/             # Training and utility scripts
├── utils/               # Utility functions
└── requirements.txt     # Project dependencies
```

## Configuration

The project uses YAML configuration files located in the `configs/` directory:

- `configs/default.yaml`: Default configuration for ResNet3D
- `configs/densenet.yaml`: Configuration for DenseNet3D

To use a specific configuration, pass it to the training script:

```bash
python scripts/train.py --config configs/densenet.yaml
```

### Weights & Biases Integration

The project includes integration with [Weights & Biases](https://wandb.ai/) for experiment tracking. To enable wandb tracking, set `use_wandb: true` in your configuration file:

```yaml
wandb:
  use_wandb: true  # Set to true to enable Weights & Biases tracking
  project: "fl-adni-classification"
  entity: "tin-hoang"
  tags: ["experiment-name"]  # Add tags to categorize your runs
  notes: "Description of your experiment"  # Add notes to describe your experiment
```

You can customize the project name, entity, tags, and notes in the configuration file.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Login to Weights & Biases:
```bash
wandb login
```

## Usage

To train a model:
```bash
python scripts/train.py --config config/default.yaml
```

## Data Format

The input data should be:
- 3D MRI images in .nii format
- A CSV file containing metadata with the following columns:
  - Image Data ID
  - Subject
  - Group (AD, MCI, NC)
  - Sex
  - Age
  - Visit
  - Modality
  - Description
  - Type
  - Acq Date
  - Format
  - Downloaded

## Model Architecture

The project uses a ResNet50-based 3D CNN architecture, implemented using the Model Factory pattern for easy extension. 