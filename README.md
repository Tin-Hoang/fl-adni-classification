# ADNI Classification

***Abstract:** This work presents a federated learning framework for automated classification of Alzheimer's Disease using neuroimaging data from the Alzheimer's Disease Neuroimaging Initiative (ADNI). We implement 3D convolutional neural networks to classify MRI scans into three diagnostic categories: Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN) controls. The framework employs a modular architecture supporting multiple 3D CNN models including ResNet3D and DenseNet3D configurations. A standardized preprocessing pipeline utilizing ANTs and FSL performs resampling to 1mm isotropic spacing, registration to ICBM152 templates, and skull stripping. The system integrates YAML-based configuration management and Weights & Biases for experiment tracking, supporting multiple ADNI dataset formats while maintaining reproducible workflows for neuroimaging-based disease classification.*

## Project Structure

```
fl-adni-classification/
├── adni_flwr/           # Federated learning implementation (Flower)
│   ├── client_app.py    # FL client application
│   ├── server_app.py    # FL server application
│   ├── server_fn.py     # Server functions
│   ├── task.py          # FL task definitions
│   └── strategies/      # FL aggregation strategies
├── adni_classification/ # Core classification components
│   ├── models/          # Model implementations
│   ├── datasets/        # Dataset implementations
│   ├── utils/           # Utility functions
│   └── config/          # Configuration management
│       ├── config.py    # Main configuration
│       └── fl_config.py # FL-specific configuration
├── scripts/             # Training and utility scripts
│   ├── train.py         # Main training script
│   ├── preprocess_mri.py # MRI preprocessing pipeline
│   ├── split_data_for_fl.py # Data splitting for FL
│   ├── filter_csv_by_images.py # CSV filtering utilities
│   ├── calculate_adni_statistics.py # Statistics calculation
│   └── [other utility scripts]
├── configs/             # Configuration YAML files
├── data/                # Data directory
├── logs/                # Training logs
├── run_multi_machines.py      # Multi-machine FL execution
├── run_multi_machines_tmux.py # Multi-machine FL with tmux
├── setup_fl_env.sh            # Environment setup script
├── config.py                  # Global configuration
├── pyproject.toml             # Project dependencies (uv)
└── README.md                  # Project documentation
```

## Setup

Using UV for dependency management:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies
uv venv --python 3.11
# Activate the virtual environment
source .venv/bin/activate

# Install required dependencies
uv pip install -e .
```

## Usage

### Basic Training (centralized training)
```bash
python scripts/train.py --config configs/default.yaml
```

### Federated Learning (using Flower)
#### Option 1: FL Local Simulation (single machine)
- Modify `pyproject.toml` to configure the FL simulation:
  - `client-config-files` is the path to the client config files, each client config file is a YAML file that contains the configuration for a single client.
  - `server-config-file` is the path to the server config file.
  - `options.num-supernodes` is the number of supernodes.
  - `options.backend.client-resources.num-cpus` is the number of CPUs per client.
  - `options.backend.client-resources.num-gpus` is the number of GPUs per client.

- Run the following command to start the FL local simulation (single machine):
```bash
flwr run .
```

Or, using the following command to start the FL server
```bash
# Single machine FL
python run_local_simulation.py configs/<experiment_name>/fl_server.yaml
```

#### Option 2: FL Multi-Machine Deployment

The project supports distributed multi-machine deployment via [Flower network communication](https://flower.ai/docs/framework/ref-flower-network-communication.html) with automated SSH tunnel establishment when needed.

##### Prerequisites

1. **SSH Access**: Ensure passwordless SSH access to all client machines or set the `FL_PASSWORD` environment variable
2. **Synchronized Project**: The project directory must be synchronized across all machines
3. **Virtual Environment**: Python virtual environment must be set up on all machines with identical paths
4. **Network Configuration**: Ensure machines can communicate via SSH (port 22) and Flower ports (if no firewall restrictions)

##### Configuration Steps

**Step 1: Configure Multi-Machine FL in `fl_server.yaml`**

Add the `multi_machine` section to your FL configuration:
<details>
<summary>Configuration for Multi-Machine FL</summary>

```yaml
fl:
  # Standard FL parameters
  num_rounds: 100
  fraction_fit: 1.0
  min_fit_clients: 4
  min_evaluate_clients: 4
  min_available_clients: 4
  local_epochs: 1
  strategy: "fedavg"
  evaluate_frequency: 5

  # Multi-machine configuration
  multi_machine:
    # Server machine configuration
    server:
      host: "otter82.eps.surrey.ac.uk"      # Server hostname/IP
      username: "th01167"                    # SSH username
      password: null                         # Use FL_PASSWORD env var
      port: 9092                            # Flower server port
      config_file: "configs/experiment/fl_server.yaml"  # Server config path
      sequential_experiment: true           # Enable sequential experiments
      train_sequential_labels:             # Training data for sequential runs
        - "data/ADNI/LABELS/seed01/client_all_train.csv"
        - "data/ADNI/LABELS/seed10/client_all_train.csv"
        - "data/ADNI/LABELS/seed42/client_all_train.csv"
      val_sequential_labels:               # Validation data for sequential runs
        - "data/ADNI/LABELS/seed01/client_all_val.csv"
        - "data/ADNI/LABELS/seed10/client_all_val.csv"
        - "data/ADNI/LABELS/seed42/client_all_val.csv"

    # Client machines configuration
    clients:
      - host: "otter77.eps.surrey.ac.uk"    # Client 1 hostname/IP
        username: "th01167"                 # SSH username
        password: null                      # Use FL_PASSWORD env var
        partition_id: 0                     # Unique client ID
        project_dir: null                   # Use global project_dir
        config_file: "configs/experiment/client1.yaml"  # Client config path
        sequential_experiment: true         # Enable sequential experiments
        train_sequential_labels:           # Training data for this client
          - "data/ADNI/LABELS/seed01/client_1_train.csv"
          - "data/ADNI/LABELS/seed10/client_1_train.csv"
          - "data/ADNI/LABELS/seed42/client_1_train.csv"
        val_sequential_labels:             # Validation data for this client
          - "data/ADNI/LABELS/seed01/client_1_val.csv"
          - "data/ADNI/LABELS/seed10/client_1_val.csv"
          - "data/ADNI/LABELS/seed42/client_1_val.csv"

      - host: "otter78.eps.surrey.ac.uk"    # Client 2 configuration
        username: "th01167"
        password: null
        partition_id: 1
        project_dir: null
        config_file: "configs/experiment/client2.yaml"
        sequential_experiment: true
        train_sequential_labels:
          - "data/ADNI/LABELS/seed01/client_2_train.csv"
          - "data/ADNI/LABELS/seed10/client_2_train.csv"
          - "data/ADNI/LABELS/seed42/client_2_train.csv"
        val_sequential_labels:
          - "data/ADNI/LABELS/seed01/client_2_val.csv"
          - "data/ADNI/LABELS/seed10/client_2_val.csv"
          - "data/ADNI/LABELS/seed42/client_2_val.csv"

    # Project configuration (shared across all machines)
    project_dir: "/user/HS402/th01167/Surrey/fl-adni-classification"
    venv_path: "/user/HS402/th01167/.venv/master/bin/python"
    venv_activate: "/user/HS402/th01167/.venv/master/bin/activate"

    # SSH configuration
    ssh:
      timeout: 30
      banner_timeout: 30
      auth_timeout: 30
```

</details>


**Step 2: Set Environment Variables**

```bash
# Set password for SSH authentication (if not using passwordless SSH)
export FL_PASSWORD="your_password_here"
```

**Step 3: Prepare Client Configuration Files**

Create individual client configuration files (e.g., `client1.yaml`, `client2.yaml`) with client-specific settings:

```yaml
# client1.yaml
data:
  train_csv_path: "data/ADNI/LABELS/seed01/client_1_train.csv"
  val_csv_path: "data/ADNI/LABELS/seed01/client_1_val.csv"
  img_dir: "data/ADNI/images"

model:
  name: "rosanna_cnn"
  num_classes: 2

training:
  batch_size: 4
  learning_rate: 0.001
  num_workers: 4

fl:
  client_id: 0
  local_epochs: 1
```

**Step 4: Deploy Multi-Machine FL**

```bash
# Multi-machine FL with tmux (recommended)
python run_multi_machines_tmux.py --config configs/experiment/fl_server.yaml

# Alternative: Multi-machine FL without tmux
python run_multi_machines.py --config configs/experiment/fl_server.yaml
```

##### Advanced Features

**Sequential Experiments**: The framework supports running multiple experiments with different random seeds automatically. Set `sequential_experiment: true` and provide lists of training/validation CSV files for different seeds.

**Automatic SSH Tunneling**: If machines cannot communicate via default Flower ports due to firewall restrictions, the system automatically establishes SSH tunnels.

**Dynamic Client Management**: The system handles client failures and reconnections automatically during training.

**Resource Monitoring**: Built-in GPU and memory monitoring across all client machines.

##### Troubleshooting

1. **SSH Connection Issues**: Ensure passwordless SSH or correct `FL_PASSWORD` environment variable
2. **Port Conflicts**: Modify `port` in server configuration if 9092 is occupied
3. **Synchronization Issues**: Ensure all machines have identical project directories and dependencies
4. **GPU Allocation**: Use `CUDA_VISIBLE_DEVICES` to control GPU usage per client

*Note: SSH tunnels will be established automatically between server and client nodes if direct communication via Flower ports is not possible.*

## Configuration

The project uses a sophisticated YAML-based configuration system with structured dataclasses located in `adni_classification/config/`. Configuration files are stored in the `configs/` directory.

### Configuration Structure

The configuration system is organized into five main sections:

#### 1. Data Configuration (`data:`)
- **Dataset paths**: `train_csv_path`, `val_csv_path`, `img_dir`
- **Dataset types**: `normal`, `cache`, `smartcache`, `persistent`
- **Image preprocessing**: `resize_size`, `resize_mode`, `spacing_size`
- **Classification modes**: `CN_MCI_AD` (3-class) or `CN_AD` (2-class)
- **Caching options**: `cache_rate`, `cache_num_workers`, `cache_dir`

#### 2. Model Configuration (`model:`)
- **Model selection**: `resnet3d`, `densenet3d`, `simple_cnn`, `securefed_cnn`, `rosanna_cnn`
- **Architecture params**: `model_depth`, `growth_rate`, `block_config`
- **Pretrained models**: `pretrained_checkpoint`, `freeze_encoder`

#### 3. Training Configuration (`training:`)
- **Optimization**: `batch_size`, `learning_rate`, `weight_decay`, `num_epochs`
- **Advanced features**: `mixed_precision`, `gradient_accumulation_steps`
- **Loss functions**: `cross_entropy`, `focal` (with `focal_alpha`, `focal_gamma`)
- **Class balancing**: `use_class_weights`, `class_weight_type`, `manual_class_weights`
- **Checkpointing**: `save_best`, `save_latest`, `save_regular`, `save_frequency`

#### 4. Federated Learning Configuration (`fl:`)
- **FL strategies**: `fedavg`, `fedprox`, `secagg`
- **Round configuration**: `num_rounds`, `local_epochs`, `evaluate_frequency`
- **Client selection**: `fraction_fit`, `min_fit_clients`, `min_available_clients`
- **Strategy-specific**: `fedprox_mu`, `secagg_noise_multiplier`

#### 5. Weights & Biases Configuration (`wandb:`)
- **Experiment tracking**: `use_wandb`, `project`, `entity`
- **Organization**: `tags`, `notes`, `run_name`


### Advanced Features

#### Dataset Caching
```yaml
data:
  dataset_type: "smartcache"  # Intelligent caching
  cache_rate: 0.5  # Cache 50% of data
  cache_num_workers: 8
  transform_device: "cuda"  # GPU transforms
```

#### Focal Loss and Class Weighting for Class Imbalance
```yaml
training:
  loss_type: "focal"
  focal_alpha: 0.25
  focal_gamma: 2.0
  use_class_weights: true
  class_weight_type: "inverse"
```

#### Mixed Precision Training
```yaml
training:
  mixed_precision: true
  gradient_accumulation_steps: 2
```


## Data Format

The input data should be:
- 3D MRI images in .nii or .nii.gz format
- A CSV label file containing metadata with the following columns:
  - image_id
  - subject_id
  - DX (AD, MCI, CN)
  - sex
  - age


## Model Architecture

The project uses a ResNet50-based 3D CNN architecture, implemented using the Model Factory pattern for easy extension.

# ADNI MRI Preprocessing Tool

Automates preprocessing for ADNI MRI images:
1. Resampling to 1mm isotropic spacing
2. Registration to a standard template
3. Skull stripping

## Requirements

- Python 3.6+
- ANTs (Advanced Normalization Tools)
- FSL (FMRIB Software Library)

## Usage

```bash
python scripts/preprocess_mri.py --input input_folder [--output output_dir] [--template template.nii.gz]
```

### Arguments

- `--input`: Path to input directory containing MRI images (required)
- `--output`: Base output directory (default: same folder name in parent directory)
- `--template`: Template for registration (default: data/ICBM152/mni_icbm152_t1_tal_nlin_sym_09a.nii)

### Output Structure

For each .nii or .nii.gz file in the input directory, the script creates:

- `<dirname>/<basename>_step1_resampling/`: Contains resampled images
- `<dirname>/<basename>_step2_registration/`: Contains registration outputs
- `<dirname>/<basename>_step3_skull_stripping/`: Contains final skull-stripped images

Where `<dirname>` is the parent directory of the input folder and `<basename>` is the name of the input folder.

The script preserves the original subfolder structure in each step's output directory.
Each output file maintains the same filename and directory structure as in the input directory.

### Example

Using this input structure:
```
/path/to/input_dir/
└── subject001/
    └── session01/
        └── scan.nii
```

You can run:
```bash
python scripts/preprocess_mri.py --input /path/to/input_dir
```

The script will create:
```
/path/to/input_dir_step1_resampling/
└── subject001/
    └── session01/
        └── scan.nii

/path/to/input_dir_step2_registration/
└── subject001/
    └── session01/
        └── scan.nii

/path/to/input_dir_step3_skull_stripping/
└── subject001/
    └── session01/
        └── scan.nii.gz
```
