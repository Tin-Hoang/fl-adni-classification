# ADNI Model Evaluation Script Usage

This document describes how to use the `scripts/test.py` script to evaluate trained ADNI classification models.

## Overview

The `test.py` script provides comprehensive evaluation of trained models on test datasets. It supports:

- **Model Loading**: Load any trained checkpoint (training checkpoints, best models, or state dicts)
- **Dataset Support**: 2-class (CN/AD) or 3-class (CN/MCI/AD) classification with MCI subtype filtering
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, AUC, top-k accuracy, confidence analysis
- **Visualization**: Confusion matrices, ROC curves, confidence histograms, prediction samples
- **Timing Analysis**: Inference time measurement per batch and per sample
- **Results Export**: JSON results, classification reports, and visualizations

## Basic Usage

```bash
# Minimal command (recommended - auto-detects model and generates output directory)
python scripts/test.py \
    --checkpoint path/to/model_checkpoint.pth \
    --test_csv path/to/test_data.csv \
    --img_dir path/to/images

# With custom settings
python scripts/test.py \
    --checkpoint path/to/model_checkpoint.pth \
    --test_csv path/to/test_data.csv \
    --img_dir path/to/images \
    --model_name rosanna_cnn \
    --classification_mode CN_AD \
    --resize_size 96 96 73 \
    --output_dir ./custom_results
```

## Command Line Arguments

### Required Arguments

- `--checkpoint`: Path to model checkpoint file
- `--test_csv`: Path to test dataset CSV file
- `--img_dir`: Directory containing MRI images

### Model & Data Configuration

- `--model_name`: Model architecture (default: auto-detect from checkpoint path)
- `--classification_mode`: "CN_MCI_AD" or "CN_AD" (default: CN_MCI_AD)
- `--mci_subtype_filter`: MCI subtypes to include for CN_AD mode (e.g., EMCI LMCI)
- `--resize_size`: Image dimensions as 3 integers [D H W] (default: 128 128 128)

### Optional Arguments

- `--output_dir`: Directory to save results (default: auto-generated based on checkpoint)
- `--batch_size`: Batch size for evaluation (default: 8)
- `--num_workers`: Number of data loading workers (default: 4)
- `--device`: Device to use - "cuda", "cpu", or specific GPU like "cuda:1" (default: auto-detect)
- `--visualize`: Generate prediction visualization plots (default: False)
- `--num_samples_viz`: Number of samples to visualize (default: 8)

## Automatic Output Directory Generation

When `--output_dir` is not specified, the script automatically generates an organized output directory based on:

1. **Checkpoint Parent Directory**: The training run folder name
2. **Checkpoint Type**: Extracted from filename (best, latest, epoch_X, etc.)
3. **Timestamp**: When the evaluation was run

### Format
```
evaluation_results/{parent_dir_name}_{checkpoint_type}_{timestamp}
```

### Examples
```bash
# Checkpoint: outputs_centralized/rosanna-3T_P1-1220images-2classes-scratch-seed101_20250717_034328/rosanna_cnn_checkpoint_best.pth
# Generated: evaluation_results/rosanna-3T_P1-1220images-2classes-scratch-seed101_20250717_034328_best_20250118_143025

# Checkpoint: experiments/resnet3d_run_42/model_checkpoint_epoch_50.pth
# Generated: evaluation_results/resnet3d_run_42_epoch50_20250118_143025

# Checkpoint: saved_models/my_model_latest.pth
# Generated: evaluation_results/saved_models_latest_20250118_143025
```

This automatic naming helps organize evaluation results and makes it easy to:
- Track which model was evaluated
- Identify the checkpoint type used
- Know when the evaluation was performed
- Avoid overwriting previous evaluation results

## Configuration

The evaluation script now uses command line arguments for all configuration, eliminating the need for separate config files. All settings are specified directly via command line arguments with smart defaults and auto-detection.

### Key Configuration Options

**Data Settings:**
- `--img_dir`: Directory containing MRI images
- `--classification_mode`: "CN_MCI_AD" (3-class) or "CN_AD" (2-class)
- `--mci_subtype_filter`: Filter MCI subtypes for CN_AD mode (e.g., EMCI LMCI)
- `--resize_size`: Image dimensions as three integers (default: 128 128 128)

**Model Settings:**
- `--model_name`: Architecture name (auto-detected or specify manually)
- Auto-detected from checkpoint path: resnet3d, rosanna_cnn, securefed_cnn, etc.
- Model-specific parameters use sensible defaults

## Supported Models

The script supports all models available in the project:

- **ResNet3D**: `resnet3d` with configurable depth
- **DenseNet3D**: `densenet3d` with growth rate and block config
- **Simple3DCNN**: `simple3dcnn`
- **SecureFedCNN**: `securefed_cnn`
- **RosannaCNN**: `rosanna_cnn` or `pretrained_cnn`

## Checkpoint Formats

The script can load various checkpoint formats:

1. **Training Checkpoints**: Complete checkpoints with `model_state_dict`
2. **State Dict Files**: Direct model state dictionaries
3. **Best Model Files**: Saved with `*_best.pth` naming

## Dataset Formats

### CSV File Requirements

The test CSV should contain:

**Original Format:**
- `Image Data ID`: Image identifier with 'I' prefix
- `Group`: Diagnosis (AD, MCI, CN)
- `Subject`: Subject identifier
- Optional `DX_bl`: For MCI subtype filtering

**Alternative Format:**
- `image_id`: Image identifier without 'I' prefix
- `DX`: Diagnosis (Dementia, MCI, CN)
- Optional `DX_bl`: For MCI subtype filtering

### Classification Modes

1. **3-Class (CN_MCI_AD)**:
   - CN = 0, MCI = 1, AD = 2
   - Full three-class classification

2. **2-Class (CN_AD)**:
   - CN = 0, AD = 1 (MCI mapped to AD)
   - Optional MCI subtype filtering

### MCI Subtype Filtering

For binary classification, you can filter MCI samples:

```yaml
data:
  classification_mode: "CN_AD"
  mci_subtype_filter: ["EMCI", "LMCI"]  # Include only Early and Late MCI
```

Valid subtypes: `CN`, `SMC`, `EMCI`, `LMCI`, `AD`

## Example Usage Scenarios

### 1. Quick Evaluation (Minimal Command - Recommended)

```bash
python scripts/test.py \
    --checkpoint outputs/resnet3d_run_001/resnet3d_best.pth \
    --test_csv data/test_set.csv \
    --img_dir data/images
# Everything auto-detected: model name, output directory, etc.
# Output: evaluation_results/resnet3d_run_001_best_20250118_143025/
```

### 2. Binary Classification with Visualization

```bash
python scripts/test.py \
    --checkpoint checkpoints/binary_model/rosanna_cnn_best.pth \
    --test_csv data/test_binary.csv \
    --img_dir data/images \
    --classification_mode CN_AD \
    --mci_subtype_filter LMCI \
    --visualize \
    --batch_size 4
# Output: evaluation_results/binary_model_best_20250118_143025/
```

### 3. Custom Model Configuration

```bash
python scripts/test.py \
    --checkpoint models/my_model.pth \
    --test_csv data/holdout_test.csv \
    --img_dir data/images \
    --model_name securefed_cnn \
    --resize_size 182 218 182 \
    --output_dir ./custom_results/securefed_final_eval \
    --device cuda:0 \
    --batch_size 16 \
    --num_workers 8
```

### 4. Advanced Configuration Example

```bash
python scripts/test.py \
    --checkpoint path/to/checkpoint.pth \
    --test_csv data/test.csv \
    --img_dir data/images \
    --model_name resnet3d \
    --classification_mode CN_MCI_AD \
    --resize_size 224 224 224 \
    --batch_size 2 \
    --device cuda:1 \
    --num_workers 8
# Full control over all parameters
```

## Output Files

The script generates several output files:

### Core Results
- `evaluation_results.json`: Complete evaluation metrics and predictions
- `classification_report.txt`: Detailed classification report
- `predictions_detailed.csv`: Per-image prediction results with paths and probabilities
- `confusion_matrix.png`: Raw confusion matrix
- `confusion_matrix_normalized.png`: Normalized confusion matrix

### Advanced Visualizations
- `roc_curves.png`: ROC curves (per-class for multi-class)
- `confidence_histogram.png`: Confidence score distributions
- `prediction_samples.png`: Sample predictions (if `--visualize` used)

### JSON Results Structure

```json
{
  "dataset_statistics": {
    "total_samples": 100,
    "num_classes": 3,
    "class_distribution": {"CN": 40, "MCI": 35, "AD": 25},
    "class_imbalance_ratio": 1.6
  },
  "evaluation_metrics": {
    "accuracy": 0.85,
    "precision_per_class": {"CN": 0.88, "MCI": 0.81, "AD": 0.87},
    "auc_scores": {"CN_vs_rest": 0.92, "macro_average": 0.89},
    "confidence_stats": {"mean_confidence": 0.82},
    "total_inference_time": 15.3
  },
  "predictions": {
    "true_labels": [0, 1, 2, ...],
    "predicted_labels": [0, 1, 2, ...],
    "predicted_probabilities": [[0.8, 0.1, 0.1], ...]
  }
}
```

### Detailed Predictions CSV Structure

The `predictions_detailed.csv` file contains per-image prediction results with the following columns:

| Column | Description |
|--------|-------------|
| `image_path` | Full path to the image file |
| `image_filename` | Just the filename (for easier reading) |
| `true_label_numeric` | Ground truth label (0, 1, 2) |
| `true_label_name` | Ground truth class name (CN, MCI, AD) |
| `predicted_label_numeric` | Predicted label (0, 1, 2) |
| `predicted_label_name` | Predicted class name (CN, MCI, AD) |
| `prediction_correct` | Boolean indicating if prediction was correct |
| `max_confidence` | Highest probability among all classes |
| `confidence_margin` | Difference between top 2 predictions |
| `prob_CN` | Probability for CN class |
| `prob_MCI` | Probability for MCI class (3-class mode only) |
| `prob_AD` | Probability for AD class |

This CSV file is especially useful for:
- **Error Analysis**: Filter by `prediction_correct=False` to study misclassified cases
- **Confidence Analysis**: Sort by `max_confidence` or `confidence_margin` to find uncertain predictions
- **Class-specific Analysis**: Filter by true or predicted labels to study specific classes
- **Manual Review**: Use image paths to examine specific cases visually

## Reported Metrics

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and macro/weighted averages
- **AUC**: Area under ROC curve (binary for 2-class, one-vs-rest for multi-class)
- **Top-k Accuracy**: Top-2 and Top-3 accuracy for multi-class

### Timing Metrics
- **Total Inference Time**: Complete evaluation time
- **Average Time per Batch**: Time per batch of samples
- **Average Time per Sample**: Time per individual sample

### Confidence Analysis
- **Mean/Std Confidence**: Statistics of prediction confidence scores
- **Min/Max Confidence**: Range of confidence scores
- **Confidence Distribution**: Histograms by correctness and class

## Memory Optimization

For large datasets or limited memory:

1. **Reduce Batch Size**: Use `--batch_size 2` or `--batch_size 4`
2. **Reduce Workers**: Use `--num_workers 0` or `--num_workers 2`
3. **CPU Evaluation**: Use `--device cpu` if GPU memory is limited
4. **Smaller Images**: Use `--resize_size 96 96 73` for lower memory usage

## Troubleshooting

### Common Issues

1. **Shape Mismatch Errors**:
   - Check `--resize_size` matches training configuration
   - Verify `--model_name` matches checkpoint

2. **Missing Images**:
   - Verify `--img_dir` path is correct
   - Check that CSV image IDs match actual files

3. **Memory Errors**:
   - Reduce `--batch_size`
   - Use `--num_workers 0`
   - Switch to CPU with `--device cpu`

4. **Auto-Detection Errors**:
   - Specify `--model_name` explicitly if auto-detection fails
   - Check checkpoint path contains recognizable model names

### Debug Tips

1. **Test with Small Dataset**: Start with a small test CSV to verify setup
2. **Check Arguments**: Ensure all paths and parameters are correct
3. **Verify Model**: Test checkpoint loading separately
4. **Monitor Memory**: Use system monitoring tools during evaluation

## Performance Tips

1. **GPU Optimization**:
   - Use appropriate batch size for your GPU
   - Enable pin_memory for faster data transfer

2. **CPU Optimization**:
   - Adjust num_workers based on CPU cores
   - Consider batch size vs. memory trade-offs

3. **Storage Optimization**:
   - Use local storage for images if possible
   - Consider image preprocessing and caching

## Integration with Training Pipeline

The test script is designed to work seamlessly with models trained using `scripts/train.py`:

1. **Auto-Detection**: Model name is automatically detected from checkpoint path
2. **Simple Arguments**: Only specify test data paths and image directory
3. **Smart Defaults**: Evaluation uses sensible defaults optimized for testing
4. **No Config Files**: No need to maintain separate config files for evaluation

This ensures consistent evaluation of your trained models with minimal setup and maximum convenience.
