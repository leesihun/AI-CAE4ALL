# SimulGenVAE v2.0.0

> **This repo is now vendored inside AI-CAE4ALL and runs through the launcher**
> (`model simulgenvae`, entrypoint `SimulGenVAE_main.py --config <file>`). The
> standalone CLI/files described below (`SimulGen-VAE.py`, `launch_ddp.py`,
> `preset.txt`, `input_data/condition.txt`, `.pickle` datasets) have been
> **retired** in favor of `--config` + the HDF5-native data loader. **See
> [CLAUDE.md](CLAUDE.md) for the current architecture, layout, config keys, and
> validation commands** — it is authoritative; the sections below describing
> installation/quick-start/configuration are historical and kept for background
> on the model architecture only.

**High-Performance Physics-Aware Variational Autoencoder for Simulation Data Generation**

SimulGenVAE is a state-of-the-art deep learning system designed for fast generation and inference of transient and static simulation data. It features hierarchical latent spaces, multiple conditioning architectures, and distributed training capabilities optimized for physics simulation workflows.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-Compatible-green.svg)](https://developer.nvidia.com/cuda-zone)

## 🚀 Key Features

### Advanced Architecture
- **Hierarchical Variational Autoencoder** with 32D main + 8D hierarchical latent spaces
- **Three Conditioning Architectures**: MLP (parametric), CNN (images), Vision Transformer (images)
- **Physics-Aware Neural Networks (PANN)** integration for domain constraints
- **Multi-scale Feature Processing** with progressive encoder-decoder networks

### High-Performance Training
- **Multi-GPU Distributed Training** with automatic scaling (DDP)
- **Mixed Precision Training** with gradient checkpointing for memory efficiency  
- **Advanced Loss Functions**: MSE, MAE, SmoothL1, Huber with KL warmup scheduling
- **Comprehensive Data Augmentation** pipelines for robust training

### Flexible Data Support
- **Simulation Arrays**: 3D physics data [parameters × timesteps × nodes]
- **Images**: PNG/JPG for CNN/Vision Transformer conditioning
- **Parametric Data**: CSV files for MLP-based conditioning
- **PCA Processing**: Efficient dimensionality reduction for large datasets

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM (16GB+ recommended)
- **GPU**: CUDA-compatible GPU with 6GB+ VRAM
- **Storage**: 10GB free space

### Recommended Configuration
- **Python**: 3.9-3.11
- **GPU**: RTX 3080/4080 or A100 with 12GB+ VRAM
- **Multi-GPU**: 2-8 GPUs for distributed training
- **Memory**: 32GB+ RAM for large datasets

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/SimulGenVAE.git
cd SimulGenVAE
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python SimulGen-VAE.py --help
```

## ⚡ Quick Start

### Single GPU Training
```bash
# Train full VAE with small architecture
python SimulGen-VAE.py --preset=1 --plot=2 --lc_only=0 --size=small

# Train only latent conditioner (requires pre-trained VAE)
python SimulGen-VAE.py --preset=1 --lc_only=1 --size=small
```

### Multi-GPU Distributed Training
```bash
# Use torchrun directly
torchrun --nproc_per_node=4 SimulGen-VAE.py --use_ddp --preset=1

# Use convenience launcher
python launch_ddp.py --gpus=2 --preset=1 --plot=2 --size=small
```

## 📁 Project Structure

```
SimulGenVAE/
├── SimulGen-VAE.py              # Main training script
├── launch_ddp.py                # Distributed training launcher
├── preset.txt                   # Model architecture presets
├── requirements.txt             # Python dependencies
│
├── input_data/
│   └── condition.txt            # Primary configuration file
│
├── modules/                     # Core implementation modules
│   ├── VAE_network.py          # Hierarchical VAE architecture
│   ├── encoder.py              # Progressive encoder networks
│   ├── decoder.py              # Progressive decoder networks
│   ├── train.py                # Main training loops
│   ├── latent_conditioner*.py  # Conditioning model variants
│   ├── data_preprocess.py      # Data processing pipeline
│   ├── augmentation.py         # Data augmentation systems
│   ├── losses.py               # Loss function implementations
│   ├── utils.py                # Core utilities and data loading
│   └── plotter.py              # Visualization and monitoring
│
├── checkpoints/                 # Training checkpoints
├── model_save/                  # Final trained models
├── output/                      # Training logs and results
└── images/                      # Image data (when using image conditioning)
```

## 🔧 Configuration

### Quick Configuration (preset.txt)
The preset system provides pre-configured architectures:

```
data_No, init_beta_divisor, num_filter_enc, latent_conditioner_filter
1                          # Dataset/preset number
0                          # Beta divisor (0 = no KL warmup)
1024 512 256 128          # VAE encoder filter progression
8 16 32 64 128 256 128    # Latent conditioner filter progression
```

### Detailed Configuration (input_data/condition.txt)

#### Dataset Dimensions
```ini
Dim1        484      # Number of parameters in simulation
Dim2        200      # Number of timesteps  
Dim3        95008    # Number of spatial nodes
num_var     1        # Number of variables
```

#### Training Parameters
```ini
Training_epochs     10002    # Total training epochs
Batch_size         16       # Training batch size
LearningR          0.001    # Learning rate
Latent_dim         8        # Hierarchical latent dimension
Latent_dim_end     32       # Main latent dimension
Loss_type          1        # 1:MSE, 2:MAE, 3:SmoothL1, 4:Huber
alpha              1000000  # KL loss scaling factor
```

#### Latent Conditioner Settings
```ini
input_type                 image    # 'image' or 'csvs'
param_dir                 /images   # Directory containing conditioning data
param_data_type           .png      # File extension for image data
n_epoch                   10000     # Conditioner training epochs
latent_conditioner_lr     0.001     # Conditioner learning rate
latent_conditioner_batch  16        # Conditioner batch size
use_spatial_attention     1         # Enable spatial attention (1=on)
```

## 🎯 Training Modes

### 1. Full VAE Training (`--lc_only=0`)
Trains the complete pipeline: encoder, decoder, and latent conditioner.
- **Use case**: New datasets, architecture changes
- **Time**: Longer training duration
- **Memory**: Higher memory requirements

### 2. Latent Conditioner Only (`--lc_only=1`) 
Trains only the conditioning network using a pre-trained VAE.
- **Use case**: Parameter space exploration, conditioning optimization
- **Time**: Faster training
- **Memory**: Lower memory requirements

### 3. End-to-End Training
Joint optimization of VAE and conditioning networks.
- **Configuration**: Set `use_e2e_training=1` in condition.txt
- **Use case**: Fine-tuning for specific downstream tasks
- **Benefits**: Optimal end-to-end performance

## 💾 Data Preparation

### Simulation Data Format
```python
# Expected format: 3D NumPy array
data_shape = (n_parameters, n_timesteps, n_nodes)
# Example: (484, 200, 95008)

# Save as pickle file
import pickle
import numpy as np

simulation_data = np.random.random((484, 200, 95008))
with open('dataset1.pickle', 'wb') as f:
    pickle.dump(simulation_data, f)
```

### Image Conditioning Data
```
images/
├── param_001.png    # First parameter set
├── param_002.png    # Second parameter set  
├── param_003.png    # Third parameter set
└── ...
```
- **Format**: PNG/JPG images
- **Resolution**: Automatically resized to 256×256
- **Naming**: Sequential or parameter-based naming

### CSV Parametric Data
```csv
param1,param2,param3,param4,...
0.1,0.5,0.8,0.3,...
0.2,0.4,0.9,0.1,...
0.3,0.6,0.7,0.5,...
```
- **Format**: Standard CSV with header
- **Processing**: Automatic PCA reduction for efficiency
- **Features**: Numerical parameters only

## 📊 Monitoring and Evaluation

### TensorBoard Integration
```bash
# Start TensorBoard
tensorboard --logdir=output/

# Access at http://localhost:6006
```

**Tracked Metrics:**
- Training/validation losses (reconstruction, KL divergence)
- Learning rate schedules
- GPU memory utilization
- Training speed (samples/second)
- Reconstruction quality metrics

### Output Directory Structure
```
output/
├── tensorboard_logs/     # TensorBoard log files
├── training_plots/       # Generated visualization plots  
├── checkpoints/          # Model checkpoints with timestamps
├── validation_results/   # Validation outputs and metrics
└── generated_samples/    # Sample reconstructions and generations
```

## 🚄 Performance Optimization

### Memory Management
```bash
# For large datasets - enable lazy loading
python SimulGen-VAE.py --load_all=0 --preset=1

# Reduce batch size for memory constraints
# Edit Batch_size in input_data/condition.txt
```

### Multi-GPU Scaling
```bash
# Scale across available GPUs
torchrun --nproc_per_node=$(nvidia-smi --list-gpus | wc -l) SimulGen-VAE.py --use_ddp

# Custom GPU count
python launch_ddp.py --gpus=4 --preset=1 --size=large
```

### Architecture Sizing
- **`--size=small`**: Memory-efficient, suitable for development and limited hardware
- **`--size=large`**: Full performance architecture for production training

## 🔬 Advanced Usage Examples

### Physics Simulation Workflow
```bash
# 1. Train VAE on simulation data
python SimulGen-VAE.py --preset=1 --lc_only=0 --size=small --plot=1

# 2. Train image conditioner using pre-trained VAE
python SimulGen-VAE.py --preset=1 --lc_only=1 --size=small

# 3. End-to-end fine-tuning (set use_e2e_training=1 in condition.txt)
python SimulGen-VAE.py --preset=1 --lc_only=0 --size=small
```

### Large-Scale Distributed Training
```bash
# Multi-node, multi-GPU training
torchrun \
  --nproc_per_node=8 \
  --nnodes=4 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=29500 \
  SimulGen-VAE.py --use_ddp --preset=2 --size=large
```

### Hyperparameter Sweeps
```bash
# Different architectures
for size in small large; do
  for preset in 1 2 3; do
    python SimulGen-VAE.py --preset=$preset --size=$size --lc_only=1
  done
done
```

## 🛠️ Command Line Reference

| Parameter | Description | Options | Default |
|-----------|-------------|---------|---------|
| `--preset` | Configuration preset | 1-5 | 1 |
| `--plot` | Visualization mode | 0=interactive, 1=save, 2=off | 2 |
| `--lc_only` | Training scope | 0=full VAE, 1=conditioner only | 0 |
| `--size` | Model architecture | small, large | small |
| `--load_all` | Data loading strategy | 0=lazy, 1=preload | 1 |
| `--use_ddp` | Distributed training | flag | False |

### DDP Launcher Options
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--gpus` | Number of GPUs | 2 |
| `--master_port` | DDP communication port | 29500 |
| `--train_latent_conditioner_only` | LC-only mode | 0 |

## 🔍 Troubleshooting

### Common Issues

#### Out of Memory Errors
```bash
# Solution 1: Reduce batch size
# Edit Batch_size in input_data/condition.txt

# Solution 2: Use small architecture  
python SimulGen-VAE.py --size=small

# Solution 3: Enable lazy loading
python SimulGen-VAE.py --load_all=0
```

#### Training Instability
```bash
# Check learning rates in condition.txt
LearningR                    0.0001  # Reduce if unstable
latent_conditioner_lr        0.0001  # Reduce if unstable

# Adjust KL warmup (beta divisor in preset.txt)
# Higher values = slower KL warmup = more stable
```

#### Distributed Training Issues
```bash
# Ensure consistent PyTorch versions across nodes
pip install torch==1.13.1 torchvision==0.14.1

# Check CUDA compatibility
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# Use different port if conflicts occur
python launch_ddp.py --master_port=29501
```

### Data Issues
- **File not found**: Verify dataset files exist and paths are correct
- **Shape mismatch**: Check Dim1, Dim2, Dim3 match actual data dimensions
- **Memory issues with large datasets**: Use lazy loading (`--load_all=0`)

## 📖 Citation

If you use SimulGenVAE in your research, please cite:

```bibtex
@software{simulgen_vae_2024,
  title={SimulGenVAE: High-Performance Physics-Aware Variational Autoencoder},
  author={Lee, SiHun},
  version={2.0.0},
  year={2024},
  url={https://github.com/yourusername/SimulGenVAE}
}
```

## 👨‍💻 Author & Contact

**SiHun Lee, Ph.D.**  
📧 kevin1007kr@gmail.com  
🔬 Specializing in Physics-Aware Neural Networks and Simulation Data Processing

## 📄 License

This project is available for research and educational purposes. Commercial use requires explicit permission from the author.

---

**Version**: 2.0.0 (Refactored & Documented)  
**Last Updated**: September 2025