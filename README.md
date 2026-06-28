# Wisteria: A Unified Multi-Scale Feature Learning Framework for DNA Language Model

<p align="center">
  <img src="assets/wisteria_logo.png" alt="Wisteria" width="200"/>
</p>

This repository contains the official implementation of **Wisteria**, a genomic language model that integrates multi-scale feature learning within a unified framework for DNA sequence modeling.

> **Note:** This codebase is built upon the [Caduceus](https://github.com/kuleshov-group/caduceus) repository. The model architecture extends Caduceus with gated convolutions, gated MLPs, and Fourier-based attention mechanisms.

## Model Overview

Wisteria introduces three key architectural innovations to the Mamba-based DNA language model:

1. **Gated Convolution-BiMamba (GCMB) Modules**: Integrate gated dilated convolutions with bidirectional Mamba blocks to capture local motifs across multiple receptive field scales.

2. **Gated MLP Modules**: Refine global features through nonlinear feature fusion with selective gating, preserving critical regulatory relationships.

3. **Fourier-based Attention (FoPE)**: Aggregates multi-scale representations in the frequency domain at the final layer, providing position-aware sequence embeddings.

### Architecture

```
Input DNA Sequence
    │
    ▼
Embedding Layer
    │
    ▼
Gated Convolution-BiMamba (GCMB) Modules  ← Multi-scale local feature extraction
    │
    ▼
Gated MLP Modules  ← Nonlinear feature refinement
    │
    ▼
Fourier-based Attention (FoPE)  ← Frequency-domain aggregation
    │
    ▼
Output Embeddings
```

## Key Files

| File/Directory | Description |
|---------------|-------------|
| `caduceus/` | Core model implementation (Wisteria architecture) |
| `caduceus/block.py` | `GatedDilatedConvWithMLP` and `Block` with MSC support |
| `caduceus/modeling_caduceus.py` | `Caduceus`, `CaduceusForMaskedLM`, `CaduceusForSequenceClassification` |
| `caduceus/configuration_caduceus.py` | Model configuration with module/attention/MSC/Fourier params |
| `caduceus/mha.py` | Multi-head attention with Fourier Position Embedding and MoH |
| `caduceus/fourier_position_embedding.py` | Fourier Position Embedding (FoPE) implementation |
| `caduceus/modeling_rcps.py` | Reverse-complement equivariant modules |
| `train.py` | Main training entry point for pre-training and fine-tuning |
| `benchmark.py` | Throughput and memory benchmarking |
| `vep_embeddings.py` | Variant effect prediction embedding extraction |
| `vep_svm.ipynb` | SVM fitting for VEP evaluation |
| `src/` | Training infrastructure (dataloaders, tasks, models) |
| `configs/` | Hydra configuration files |

## Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- PyTorch 2.0+
- flash-attn (optional, for efficient attention)
- mamba-ssm
- causal-conv1d

### Setup

```bash
# Create conda environment
conda env create -f caduceus_env.yml
conda activate caduceus_env

# Or install manually
pip install torch torchvision torchaudio
pip install mamba-ssm causal-conv1d
pip install flash-attn --no-build-isolation
pip install pytorch-lightning hydra-core wandb transformers datasets
```

## Usage

### Model Configuration

The Wisteria model is configured via Hydra configs. The key configuration file is `configs/model/caduceus.yaml`.

Key configuration parameters:

```yaml
d_model: 128              # Model dimension
n_modules: 1              # Number of modules
layers_per_module: 8      # Layers per module
conv_layers_per_module: 5 # Layers with gated convolution per module
attn_layer_in_module: 4   # Position of attention layer in module
use_fourier_pos_emb: true # Enable Fourier Position Embedding
```

### Pre-training

```bash
python -m train \
  experiment=hg38/hg38 \
  dataset.max_length=1024 \
  dataset.batch_size=1024 \
  model=caduceus \
  model.config.d_model=128 \
  model.config.n_modules=1 \
  model.config.layers_per_module=8 \
  model.config.conv_layers_per_module=5 \
  optimizer.lr="8e-3" \
  trainer.max_steps=10000
```

### Downstream Fine-tuning

**Genomic Benchmarks:**
```bash
python -m train \
  experiment=hg38/genomic_benchmark \
  dataset.dataset_name="dummy_mouse_enhancers_ensembl" \
  model=caduceus \
  train.pretrained_model_path="<path to checkpoint>"
```

**Nucleotide Transformer Benchmark:**
```bash
python -m train \
  experiment=hg38/nucleotide_transformer \
  dataset.dataset_name="${task}" \
  model=caduceus \
  train.pretrained_model_path="<path to checkpoint>"
```

### Variant Effect Prediction (VEP)

```bash
# Step 1: Extract embeddings
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  vep_embeddings.py \
  --num_workers=2 \
  --seq_len=131072 \
  --model_name_or_path="<path to model>"

# Step 2: Fit SVM (see vep_svm.ipynb)
```

### Benchmarking

```bash
python benchmark.py
```

## Configuration Details

### GCMB Module (Gated Convolution-BiMamba)

Each GCMB module consists of:
- **Gated Dilated Convolutions**: Two depthwise convolutions with different dilation rates extract local features; one path is activated with GeLU, the other with sigmoid for gating.
- **BiMamba Block**: Bidirectional Mamba captures long-range dependencies in both directions.
- **Gated MLP**: Nonlinear feature fusion with SiLU-activated gating.

Configure via:
- `conv_layers_per_module`: Number of layers using dilated convolutions
- `dilation_base`: Base for computing dilation rates
- `MSC_layer_idx`: Which layers use multi-scale convolutions

### Fourier Position Embedding (FoPE)

Key parameters:
- `use_fourier_pos_emb`: Enable FoPE (replaces RoPE)
- `fourier_max_seq_len`: Maximum sequence length
- `fourier_learnable`: Make Fourier coefficients learnable
- `fourier_ignore_zero`: Zero out under-trained frequencies
- `fourier_separate_head`: Per-head Fourier coefficients

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{wisteria2025,
  title={Wisteria: A Unified Multi-Scale Feature Learning Framework for DNA Language Model},
  author={},
  journal={Pattern Recognition},
  year={2025}
}
```

## Acknowledgements

This repository is built upon the [Caduceus](https://github.com/kuleshov-group/caduceus) codebase, which was adapted from [HyenaDNA](https://github.com/HazyResearch/hyena-dna) and leverages the [Mamba](https://github.com/state-spaces/mamba) architecture.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
