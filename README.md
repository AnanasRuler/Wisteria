# Wisteria: A Unified Multi-Scale Feature Learning Framework for DNA Language Model

Official implementation of **Wisteria**, a genomic language model that integrates multi-scale feature learning within a unified framework. Wisteria augments the Mamba-based architecture with **gated dilated convolutions** to capture local motifs and regulatory patterns, while **gated MLPs** refine global dependencies, and a final **Fourier-based attention** mechanism (FoPE) aggregates multi-scale representations.

## Architecture

1. **Gated Convolution-BiMamba (GCMB) Modules** — gated dilated convolutions for multi-scale local feature extraction, processed by bidirectional Mamba
2. **Gated MLP Modules** — nonlinear feature fusion with selective gating for global dependency refinement  
3. **Fourier-based Attention (FoPE)** — frequency-domain aggregation for position-aware embeddings

## Installation

```bash
conda env create -f wisteria_env.yml
conda activate wisteria_env
```

## Usage

### Pre-training

```bash
python -m train \
  experiment=hg38/hg38 \
  dataset.max_length=1024 \
  dataset.batch_size=1024 \
  model=wisteria \
  model.config.d_model=128 \
  optimizer.lr="8e-3" \
  trainer.max_steps=10000
```

### Downstream Fine-tuning

**Genomic Benchmarks:**
```bash
python -m train experiment=hg38/genomic_benchmark \
  dataset.dataset_name="dummy_mouse_enhancers_ensembl" \
  model=wisteria train.pretrained_model_path="<checkpoint>"
```

**Nucleotide Transformer Benchmark:**
```bash
python -m train experiment=hg38/nucleotide_transformer \
  dataset.dataset_name="${task}" model=wisteria \
  train.pretrained_model_path="<checkpoint>"
```

### Variant Effect Prediction

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  vep_embeddings.py --num_workers=2 --seq_len=131072 \
  --model_name_or_path="<checkpoint>"
# Then: vep_svm.ipynb
```

### Benchmarking

```bash
python benchmark.py
```

## Key Configuration

```yaml
# configs/model/wisteria.yaml
d_model: 128
n_modules: 1              # Number of hierarchical modules
layers_per_module: 8      # Layers per module
conv_layers_per_module: 5 # GCMB layers (gated convolution)
attn_layer_in_module: 4   # Attention layer position (FoPE)
use_fourier_pos_emb: true # Enable Fourier Position Embedding
```

## Project Structure

| Directory/File | Description |
|---|---|
| `wisteria/` | Core model (GCMB, BiMamba, FoPE, gated MLP) |
| `wisteria/block.py` | `GatedDilatedConvWithMLP` + `Block` |
| `wisteria/modeling_wisteria.py` | `Wisteria` model (MLM & classification) |
| `wisteria/mha.py` | Multi-head attention + FoPE + MoH |
| `wisteria/fourier_position_embedding.py` | Fourier Position Embedding |
| `wisteria/modeling_rcps.py` | Reverse-complement equivariant modules |
| `train.py` | Pre-training & fine-tuning |
| `benchmark.py` | Throughput & memory benchmarking |
| `vep_embeddings.py` / `vep_svm.ipynb` | Variant Effect Prediction |
| `src/` | Training infrastructure (dataloaders, tasks) |
| `configs/` | Hydra configurations |

## Citation

```bibtex
@article{wisteria2025,
  title={Wisteria: A Unified Multi-Scale Feature Learning Framework for DNA Language Model},
  journal={Pattern Recognition},
  year={2025}
}
```

## Acknowledgements

Built upon [Caduceus](https://github.com/kuleshov-group/caduceus), adapted from [HyenaDNA](https://github.com/HazyResearch/hyena-dna), using the [Mamba](https://github.com/state-spaces/mamba) architecture.

## License

Apache License 2.0 — [LICENSE](LICENSE).
