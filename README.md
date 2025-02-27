# Sparse Brains are Also Adaptive Brains: Cognitive-Load-Aware Dynamic Activation for LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repo is the implementation of article "Sparse Brains are Also Adaptive Brains"(https://arxiv.org/abs/2502.19078).

CLADA (**C**ognitive-**L**oad-**A**ware **D**ynamic **A**ctivation) is a novel framework for efficient LLM inference that dynamically adjusts neuron activation patterns based on real-time cognitive load metrics. Inspired by human neurolinguistic mechanisms (N400/P600 ERPs), CLADA achieves **~20% inference speedup** with **<2% accuracy drop** across six LLM architectures and nine benchmarks.

## Key Features
- **Dual Sparsity Mechanisms**  
  Combines _global statistical sparsity_ (prefix-driven) and _local semantic adaptability_ (surprisal/entropy-modulated).
- **Training-Free Optimization**  
  No model retraining or architectural changes required.
- **Cross-Architecture Support**  
  Compatible with GPT-style, LLaMA, Mistral, and OPT models.
- **Cognitive Load Metrics**  
  Implements surprisal and entropy-based dynamic thresholding.
- **Low-Memory Mode**  
  Supports 8-bit quantization for memory-constrained environments.

## Installation
```bash
# Create conda environment
conda create -n clada python=3.9
conda activate clada

# Install core dependencies
pip install torch>=2.0 transformers>=4.30 h5py tqdm numpy matplotlib

# For 8-bit quantization (optional):
pip install bitsandbytes
```

## Quick Start

### Environment

```bash
conda create -n clada python=3.9
conda activate clada

pip install -r requirements.txt

pip3 install torch==2.0.0+cu1** torchvision==0.**.0+cu1** torchaudio==2.0.1+cu1** --extra-index-url https://download.pytorch.org/whl/cu1**
```

### Configuration

Modify`ConditionalComputationConfig`in`src/clada/clada.py`or use command-line arguments:
```bash
# Example config setting
config = ConditionalComputationConfig(
    model_path="./llama-2-7b-hf",
    data_path="./data/xsum/train.jsonl",
    output_dir="./results",
    ffn_reduction_ratio=0.3,
    top_k_ratio=0.4,
    max_new_tokens=512
)
```

#### Running Inference

```bash
# Baseline (full activation)
python CLADA.py --baseline_run --model_path ./llama-2-7b-hf

# CLADA dynamic activation
python CLADA.py --model_path ./llama-2-7b-hf --top_k_ratio 0.4

# Custom thresholds
python CLADA.py --surprisal_threshold 0.65 --entropy_threshold 0.7

```

#### Key Arguments

|Parameter|Description|Default|
|---|---|---|
|`--model_path`|Pretrained model directory|`./pretrained_model`|
|`--data_path`|JSONL dataset path|`./data/xsum/train.jsonl`|
|`--top_k_ratio`|Top-k neuron ratio|0.3|
|`--surprisal_threshold`|Surprisal percentile threshold|0.7|
|`--update_window`|Threshold adaptation frequency|10 tokens|
|`--low_memory_mode`|Enable 8-bit quantization|False|

## Experimental Results

CLADA achieves significant speedup while maintaining accuracy:

|Model|Speedup|Accuracy Retention|
|---|---|---|
|LLaMA-2-7B|22.1%|98.9%|
|Mistral-7B|25.3%|99.2%|
|OPT-2.7B|18.6%|97.8%|

# Citation

If you use CLADA in your research, please cite(Will update later):
```bibtex
@misc{yang2025sparsebrainsadaptivebrains,
      title={Sparse Brains are Also Adaptive Brains: Cognitive-Load-Aware Dynamic Activation for LLMs}, 
      author={Yiheng Yang and Yujie Wang and Chi Ma and Lei Yu and Emmanuele Chersoni and Chu-Ren Huang},
      year={2025},
      eprint={2502.19078},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.19078}, 
}
```

## Acknowledgements
- Inspired by human neurolinguistic mechanisms (N400/P600 ERPs) and Griffin(Dong, H, 2024)
- Built with PyTorch and Hugging Face Transformers
- Supported by Meituan compute resources


