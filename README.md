# DLGN: Differentiable Logic-Gate Classification

A pure-PyTorch implementation for studying trainable two-input logic-gate networks on MNIST and CIFAR-10. Each neuron has fixed sparse input connections and learns a mixture over 16 Boolean operators; inference can collapse each mixture to its argmax operator.

This is a local implementation of the method in [Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277), not the authors' official code. It provides a foundation for the related [convolutional implementation](https://github.com/13710833172/Conv-DLGN) and [Logic-SNN experiments](https://github.com/13710833172/Logic-SNN).

## Method

```text
Image -> flatten / threshold embedding -> sparse two-input logic layers
      -> class-group sum -> class scores
Soft training: mixtures of operators
Hard inference: one operator per gate, evaluated through truth tables
```

The implementation includes deterministic connectivity, validation-based checkpoint selection, gate statistics, hard export, and standalone evaluation. Gate counts and the number of trainable gate logits are different quantities: each soft gate has 16 logits.

## Recorded results

Representative completed configurations, summarized from local experiment results:

| Dataset / configuration | Soft test | Hard test |
| --- | ---: | ---: |
| MNIST normal | 98.15% | 97.86% |
| CIFAR-10 small | 48.80% | 47.91% |
| CIFAR-10 large | 55.58% | 55.44% |

These are observations under different settings, not multi-seed averages or a claim of reproducing the paper's full accuracy/speed results. Datasets, checkpoints, and experiment-history files are kept outside this repository. The summary was checked against local metrics; experiments were not rerun for this documentation update.

## Setup and training

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Full training configuration; this is not a short smoke run.
python src/train.py --config configs/mnist_small_paper.yaml
```

Other configurations cover MNIST normal and CIFAR-10 small/large models. Large widths are memory-intensive; the scripts in `scripts/check_*_feasibility.py` provide resource probes.

After training, replace the checkpoint path with the generated file:

```bash
python src/evaluate.py --checkpoint checkpoints/paper_small/best_model.pt
python src/discretize.py --checkpoint checkpoints/paper_small/best_model.pt
```

For a basic operator/structure check:

```bash
python -m pytest tests/test_logic_ops.py tests/test_connectivity.py tests/test_output_grouping.py -v
```

## Data and protocol boundaries

- MNIST currently uses continuous flattened pixels for training and thresholded pixels for validation/test. This behavior is explicit in `src/data/mnist.py`; do not assume identical transforms or exact paper fidelity.
- CIFAR-10 uses threshold-based binary embeddings, with separate small and large input dimensions.
- The data-loading source is tracked under `src/data/`; the root `/data/` directory is reserved for downloaded datasets.
- Validation selects checkpoints. Test evaluation is separate from selection; the optional per-epoch test diagnostic is disabled in the distributed example configurations.
- Pure-PyTorch hard execution is not the paper's optimized packed/CUDA backend. No paper speed or energy claim is reproduced here.

## Repository guide

| Path | Purpose |
| --- | --- |
| `src/data/` | Dataset loading and embedding |
| `src/models/`, `src/ops/` | Soft/hard networks and Boolean operators |
| `src/train.py`, `src/evaluate.py`, `src/discretize.py` | Train, evaluate, export |
| `configs/` | Current experiment configurations |
| [REFERENCE_NOTES.md](REFERENCE_NOTES.md) | Attribution and implementation scope |

