# NOAH: Benchmarking Narrative Prior Driven Hallucination and Omission in Video Large Language Models

This repository is the official implementation of the paper **"NOAH: Benchmarking Narrative Prior Driven Hallucination and Omission in Video Large Language Models"**.
[[Project Page]](https://anonymous550520.github.io/)

## Download NOAH Video Dataset

### Installation

Create a conda environment and install dependencies:

```bash
conda create -n noah python=3.10 -y
conda activate noah
pip install -r requirements.txt
```

### Build Dataset

Run the dataset building script:

```bash
./scripts/build_dataset.sh
```

The dataset will be saved to `./data/noah/videos/`

## Running Captioning and QA

### Captioning

```bash
# Run captioning on the dataset
./scripts/run_captioning.sh MODEL_NAME

# Example
./scripts/run_captioning.sh video_llama3_7b
```

### QA

```bash
# Run QA on the dataset
./scripts/run_qa.sh MODEL_NAME

# Example
./scripts/run_qa.sh video_llama3_7b
```

## Evaluating Captioning and QA Results

### Captioning Evaluation

```bash
# Evaluate captioning results
./scripts/evaluate_captions.sh
```

### QA Evaluation

```bash
# Evaluate QA results
./scripts/evaluate_qa.sh
```

## Citation

If you use NOAH in a research paper, please cite our work as follows:

```bibtex
@article{anonymous2025,
  author    = {Anonymous},
  title     = {NOAH: Benchmarking Narrative Prior Driven Hallucination and Omission in Video Large Language Models},
  journal   = {Under Review},
  year      = {2026},
}
```