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

## Download NOAH Video Dataset

The NOAH video files are hosted on Hugging Face: [[Dataset]](https://huggingface.co/datasets/anonymous550520/NOAH)

Download the dataset to `./data/videos/`

```bash
mkdir -p ./data/videos

hf download anonymous550520/NOAH \
  --repo-type dataset \
  --local-dir ./data/videos
```

After downloading, extract the video archives:

```
cd ./data/videos

tar -xvf videos/noah_original_videos.tar
tar -xvf videos/noah_composite_videos.tar
```

This will create the following directories:

```
data/videos/
  noah_original_videos/
  noah_composite_videos/
```

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
# Evaluate on composite videos
./scripts/evaluate_captions_composite.sh

# Evaluate on original videos
./scripts/evaluate_captions_original.sh
```

### QA Evaluation

```bash
# Evaluate narrative QA results
./scripts/evaluate_qa.sh
```
