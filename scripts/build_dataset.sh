#!/bin/bash

# Unzip dataset
unzip -o data/noah/noah_dataset.zip -d data/noah/

python src/dataset/download.py

python src/dataset/extract_clips.py

python src/dataset/build_dataset.py

# Clean up temporary directories
rm -rf data/activitynet_clips
rm -rf data/activitynet_videos