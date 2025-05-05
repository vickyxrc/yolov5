#!/bin/bash
# Get all webp files
images=($(find raw_images -name "*.webp"))
total=${#images[@]}

if [ $total -eq 0 ]; then
  echo "No webp files found in raw_images directory!"
  exit 1
fi

# Calculate splits (70% train, 20% val, 10% test)
train=$(echo "$total * 0.7 / 1" | bc)
val=$(echo "$total * 0.2 / 1" | bc)

# Make sure directories exist
mkdir -p data/images/train data/images/val data/images/test
mkdir -p data/labels/train data/labels/val data/labels/test

echo "Found $total webp files to split"

# Move files to train directory
for ((i=0; i<train; i++)); do
  cp "${images[i]}" data/images/train/
  filename=$(basename "${images[i]}")
  echo "Copied $filename to train set"
done

# Move files to val directory
for ((i=train; i<train+val; i++)); do
  cp "${images[i]}" data/images/val/
  filename=$(basename "${images[i]}")
  echo "Copied $filename to validation set"
done

# Move remaining files to test directory
for ((i=train+val; i<total; i++)); do
  cp "${images[i]}" data/images/test/
  filename=$(basename "${images[i]}")
  echo "Copied $filename to test set"
done

echo "Dataset split complete:"
echo "Train: $(ls data/images/train | wc -l) images"
echo "Val: $(ls data/images/val | wc -l) images"
echo "Test: $(ls data/images/test | wc -l) images"
