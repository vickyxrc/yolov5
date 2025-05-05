#!/bin/bash
# Get all image files
images=($(find raw_images -name "*.jpg" -o -name "*.png" -o -name "*.jpeg"))
total=${#images[@]}
train=$(echo "$total * 0.7 / 1" | bc)
val=$(echo "$total * 0.2 / 1" | bc)

# Move files to train directory
for ((i=0; i<train; i++)); do
  cp "${images[i]}" data/images/train/
done

# Move files to val directory
for ((i=train; i<train+val; i++)); do
  cp "${images[i]}" data/images/val/
done

# Move remaining files to test directory
for ((i=train+val; i<total; i++)); do
  cp "${images[i]}" data/images/test/
done

echo "Dataset split complete:"
echo "Train: $(ls data/images/train | wc -l) images"
echo "Val: $(ls data/images/val | wc -l) images"
echo "Test: $(ls data/images/test | wc -l) images"
