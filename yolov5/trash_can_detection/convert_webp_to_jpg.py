from PIL import Image
import os
import glob

# Paths
train_webp = 'data/images/train/*.webp'
val_webp = 'data/images/val/*.webp'
test_webp = 'data/images/test/*.webp'

# Create output directories
os.makedirs('data/images/train_jpg', exist_ok=True)
os.makedirs('data/images/val_jpg', exist_ok=True)
os.makedirs('data/images/test_jpg', exist_ok=True)

# Convert train images
for img_path in glob.glob(train_webp):
    base_name = os.path.basename(img_path).replace('.webp', '.jpg')
    out_path = os.path.join('data/images/train_jpg', base_name)
    try:
        img = Image.open(img_path)
        img.save(out_path, 'JPEG')
        print(f"Converted {img_path} to {out_path}")
    except Exception as e:
        print(f"Error converting {img_path}: {e}")

# Convert val images
for img_path in glob.glob(val_webp):
    base_name = os.path.basename(img_path).replace('.webp', '.jpg')
    out_path = os.path.join('data/images/val_jpg', base_name)
    try:
        img = Image.open(img_path)
        img.save(out_path, 'JPEG')
        print(f"Converted {img_path} to {out_path}")
    except Exception as e:
        print(f"Error converting {img_path}: {e}")

# Convert test images
for img_path in glob.glob(test_webp):
    base_name = os.path.basename(img_path).replace('.webp', '.jpg')
    out_path = os.path.join('data/images/test_jpg', base_name)
    try:
        img = Image.open(img_path)
        img.save(out_path, 'JPEG')
        print(f"Converted {img_path} to {out_path}")
    except Exception as e:
        print(f"Error converting {img_path}: {e}")

print("Conversion complete!")
