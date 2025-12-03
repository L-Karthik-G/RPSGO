import os
import shutil
from pathlib import Path
from PIL import Image
import random

def create_yolo_dataset():
    """Convert rock-paper-scissors dataset to YOLO format"""
    
    # Paths
    source_dir = 'dataset'
    output_dir = 'yolo_dataset'
    
    # Create output structure
    for split in ['train', 'val']:
        os.makedirs(f'{output_dir}/images/{split}', exist_ok=True)
        os.makedirs(f'{output_dir}/labels/{split}', exist_ok=True)
    
    # Class mapping
    classes = {'rock': 0, 'paper': 1, 'scissors': 2}
    
    print("Converting dataset to YOLO format...")
    
    for class_name, class_id in classes.items():
        class_path = os.path.join(source_dir, class_name)
        
        if not os.path.exists(class_path):
            print(f"Warning: {class_path} not found!")
            continue
        
        images = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Processing {len(images)} images for class '{class_name}'...")
        
        # Shuffle and split 80/20
        random.shuffle(images)
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Process training images
        for img_name in train_images:
            process_image(class_path, img_name, class_id, output_dir, 'train')
        
        # Process validation images
        for img_name in val_images:
            process_image(class_path, img_name, class_id, output_dir, 'val')
    
    # Create data.yaml
    create_yaml(output_dir)
    
    print(f"\nDataset conversion complete!")
    print(f"Train images: {len(os.listdir(f'{output_dir}/images/train'))}")
    print(f"Val images: {len(os.listdir(f'{output_dir}/images/val'))}")

def process_image(class_path, img_name, class_id, output_dir, split):
    """Process single image and create YOLO label"""
    src_img = os.path.join(class_path, img_name)
    
    try:
        # Open image to get dimensions
        img = Image.open(src_img)
        img_width, img_height = img.size
        
        # Create bounding box (assume hand is centered, covering 80% of image)
        # Format: class_id center_x center_y width height (all normalized 0-1)
        center_x = 0.5
        center_y = 0.5
        bbox_width = 0.8
        bbox_height = 0.8
        
        # Copy image
        new_img_name = f"{class_id}_{img_name}"
        dst_img = os.path.join(output_dir, 'images', split, new_img_name)
        shutil.copy(src_img, dst_img)
        
        # Create label file
        label_name = os.path.splitext(new_img_name)[0] + '.txt'
        dst_label = os.path.join(output_dir, 'labels', split, label_name)
        
        with open(dst_label, 'w') as f:
            f.write(f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}\n")
    
    except Exception as e:
        print(f"Error processing {img_name}: {e}")

def create_yaml(output_dir):
    """Create data.yaml file for YOLO training"""
    yaml_content = f"""
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

# Classes
nc: 3
names: ['rock', 'paper', 'scissors']
"""
    
    with open(f'{output_dir}/data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"\nCreated data.yaml at {output_dir}/data.yaml")

if __name__ == '__main__':
    random.seed(42)  # For reproducibility
    create_yolo_dataset()