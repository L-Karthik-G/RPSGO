"""
Optimized Training Script for Rock Paper Scissors Detection
Dataset: ~3200 images
Hardware: RTX 5050 (8GB VRAM) + 16GB RAM
"""

from ultralytics import YOLO
import torch
import os
import yaml

def check_hardware():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return True
    else:
        response = input("No GPU detected. Continue on CPU? (y/n): ")
        return response.lower() == 'y'

def check_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return False
    try:
        with open(dataset_path, 'r') as f:
            data = yaml.safe_load(f)
        train_path = data.get('train', 'train/images')
        if not os.path.isabs(train_path):
            base_dir = os.path.dirname(dataset_path)
            train_path = os.path.join(base_dir, train_path)
        if os.path.exists(train_path):
            return True
        else:
            print(f"Train path not found: {train_path}")
            return False
    except Exception as e:
        print(f"Could not read dataset file: {e}")
        return False

def train_model():
    if not check_hardware():
        return
    dataset_path = 'rock-paper-scissors.v1i.yolov8/data.yaml'
    if not check_dataset(dataset_path):
        return

    choice = input("Choose mode - 1) Fine-tune  2) From scratch  (enter 1 or 2): ").strip()
    if choice == '1':
        model_path = 'runs/detect/rps_model/weights/best.pt'
        if not os.path.exists(model_path):
            model = YOLO('yolov8n.pt')
            epochs = 80
            lr0 = 0.01
        else:
            model = YOLO(model_path)
            epochs = 60
            lr0 = 0.001
    else:
        model = YOLO('yolov8n.pt')
        epochs = 80
        lr0 = 0.01

    confirm = input("Start training? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return

    try:
        results = model.train(
            data=dataset_path,
            epochs=epochs,
            patience=20,
            imgsz=640,
            batch=24,
            device=0,
            amp=True,
            workers=8,
            cache=False,
            lr0=lr0,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            cos_lr=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            mosaic=1.0,
            name='rps_optimized',
            exist_ok=True,
            pretrained=True,
            optimizer='auto',
            verbose=True,
            seed=42,
            deterministic=True,
            val=True,
            split='val',
            save=True,
            plots=True,
        )
        best_model_path = 'runs/detect/rps_optimized/weights/best.pt'
        last_model_path = 'runs/detect/rps_optimized/weights/last.pt'
        print(f"Training complete. Best: {best_model_path}  Last: {last_model_path}")
        try:
            import pandas as pd
            df = pd.read_csv('runs/detect/rps_optimized/results.csv')
            df.columns = df.columns.str.strip()
            final_map = df['metrics/mAP50(B)'].iloc[-1]
            best_map = df['metrics/mAP50(B)'].max()
            final_precision = df['metrics/precision(B)'].iloc[-1]
            final_recall = df['metrics/recall(B)'].iloc[-1]
            print(f"mAP50: {final_map:.3f} (Best: {best_map:.3f})  Precision: {final_precision:.3f}  Recall: {final_recall:.3f}")
        except Exception:
            pass
    except KeyboardInterrupt:
        print("Training interrupted by user. Progress saved.")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == '__main__':
    train_model()