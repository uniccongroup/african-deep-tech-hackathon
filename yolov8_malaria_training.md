# YOLOv8 Malaria Classification Training Guide

## Dataset Structure
Your current structure is ideal for YOLOv8 classification:
```
hackathon/
└── cell_images/
    ├── Parasitized/
    │   ├── C33P1thinF_IMG_20150619_114756_cell_179.png
    │   ├── C33P1thinF_IMG_20150619_114756_cell_180.png
    │   └── ... (more parasitized cell images)
    └── Uninfected/
        ├── C1_thinF_IMG_20150604_104722_cell_9.png
        ├── C1_thinF_IMG_20150604_104722_cell_15.png
        └── ... (more uninfected cell images)
```

## Step 1: Environment Setup

### Install Required Packages
```bash
pip install ultralytics
pip install torch torchvision torchaudio
pip install opencv-python
pip install matplotlib
pip install scikit-learn
```

### Import Libraries
```python
import os
import shutil
from pathlib import Path
import random
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
```

## Step 2: Dataset Preparation

### Create Train/Val/Test Split
```python
import os
import shutil
import random
from pathlib import Path

def create_dataset_split(source_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split dataset into train/val/test for YOLOv8 classification
    """
    # Create output directories
    splits = ['train', 'val', 'test']
    classes = ['Parasitized', 'Uninfected']
    
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)
    
    # Process each class
    for cls in classes:
        class_dir = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        # Split images
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        # Copy images to respective directories
        for img in train_images:
            shutil.copy2(os.path.join(class_dir, img), os.path.join(output_dir, 'train', cls, img))
        
        for img in val_images:
            shutil.copy2(os.path.join(class_dir, img), os.path.join(output_dir, 'val', cls, img))
        
        for img in test_images:
            shutil.copy2(os.path.join(class_dir, img), os.path.join(output_dir, 'test', cls, img))
        
        print(f"{cls}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

# Usage
source_directory = "hackathon/cell_images"
output_directory = "malaria_dataset"
create_dataset_split(source_directory, output_directory)
```

## Step 3: Training Configuration

### Create Training Script
```python
from ultralytics import YOLO
import torch

def train_malaria_classifier():
    """
    Train YOLOv8 classification model for malaria detection
    """
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load YOLOv8 classification model
    model = YOLO('yolov8n-cls.pt')  # nano model for faster training
    # Alternative models: yolov8s-cls.pt, yolov8m-cls.pt, yolov8l-cls.pt, yolov8x-cls.pt
    
    # Training parameters
    results = model.train(
        data='malaria_dataset',  # path to dataset
        epochs=100,              # number of training epochs
        imgsz=224,              # image size (224x224 is standard for classification)
        batch=16,               # batch size (adjust based on your GPU memory)
        device=device,          # device to use
        workers=4,              # number of worker threads
        patience=10,            # early stopping patience
        save=True,              # save model checkpoints
        save_period=10,         # save checkpoint every 10 epochs
        val=True,               # validate during training
        plots=True,             # save training plots
        verbose=True,           # verbose output
        seed=42,                # random seed for reproducibility
        
        # Data augmentation parameters
        hsv_h=0.015,           # HSV-Hue augmentation
        hsv_s=0.7,             # HSV-Saturation augmentation  
        hsv_v=0.4,             # HSV-Value augmentation
        degrees=0.0,           # rotation degrees
        translate=0.1,         # translation
        scale=0.5,             # scale
        shear=0.0,             # shear
        perspective=0.0,       # perspective
        flipud=0.0,            # flip up-down
        fliplr=0.5,            # flip left-right
        mosaic=0.0,            # mosaic augmentation (not typically used for classification)
        mixup=0.0,             # mixup augmentation
        copy_paste=0.0,        # copy-paste augmentation
        
        # Optimization parameters
        optimizer='AdamW',      # optimizer (SGD, Adam, AdamW)
        lr0=0.001,             # initial learning rate
        lrf=0.1,               # final learning rate factor
        momentum=0.937,        # momentum
        weight_decay=0.0005,   # weight decay
        warmup_epochs=3,       # warmup epochs
        warmup_momentum=0.8,   # warmup momentum
        warmup_bias_lr=0.1,    # warmup bias learning rate
        
        # Model parameters
        dropout=0.0,           # dropout rate
        
        # Project organization
        project='malaria_classification',  # project name
        name='yolov8_malaria',            # experiment name
    )
    
    return model, results

# Train the model
model, results = train_malaria_classifier()
```

## Step 4: Model Evaluation

### Evaluate on Test Set
```python
def evaluate_model(model_path, test_data_path):
    """
    Evaluate trained model on test set
    """
    # Load trained model
    model = YOLO(model_path)
    
    # Validate on test set
    results = model.val(data=test_data_path, split='test')
    
    print(f"Test Accuracy: {results.top1:.4f}")
    print(f"Test Top-5 Accuracy: {results.top5:.4f}")
    
    return results

# Evaluate the model
test_results = evaluate_model('malaria_classification/yolov8_malaria/weights/best.pt', 
                             'malaria_dataset')
```

### Create Confusion Matrix
```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def create_confusion_matrix(model_path, test_dir):
    """
    Create confusion matrix for the trained model
    """
    model = YOLO(model_path)
    
    # Get predictions for test set
    test_parasitized = os.path.join(test_dir, 'test', 'Parasitized')
    test_uninfected = os.path.join(test_dir, 'test', 'Uninfected')
    
    y_true = []
    y_pred = []
    
    # Process Parasitized images
    for img_name in os.listdir(test_parasitized):
        img_path = os.path.join(test_parasitized, img_name)
        results = model(img_path)
        predicted_class = results[0].names[results[0].probs.top1]
        y_true.append('Parasitized')
        y_pred.append(predicted_class)
    
    # Process Uninfected images
    for img_name in os.listdir(test_uninfected):
        img_path = os.path.join(test_uninfected, img_name)
        results = model(img_path)
        predicted_class = results[0].names[results[0].probs.top1]
        y_true.append('Uninfected')
        y_pred.append(predicted_class)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['Parasitized', 'Uninfected'])
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Parasitized', 'Uninfected'],
                yticklabels=['Parasitized', 'Uninfected'])
    plt.title('Confusion Matrix - Malaria Classification')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return cm

# Create confusion matrix
cm = create_confusion_matrix('malaria_classification/yolov8_malaria/weights/best.pt', 
                            'malaria_dataset')
```

## Step 5: Model Inference

### Single Image Prediction
```python
def predict_single_image(model_path, image_path):
    """
    Predict malaria status for a single image
    """
    model = YOLO(model_path)
    
    # Make prediction
    results = model(image_path)
    
    # Get prediction details
    predicted_class = results[0].names[results[0].probs.top1]
    confidence = results[0].probs.top1conf.item()
    
    print(f"Image: {image_path}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    
    return predicted_class, confidence

# Example usage
prediction, confidence = predict_single_image(
    'malaria_classification/yolov8_malaria/weights/best.pt',
    'path/to/test/image.png'
)
```

### Batch Prediction
```python
def batch_predict(model_path, image_folder):
    """
    Predict malaria status for multiple images
    """
    model = YOLO(model_path)
    
    results = []
    for img_name in os.listdir(image_folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, img_name)
            prediction = model(img_path)
            
            predicted_class = prediction[0].names[prediction[0].probs.top1]
            confidence = prediction[0].probs.top1conf.item()
            
            results.append({
                'image': img_name,
                'prediction': predicted_class,
                'confidence': confidence
            })
    
    return results

# Example usage
batch_results = batch_predict(
    'malaria_classification/yolov8_malaria/weights/best.pt',
    'path/to/test/folder'
)
```

## Step 6: Model Optimization Tips

### Hyperparameter Tuning
```python
def hyperparameter_tuning():
    """
    Perform hyperparameter tuning for better performance
    """
    model = YOLO('yolov8n-cls.pt')
    
    # Hyperparameter search space
    search_space = {
        'lr0': [0.001, 0.01, 0.1],
        'batch': [8, 16, 32],
        'epochs': [50, 100, 150],
        'imgsz': [224, 256, 320]
    }
    
    best_results = None
    best_params = None
    
    for lr in search_space['lr0']:
        for batch in search_space['batch']:
            for epochs in search_space['epochs']:
                for imgsz in search_space['imgsz']:
                    results = model.train(
                        data='malaria_dataset',
                        epochs=epochs,
                        imgsz=imgsz,
                        batch=batch,
                        lr0=lr,
                        val=True,
                        verbose=False,
                        project='hyperparameter_tuning',
                        name=f'lr{lr}_batch{batch}_epochs{epochs}_imgsz{imgsz}'
                    )
                    
                    if best_results is None or results.results_dict['metrics/accuracy_top1'] > best_results:
                        best_results = results.results_dict['metrics/accuracy_top1']
                        best_params = {'lr0': lr, 'batch': batch, 'epochs': epochs, 'imgsz': imgsz}
    
    print(f"Best parameters: {best_params}")
    print(f"Best accuracy: {best_results}")
    
    return best_params

# Run hyperparameter tuning
best_params = hyperparameter_tuning()
```

## Step 7: Model Deployment

### Export Model for Production
```python
def export_model(model_path, export_formats=['onnx', 'tensorrt']):
    """
    Export trained model to different formats for deployment
    """
    model = YOLO(model_path)
    
    for format in export_formats:
        try:
            model.export(format=format)
            print(f"Model exported to {format} format successfully")
        except Exception as e:
            print(f"Failed to export to {format}: {e}")

# Export model
export_model('malaria_classification/yolov8_malaria/weights/best.pt')
```

### Create API Endpoint
```python
from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load model once when starting the server
model = YOLO('malaria_classification/yolov8_malaria/weights/best.pt')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        image_data = request.json['image']
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Make prediction
        results = model(image)
        predicted_class = results[0].names[results[0].probs.top1]
        confidence = results[0].probs.top1conf.item()
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': float(confidence),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        })

if __name__ == '__main__':
    app.run(debug=True)
```

## Key Training Tips

1. **Data Quality**: Ensure your images are of good quality and properly labeled
2. **Data Augmentation**: Use appropriate augmentation to improve model generalization
3. **Model Size**: Start with yolov8n (nano) for faster training, then try larger models if needed
4. **Learning Rate**: Start with 0.001 and adjust based on training progress
5. **Early Stopping**: Use patience parameter to prevent overfitting
6. **Validation**: Always validate during training to monitor performance
7. **Class Balance**: Ensure balanced dataset or use class weights if imbalanced

## Expected Performance

With proper training, you should expect:
- **Training Time**: 1-3 hours depending on dataset size and hardware
- **Accuracy**: 95%+ on malaria cell classification
- **Inference Speed**: Real-time prediction capability
- **Model Size**: 5-50MB depending on model variant

## Next Steps

1. Run the dataset preparation script
2. Start training with the provided configuration
3. Monitor training progress and adjust parameters as needed
4. Evaluate model performance on test set
5. Deploy the model for real-world use

Good luck with your malaria classification project!