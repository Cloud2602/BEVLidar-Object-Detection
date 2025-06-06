# BevLidar Object Detection

This project focuses on object detection using Bird’s Eye View (BEV) images generated from LiDAR point clouds. The dataset was created starting from the **TruckScenes-mini** dataset and consists of 800 images.

### 📄 Paper reference

For more information and to explore the source dataset in detail, refer to the \\TODO ADD PAPER.

▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

## 📦 Dataset and training summary

The dataset was generated using the `generate_bev_and_labels.py` script, which converts LiDAR point clouds into BEV images with and without Z-axis filtering. YOLOv8-compatible annotations were also produced.  
The model was trained using the Ultralytics YOLOv8 framework in two steps: an initial pretraining phase with a frozen backbone, followed by full fine-tuning with data augmentation.

\\TODO ADD IMAGES

▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

## 🧪 Testing and evaluation

To test the models, open the notebook and navigate to the section titled **YOLO Testing**.

#### ▶️ Setup and run

Install the required dependencies and download the dataset and model weights:

```bash
!pip install ultralytics
!pip install -U gdown

# Download and unzip dataset
!gdown 'https://drive.google.com/uc?id=YOUR_DATASET_ID'
!unzip -q DATASET_FINISHED.zip -d DATASET

# Download trained model weights
!gdown 'https://drive.google.com/uc?id=YOUR_WEIGHTS_ID'
```

### 📊 Model validation

The notebook includes two validation blocks to evaluate and compare performance:

- **Results from YOLO model**: validation of a pretrained YOLOv8 model  
- **Results from my trained YOLO model**: validation of the custom-trained model on BEV data

#### 📈 Metrics comparison  \\TODO ADD METRICS

| Model                   | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|------------------------|---------|--------------|-----------|--------|
| Pretrained YOLOv8      |  XX.X%  |     XX.X%    |   XX.X%   | XX.X%  |
| Custom-trained YOLOv8  |  **YY.Y%**  |     **YY.Y%**    |   **YY.Y%**   | **YY.Y%**  |

▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

### 🖼️ Visual testing on example images

The last section of the notebook allows testing the trained model on three selected BEV images. The predictions are visually compared with the ground truth annotations.

\\TODO ADD IMAGES
