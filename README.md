# Accurate Cell Segmentation in Microscopy Imaging
This repository provides the Python implementation of our work on **3D cell segmentation** in microscopy images. We propose a cascaded deep learning architecture [1] consisting of two deep learning networks: an **image regression network** and a **voxel-wise classification network**. First, the double-decoder U-Net-based image regression network estimates two **Euclidean distance maps**, including an enhanced cell-interior map and a border-enhanced map. Next, the difference of these two maps is fed as input to another U-Net-based voxel-wise classification network, which outputs a **semantic segmentation mask**. Finally, an instance-wise mask is provided by utilizing the outputs of the two networks in a classical **seeded watershed** approach.

![cell-segmentation](cell_segmentation_overview_diagram.png)
---

## Installation
### Install environment for 'distance map estimation network':
```bash
mamba create -n map_est python=3.8
conda activate map_est
mamba install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
mamba install -c conda-forge openh264
mamba install conda-forge::python-graphviz
pip install -r requirements.txt
```
### Install environment for 'voxel-wise classification network':
```bash
conda create --name final_seg python=3.8
conda activate final_seg
cd MONAI-0.5.2
pip install -e ".[all]"
```
---

## Training
To train the network for the segmentation task, input–label pairs are needed.

- **Input:** intensity image (`uint8`)
- **Label:** instance image (`uint16`)

Training data should be organized as:
./train_data/01 → input images
./train_data/01_ST/SEG → label images

### Train distance map estimation network
```bash
# Activate environment
conda activate map_est
# Step 1: Generate input–ground-truth (distance maps) pairs
python prepare_groundtruth_network_distance_maps.py
# Step 2: Train the distance map estimation network
python Train_network_distance_maps.py
# The trained model will be saved in:
# ./Trained_model_distance_maps/distance_model_01.pth
```

### Train semantic segmentation network
```bash
# Activate environment
conda activate final_seg
# Step 1: Generate input (difference distance map) and ground-truth (3-class voxel-wise mask)
python prepare_groundtruth_semantic_seg_network.py
# Step 2: Train the semantic segmentation network
python Train_network_semantic_segmentation.py
# The trained model will be saved in:
# ./Trained_model_semantic_seg/best_metric_model_dicefocal.pth
```
---

## Testing
A test image of a synthetic bacterial biofilm (3D image) and **pre-trained models** for both networks are provided.  
Download the pre-trained models from [**here**](https://drive.google.com/file/d/1vMreJ3b3GYIKjEA5xucIr6T91DCQRGC2/view?usp=drive_link) and extract them into the current working directory.

### Evaluate distance map estimation network
```bash
# Activate environment
conda activate map_est

# Run the evaluation script
python evaluate_distance_map_estimation.py

# Note:
# - If using a newly trained model, update the 'path_model' variable in the script.
# - Otherwise, the provided pre-trained model will be used by default.
```
### Evaluate semantic segmentation network
```bash
# Activate environment
conda activate final_seg

# Run the evaluation script
python evaluate_semantic_segmentation.py

# Note:
# - If using a newly trained model, update the 'path_model' variable in the script.
# - Otherwise, the provided pre-trained model will be used by default.
```

## Citation:
[1] <a href="https://www.sciencedirect.com/science/article/abs/pii/S0957417423025964" target="_blank">Toma, T. T., Wang, Y., Gahlmann, A., & Acton, S. T. (2023, October). DeepSeeded: Volumetric segmentation of dense cell populations with a cascade of deep neural networks in bacterial biofilm
applications. Expert Systems with Applications, 122094.</a>
    



