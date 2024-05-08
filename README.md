This repository provides the Python implementation of our work on cell segmentation in 3D microscopy images using a cascaded deep learning architecture proposed in the paper[1]. The cascaded architecture consists of two deep learning networks: an image regression network and a voxel-wise classification network. First, the image regression network estimates two Euclidean distance maps, including an enhanced cell-interior map and a border map. Next, the difference of these two maps is fed as input to the voxel-wise classification network, which outputs a semantic segmentation mask. Finally, an instance-wise mask is provided by utilizing the outputs of the two networks in a classical seeded watershed approach.

# Installation
## Install environment for the regression network:
mamba create -n map_est python=3.8 <br />
conda activate map_est <br />
mamba install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia <br />
mamba install -c conda-forge openh264 <br />
mamba install conda-forge::python-graphviz <br />
pip install -r requirements.txt

## Install environment for the voxel-wise classification network:
conda create --name final_seg python=3.8 <br />
conda activate final_seg <br />
cd MONAI-0.5.2 <br />
pip install -e ".[all]" <br />

