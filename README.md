# PyTorch UNet-Dreamer

## SAM annotation pipeline 
- Initially I tried this but this is not robust to reflections so left it after a point...still one can try for faster labelling...otherwise for doing the manual annotation use Roboflow

scripts/sam_annotation_pipeline.py

For environment instructions follow https://github.com/facebookresearch/segment-anything

## Installation

1. Set up a new environment with an environment manager (recommended):
   1. [conda](https://docs.conda.io/en/latest/miniconda.html):
      1. `conda create --name unet_dreamer -y`
      2. `conda activate unet_dreamer`
      3. `conda install python=3.8 -y`<br>

      OR 

   2. [venv](https://docs.python.org/3/library/venv.html):
      1. `python3 -m venv unet_dreamer`
      2. `source unet_dreamer/bin/activate`
2. Install the libraries:
`pip install -r requirements.txt`

## Data preparation
- First place the images and their corresponding masks in the 'seg-cam_images' directory. Currently, there are 3 subdirectories inside it, representing dataset captured under different
conditions. All will be used as part of train/val dataset.
- Since the images were annotated using Roboflow so the images and segmentation masks were by default downloaded in the same directory, for example, 'seg-cam_images/multi_agent' has both the images and the target segmentation masks (with each pixel representing an integer between 0-3 since there were 4 semantic categories in autonomous rover task), similarly for other subdirectories.
<br>
- Once the dataset is placed in the 'seg-cam_images' directory, then we can run the 'scripts/create_dataset.py' which will split the images/segmentation masks into the 'dataset_large/train' and 'dataset_large/val'. Feel free to change the split ratio in the 'scripts/create_dataset.py'. A small sample of the rover segemntation dataset is left there for reference. Before starting the training do ensure to empty the 'seg-cam_images', 'dataset_large/train/images', 'dataset_large/train/targets', 'dataset_large/val/images', and 'dataset_large/val/targets' directories.

## Training
- Run the 'train_notebook.ipynb'

- The trained weights will be placed in the 'checkpoints' directory

## Inference
- Run the 'inference_notebook.ipynb'