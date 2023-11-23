# BMT_GAN_stopping
Evaluation of stopping criteria for CycleGAN used to create synthetic biomedical segmentation data.

## Prerequisites
* [Anaconda Distribution](https://www.anaconda.com/distribution/#download-section).
* CUDA capable GPU.

## Installation
Clone the repository
```
git clone https://github.com/MoritzBoe/BMT_GAN_stopping.git
cd BMT_GAN_stopping
```
Create a new virtual environment:
```
conda env create -f requirements.yml
```
Activate the virtual environment:
```
conda activate BMT_GANstopping
```

## Data
For new real-world data, create a LightningDataModule in data_modules.py and add it to the get_data_module function. In this README, we describe usage on the BBBC039v1 dataset. 

Download the data from https://bbbc.broadinstitute.org/BBBC039 and extract the images to datasets/BBBC039/original/samples and the masks to datasets/BBBC039/original/labels/.

Run 
```
preprocess_bbbc039_dataset.py
```

## Training
The scrips are created to work with Weigths & Biases. You need to create an account at https://wandb.ai/. For personal usage, this is free. To setup W&B initialize it with:
```
wandb login YOURAPIKEY
```
You can find your API-Key under https://wandb.ai/settings.

The final setting used for our experiments are:
```
python lit_cyclegan.py --entity ml4home --project bbbc039 --data_module bbbc039 --path_train_A /srv/user/boehland/bmt-gan_stopping/datasets/bbbc039/synthetic/GAN/train/labels_preprocessed/ --path_train_B /srv/user/boehland/bmt-gan_stopping/datasets/bbbc039/preprocessed/train/samples/ --path_val_A /srv/user/boehland/bmt-gan_stopping/datasets/bbbc039/synthetic/GAN/val/labels_preprocessed/ --path_val_B /srv/user/boehland/bmt-gan_stopping/datasets/bbbc039/preprocessed/val/samples/ --batch_size 4 --crop_size 256 256 --criterion_idt_B MSE --criterion_idt_A MSE --criterion_cycle_ABA MSE --criterion_cycle_BAB MSE --inpainting_start_epoch 9999 --minmax_scale_A -1 1 --minmax_scale_B -1 1 --n_epochs_lr_fix 400 --n_epochs_lr_decay 400 -evm -cpn 10
```

Further hyperparameters can be displayed by running:
```
python lit_cyclegan.py -h
```
The optional hyperparameters are the hyperparameters used to configure the CycleGAN. The pl.Trainer arguments are the standard PyTorch lightning hyperparameters.

## Inference
BMT_infer.py is used to create new synthetic images from the label images. A folder with the ckeckpoints needs to be provided with --run_folder. Choose the desired folder from logs/bbbc039/[run]. The script will synthesize images for all checkpoints in this folder. The parent folder of all label images transferred is provided with --parent_test_images_A. It is set by default to the correct folder for the bbbc039 dataset which is datasets/bbbc039/synthetic/downstream/. All images in subfolders named labels_preprocessed are transferred. The synthesized images are saved to datasets/bbbc039/synthetic/downstream/[train/test/val]/samples/[model_name]/[checkpoint_name].

## Train segmentation model
After synthesizing paired training data, a segmentation model can be trained. In this work, we used KAIDA (https://gitlab.kit.edu/kit/iai/ml4home/kaida). The segmentation results on the BBBC039v1 test data can finally be compared to the metrics acquired during GAN training.