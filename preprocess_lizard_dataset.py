from pathlib import Path
from PIL import Image
import numpy as np


lizard_folder = Path("datasets/lizard/original")
lizard_images = lizard_folder.joinpath("images.npy")
lizard_labels = lizard_folder.joinpath("labels.npy")
train_folder = lizard_folder.parent.joinpath("train")
val_folder = lizard_folder.parent.joinpath("val")
test_folder = lizard_folder.parent.joinpath("test")
train_img_folder = train_folder.joinpath("samples")
train_label_folder = train_folder.joinpath("labels")
val_img_folder = val_folder.joinpath("samples")
val_label_folder = val_folder.joinpath("labels")
test_img_folder = test_folder.joinpath("samples")
test_label_folder = test_folder.joinpath("labels")


train_img_folder.mkdir(exist_ok=True, parents=True)
train_label_folder.mkdir(exist_ok=True, parents=True)
val_img_folder.mkdir(exist_ok=True, parents=True)
val_label_folder.mkdir(exist_ok=True, parents=True)
test_img_folder.mkdir(exist_ok=True, parents=True)
test_label_folder.mkdir(exist_ok=True, parents=True)

images = np.load(lizard_images)
labels = np.load(lizard_labels)

idx = []
for i in range(labels.shape[0]):
    if labels[i,...,0].max()>0:
        idx.append(i)


train_idx = idx[0:int(len(idx)*0.6)]
val_idx = idx[int(len(idx)*0.6):int(len(idx)*0.8)]
test_idx = idx[int(len(idx)*0.8):]

for i in train_idx:
    img = Image.fromarray(images[i,...])
    label = Image.fromarray(labels[i,...,0].astype("uint16"))
    img.save(train_img_folder.joinpath(f"{i:05d}.tif"))
    label.save(train_label_folder.joinpath(f"{i:05d}.tif"))   
for i in val_idx:
    img = Image.fromarray(images[i,...])
    label = Image.fromarray(labels[i,...,0].astype("uint16"))
    img.save(val_img_folder.joinpath(f"{i:05d}.tif"))
    label.save(val_label_folder.joinpath(f"{i:05d}.tif"))  
for i in test_idx:
    img = Image.fromarray(images[i,...])
    label = Image.fromarray(labels[i,...,0].astype("uint16"))
    img.save(test_img_folder.joinpath(f"{i:05d}.tif"))
    label.save(test_label_folder.joinpath(f"{i:05d}.tif"))    
