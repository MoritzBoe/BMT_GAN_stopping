import numpy as np
from pathlib import Path
from PIL import Image
import skimage.morphology

def get_img_paths(folder, ending):
    if "*." in ending:
        ending = ending
    elif "." in ending:
        ending = "*"+ending
    else:
        ending = "*."+ending
    return list(Path(folder).glob(ending))

def extract_percentile(data, percentile_min, percentile_max):
    min_value = np.percentile(data, percentile_min)
    max_value = np.percentile(data, percentile_max)
    return min_value, max_value


if __name__ == "__main__":
    input_folder = Path("datasets/bbbc039/original/")
    output_folder = Path("datasets/bbbc039/preprocessed/")
    split = [120, 40, 40]
    
    output_folder_train_samples = output_folder.joinpath("train", "samples")
    output_folder_val_samples = output_folder.joinpath("val", "samples")
    output_folder_test_samples = output_folder.joinpath("test", "samples")
    output_folder_train_samples.mkdir(parents=True, exist_ok=True)
    output_folder_val_samples.mkdir(parents=True, exist_ok=True)
    output_folder_test_samples.mkdir(parents=True, exist_ok=True)

    output_folder_train_labels = output_folder.joinpath("train", "labels")
    output_folder_val_labels = output_folder.joinpath("val", "labels")
    output_folder_test_labels = output_folder.joinpath("test", "labels")
    output_folder_train_labels.mkdir(parents=True, exist_ok=True)
    output_folder_val_labels.mkdir(parents=True, exist_ok=True)
    output_folder_test_labels.mkdir(parents=True, exist_ok=True)

    labels_paths = sorted(get_img_paths(input_folder.joinpath("labels"), ".png"))
    samples_paths = sorted(get_img_paths(input_folder.joinpath("samples"), ".tif"))
    
    for idx, img_path in enumerate(samples_paths):
        img = np.array(Image.open(img_path)).astype(np.uint16)
        
        # normalize to 1, 99 percentile and afterwards to range of uint16
        norm_range = extract_percentile(img, 1, 99)

        img[img<norm_range[0]] = norm_range[0]
        img[img>norm_range[1]] = norm_range[1]
        img = ((img - norm_range[0]) * ((np.iinfo(img.dtype).max) / (norm_range[1] - norm_range[0]))).astype(np.uint16)
        
        img = Image.fromarray(img)
        label = np.array(Image.open(labels_paths[idx]))[:,:,0].astype(np.uint16)
        label = skimage.morphology.label(label).astype("uint16")
        label = Image.fromarray(label)

        if idx < split[0]:
            img.save(output_folder_train_samples.joinpath(img_path.name))
            label.save(output_folder_train_labels.joinpath(img_path.name))
        elif idx < np.sum(split[0:2]):
            img.save(output_folder_val_samples.joinpath(img_path.name))
            label.save(output_folder_val_labels.joinpath(img_path.name))
        elif idx < np.sum(split):
            img.save(output_folder_test_samples.joinpath(img_path.name))
            label.save(output_folder_test_labels.joinpath(img_path.name)) 
    
    print("Finished")
    
    