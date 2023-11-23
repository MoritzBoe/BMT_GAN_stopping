import numpy as np
from pathlib import Path
import os
import skimage.morphology
from skimage.measure import regionprops
from PIL import Image


def extract_percentile(data, percentile_min, percentile_max):
    min_value = np.percentile(data, percentile_min)
    max_value = np.percentile(data, percentile_max)
    data_percentile = [x for x in data if min_value<=x<=max_value]
    return data_percentile

def extract_properties(folder, save_labels=False, dim=None, remove_percentiles=False):
    """Extract object properties from data.

    Args:
        folder (str or Path): Folder with the label images.
        save_labels (bool, optional): Save label images.
        dim ([None, int], optional): Dimension for mask. If None, 2D image is expected. Defaults to None.
        remove_percentiles (bool, optional): Remove <5% and >95% percentile data before calculating properties. Defaults to False.
    """
    props = {}
    props["area"] = []
    props["equivalent_diameter"] = []
    props["major_axis_length"] = []
    props["minor_axis_length"] = []
    props["eccentricity"] = []
    props["cells_per_image"] = []
    filenames = os.listdir(folder)
    folder = Path(folder)
    
    if save_labels:
        labels_folder = folder.parent.joinpath("labels")
        labels_folder.mkdir(exist_ok=True)

    for filename in filenames:
        mask = np.array(Image.open(folder.joinpath(filename)))
        if dim != None:
            mask = mask[:,:,dim]
        label, num_features_skimage = skimage.morphology.label(mask, return_num=True)
        
        # extract elements per image before removing border elements
        props["cells_per_image"].append(num_features_skimage)
        
        if np.round(1.0*119) == num_features_skimage:
            print(f"{filename=}")
        # remove border elements to remove bias for shape features
        x = np.concatenate((np.arange(0,label.shape[0]),
                            np.arange(0,label.shape[0]),
                            np.zeros(label.shape[1]),
                            np.full(label.shape[1], fill_value=label.shape[0]-1)), axis=0).astype(np.uint16)
        y = np.concatenate((np.zeros(label.shape[0]),
                            np.full(label.shape[0], fill_value=label.shape[1]-1),
                            np.arange(0,label.shape[1]),
                            np.arange(0,label.shape[1])), axis=0).astype(np.uint16)
        border_elements = np.unique(label[x,y])
        for element in border_elements:
            if element == 0:
                continue
            label[np.where(label == element)] = 0

        if num_features_skimage == 0:
            print("Keine Zellen in Label", filename, "gefunden!")

        if save_labels:
            label_PIL = Image.fromarray(label.astype("uint16")) 
            label_PIL.save(labels_folder.joinpath(filename))


        probs_img = regionprops(label)

        # Extract data
        for prob in probs_img:
            props["area"].append(prob.area)
            props["equivalent_diameter"].append(prob.equivalent_diameter)
            props["major_axis_length"].append(prob.major_axis_length)
            props["minor_axis_length"].append(prob.minor_axis_length)
            props["eccentricity"].append(prob.eccentricity)

    # Extract perc_min-perc_max percentile to account for extreme values
    percentil_min = 5
    percentil_max = 95
    if remove_percentiles:
        props["area"] = extract_percentile(props["area"], percentil_min, percentil_max)
        props["equivalent_diameter"] = extract_percentile(props["equivalent_diameter"], percentil_min, percentil_max)
        props["major_axis_length"] = extract_percentile(props["major_axis_length"], percentil_min, percentil_max)
        props["minor_axis_length"] = extract_percentile(props["minor_axis_length"], percentil_min, percentil_max)
        props["eccentricity"] = extract_percentile(props["eccentricity"], percentil_min, percentil_max)
        props["cells_per_image"] = extract_percentile(props["cells_per_image"], percentil_min, percentil_max)

    area_mean = np.mean(props["area"])
    area_median = np.median(props["area"])
    area_std = np.std(props["area"])
    equivalent_diameter_mean = np.mean(props["equivalent_diameter"])
    equivalent_diameter_median = np.median(props["equivalent_diameter"])
    equivalent_diameter_std = np.std(props["equivalent_diameter"])
    major_axis_length_mean = np.mean(props["major_axis_length"])
    major_axis_length_median = np.median(props["major_axis_length"])
    major_axis_length_std = np.std(props["major_axis_length"])
    minor_axis_length_mean = np.mean(props["minor_axis_length"])
    minor_axis_length_median = np.median(props["minor_axis_length"])
    minor_axis_length_std = np.std(props["minor_axis_length"])
    eccentricity_mean = np.mean(props["eccentricity"])
    eccentricity_median = np.median(props["eccentricity"])
    eccentricity_std = np.std(props["eccentricity"])
    cells_per_image_mean = np.mean(props["cells_per_image"])
    cells_per_image_median = np.median(props["cells_per_image"])
    cells_per_image_std = np.std(props["cells_per_image"])


    properties_print = ["Cell area mean: " + str(area_mean), "\nCell area median: " + str(area_median),
            "\nCell area std: " + str(area_std),
            "\nEquivalent diameter mean: " + str(equivalent_diameter_mean),
            "\nEquivalent diameter median: " + str(equivalent_diameter_median),
            "\nEquivalent diameter std: " + str(equivalent_diameter_std),
            "\nMajor axis length mean: " + str(major_axis_length_mean),
            "\nMajor axis length median: " + str(major_axis_length_median),
            "\nMajor axis length std: " + str(major_axis_length_std),
            "\nMinor axis length mean: " + str(minor_axis_length_mean),
            "\nMinor axis length median: " + str(minor_axis_length_median),
            "\nMinor axis length std: " + str(minor_axis_length_std),
            "\nEccentricity mean: " + str(eccentricity_mean), "\nEccentricity median: " + str(eccentricity_median),
            "\nEccentricity std: " + str(eccentricity_std),
            "\nCells per Image mean: " + str(cells_per_image_mean),
            "\nCells per Image median: " + str(cells_per_image_median),
            "\nCells per Image std: " + str(cells_per_image_std)]

    file_results = open(folder.parent.joinpath("properties.txt"), "w+")
    file_results.writelines(properties_print)
    file_results.close()
    
    return props

if __name__ == "__main__":
    folder = Path("datasets/bbbc039/masks")  # needs dim=0
    extract_properties(folder, save_labels=False, dim=0, remove_percentiles=True)
