from matplotlib.pyplot import imshow
import numpy as np
import random
import os
from tifffile import imwrite, imread
import warnings
from sklearn.mixture import GaussianMixture
import cv2 as cv
from pyefd import elliptic_fourier_descriptors, reconstruct_contour
from pathlib import Path
from skimage.measure import label, regionprops
from PIL import Image
from extract_dataset_properties import extract_properties

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class Dataset:
    def __init__(self, label_size, number_of_labels):
        self.label_size = label_size
        self.number_of_labels = number_of_labels
        self.labels = []

        for i in range(self.number_of_labels):
            self.labels.append(np.zeros(self.label_size, dtype='uint16'))

        self.label_pipeline = []
        self.labels_path = []

    def add_to_label_pipeline(self, shape):
        # Add a geometic shape to the pipeline for the label images
        self.label_pipeline.append(shape)
        print("Shape added to pipeline")

    def run_label_pipeline(self, label_number=-1):
        if label_number == -1:
            for idx, label in enumerate(self.labels):
                print(f"Creating label {idx+1} of {len(self.labels)}")
                for shape in self.label_pipeline:
                    self.labels[idx] = shape.draw(self.labels[idx])
        else:
            for shape in self.label_pipeline:
                self.labels[label_number] = shape.draw(self.labels[label_number])

    def show(self, number):
        imshow(self.labels[number])

    def save_labels(self, labels_path, ids=np.asarray([]), binarize=False):
        self.labels_path = Path(labels_path)
        create_directory(self.labels_path)
        if ids.size:
            for idx in ids:
                if binarize:
                    label = ((self.labels[idx] > 0)*np.iinfo(np.uint16).max).astype("uint16")
                else:
                    label = self.labels[idx]
                imwrite(self.labels_path.joinpath(f'label_{idx:05d}.tif'), label)
        else:
            for idx, label in enumerate(self.labels):
                if binarize:
                    label = ((label > 0)*np.iinfo(np.uint16).max).astype("uint16")
                imwrite(self.labels_path.joinpath(f'label_{idx:05d}.tif'), label)

    def save_LII(self, path, split):
        """Save created images to the format required for the label image influence evaluation

        Args:
            path (str or Path]): Path to the main folder where the data is saved
            split (list of int): Data split with number of images in [GAN_train, downstream_train, downstream_val, downstream_test]
        """
        assert np.sum(split) == self.number_of_labels, f"Sum of {split=} and {self.number_of_labels=} does not match"
        path = Path(path)
        path_GAN_train = path.joinpath("GAN", "train", "labels")
        path_GAN_val = path.joinpath("GAN", "val", "labels")
        path_downstream_train = path.joinpath("downstream", "train", "labels")
        path_downstream_val = path.joinpath("downstream", "val", "labels")
        path_downstream_test = path.joinpath("downstream", "test", "labels")

        GAN_train_ids = np.arange(0, split[0]).astype("int32")
        GAN_val_ids = np.arange(split[0], np.sum(split[0:2])).astype("int32")
        downstream_train_ids = np.arange(np.sum(split[0:2]), np.sum(split[0:3])).astype("int32")
        downstream_val_ids = np.arange(np.sum(split[0:3]), np.sum(split[0:4])).astype("int32")
        downstream_test_ids = np.arange(np.sum(split[0:4]), np.sum(split)).astype("int32")

        self.save_labels(path_GAN_train, GAN_train_ids, binarize=False)
        self.save_labels(path_GAN_val, GAN_val_ids, binarize=False)
        self.save_labels(path_downstream_train, downstream_train_ids, binarize=False)
        self.save_labels(path_downstream_val, downstream_val_ids, binarize=False)
        self.save_labels(path_downstream_test, downstream_test_ids, binarize=False)


    def add_to_image_pipeline(self, transformation):
        # Add a geometic shape to the pipeline for the label images
        self.image_pipeline.append(transformation)
        print("Shape added to pipeline")

    def run_image_pipeline(self, image_number=-1):
        for idx, label in enumerate(self.labels):
            self.images.append((self.images_foreground - self.images_background) * label.astype('uint16') +
                               self.images_background)

        if image_number == -1:
            for idx, image in enumerate(self.images):
                for transformation in self.image_pipeline:
                    self.images[idx] = transformation.apply(self.images[idx], self.labels[idx])
        else:
            for transformation in self.image_pipeline:
                self.images[image_number] = transformation.apply(self.images[image_number], self.labels[idx])

class EFD:
    def __init__(self, img_folder, ending, order, area, center=(0, 0), random_center=False, fully_visible=True,
                 max_overlap=0, random_rotation=False, num_shapes=(1, 0)):
        # 95% of values lie between mean +- 2*std for normal distributions
        self.img_folder = img_folder
        self.ending = ending
        self.order = order
        self.area = list(area) + [max(1, (area[0] - 2*area[1]))] + [area[0] + 2*area[1]]
        self.center = center
        self.random_center = random_center
        self.fully_visible = fully_visible
        self.max_overlap = max_overlap
        self.minor_axis_length = 0
        self.major_axis_length = 0
        self.random_rotation = random_rotation
        self.rotation = 0
        self.num_shapes = list(num_shapes) + [max(1, (num_shapes[0] - 2*num_shapes[1]))] + [num_shapes[0] + 2*num_shapes[1]]
        self.efds = np.array(self.get_efd_descriptors(self.order, self.img_folder, self.ending, remove_border_elements=True))
        self.gmm = GaussianMixture().fit(self.efds.reshape((self.efds.shape[0], -1)))
        self.smaller = []
        self.bigger = []

    def draw(self, label):
        if self.num_shapes[1] != 0:  # Standard deviation not 0
            num_shapes = np.random.normal(self.num_shapes[0], self.num_shapes[1])
            if num_shapes < self.num_shapes[2]:
                num_shapes = self.num_shapes[2]
            elif num_shapes > self.num_shapes[3]:
                num_shapes = self.num_shapes[3]
            num_shapes = np.round(num_shapes).astype("uint16")
        else:
            num_shapes = self.num_shapes[0]
        for i in range(num_shapes):
            x, y, shift = self.create_efd_poly()
            if self.random_center:
                for j in range(50000):
                    center = self.get_random_center(label.shape, shift)
                    coords_x = x + center[0]
                    coords_y = y + center[1]
                    placable, (coords_x, coords_y) = self.check_overlap_borders(label, coords_x, coords_y)
                    if placable:
                        label[coords_x, coords_y] = label.max()+1
                        break
                    if j == 9999:
                        warnings.warn(f"No spot found to place the circle with the given radius and max_overlap!\n "
                                      f"{i} out of {num_shapes} placed! Getting new ellipse to place")
            else:
                x += self.center[0]
                y += self.center[1]
                label[x, y] = label.max()+1
        return label
    
    def create_efd_poly(self):
        efd_sample = self.gmm.sample(1)[0]
        efd_sample = self.random_rotate_efd(efd_sample)
        area = self.get_random_area()
        
        # start with circle scaling
        scaling = np.sqrt(area/np.pi)
        multiplier = 0.5
        accept_side = np.random.choice([True, False])
        for i in range(1000):
            poly = reconstruct_contour(efd_sample, num_points=300)*scaling
            
            # convert to int and shift in positive range
            shift = abs(poly.astype(int).min())+1
            poly = poly.astype(int) + shift
            img_poly = np.zeros((poly[:,0].max()+20, poly[:,1].max()+20))     
            poly= tuple([np.expand_dims(poly,1)])
            # draw in image and extract list of pixels
            img_poly = cv.drawContours(img_poly, poly, -1, (255), -1)                    
            x, y = img_poly.nonzero()
            
            # adapt scaling
            # print(f"{area=}, {len(x)=}, {scaling=}")
            scaling = scaling * (1 + (1 - (len(x)/area))*multiplier)   # multiplier because shape does not scale linearly with multiplier. Therefore we have to approximate
            
            # break if size is in 2% of accept side to tackle all values being to small or to high do to accept range
            # this can still lead to shapes mean being to small, since start scaling with circle tends to be to small and therefore
            # the approximation is mostly performed from the left side
            
            if len(x) == area:
                break
            # if accept_side:
            #     # left side
            #     if 1.0 >= len(x)/area >= 0.98:
            #         self.smaller.append(len(x))
            #         break
            # else:
            #     if 1.02 >= len(x)/area >= 1.0:
            #         self.bigger.append(len(x))
            #         break    
            multiplier *= 0.99
        if i == 999:
            print(f"Not able to fit area. Place object anyways. Desired area: {area}, object area: {len(x)}")
        return x, y, shift

    def random_rotate_efd(self, efd_sample):
        rotation = 2*np.pi*np.random.random()
        rot_matrix = np.array([[np.cos(rotation), np.sin(rotation)],[-np.sin(rotation), np.cos(rotation)]])
        efd_sample = (np.array(efd_sample).reshape([-1, 2, 2]))
        efd_sample_rot = rot_matrix@efd_sample 
        efd_sample_rot = efd_sample_rot.reshape(-1, 4)
        return efd_sample_rot

    def get_random_center(self, img_size, shift):
        # -shift to account for the center of the object already moved into the positive
        return (random.randint(0 - shift, img_size[0]), random.randint(0 - shift, img_size[1]))
        #return (random.randint(0 - shift, img_size[0] - shift - 1), random.randint(0 - shift, img_size[1]- shift-  1))

    def get_random_area(self):
        if self.area[1] != 0:  # Standard deviation not 0
            area = np.random.normal(self.area[0], self.area[1])
            if area < self.area[2]:
                area = self.area[2]
            elif area > self.area[3]:
                area = self.area[3]
        else:
            area =  self.area[0]
        return np.round(area).astype("uint16")


    def check_overlap_borders(self, label, x, y):
        # check for shape outside of image
        if (x.max() >= label.shape[0]) or (y.max() >= label.shape[1]) or (x.min()<0)or (y.min()<0):
            if self.fully_visible:
                return False, (x, y)
            else:  # remove out of image pixels
                y = y[x<label.shape[0]]
                x = x[x<label.shape[0]]
                x = x[y<label.shape[1]]
                y = y[y<label.shape[1]]
                y = y[x>=0]
                x = x[x>=0]
                x = x[y>=0]
                y = y[y>=0]

        # objects can overlap by max_overlap in percent compared to the size of each object
        new_shape_unique_objects, new_shape_unique_counts = np.unique(label[x, y], return_counts=True)
        
        # no overlap
        if np.all(new_shape_unique_objects == [0]):
            return True, (x, y)
        
        # object lies completely in other objects (np.unique() returns sorted, if new_shape_unique_objects[0] != 0 => completely in other objects)
        if new_shape_unique_objects[0] != 0:
            return False, (x, y)
        
        # check if overlap for new object is too high
        overlap_px = new_shape_unique_counts[np.where(new_shape_unique_objects != 0)].sum()
        if overlap_px / len(x) > self.max_overlap:
            return False, (x, y)
        
        # check if overlap with existing objects is too high
        # get pixel counts for each existing object
        unique_objects_label, unique_counts_label = np.unique(label, return_counts=True)
        
        # check for all values except 0 whether it is less than max_overlap
        for id, object_id in enumerate(new_shape_unique_objects):
            if object_id == 0:
                continue
            # get total count for current object in existing label
            count = unique_counts_label[np.where(unique_objects_label==object_id)]
            if new_shape_unique_counts[id]/count >= self.max_overlap:
                return False, (x, y)
        return True, (x, y)
        
        
    def get_efd_descriptors(self, order, folder, ending, remove_border_elements=True):
        img_paths = self.get_img_paths(folder, ending)
        edfs = []
        for img_path in img_paths:
            img = np.array(Image.open(img_path))
            l = label(img)
            
            if remove_border_elements:
                x = np.concatenate((np.arange(0,l.shape[0]),
                                    np.arange(0,l.shape[0]),
                                    np.zeros(l.shape[1]),
                                    np.full(l.shape[1], fill_value=l.shape[0]-1)), axis=0).astype(np.uint16)
                y = np.concatenate((np.zeros(l.shape[0]),
                                    np.full(l.shape[0], fill_value=l.shape[1]-1),
                                    np.arange(0,l.shape[1]),
                                    np.arange(0,l.shape[1])), axis=0).astype(np.uint16)
                border_elements = np.unique(l[x,y])
                for element in border_elements:
                    if element == 0:
                        continue
                    l[np.where(l == element)] = 0
                
            props = regionprops(l)
            
            for prop in props:
                if prop["area"] < 2:
                    continue
                nucleus = prop["image_filled"].astype(np.uint8)*255
                # convert to rgb with cell in R and padd
                expansion = 5
                nucleus = np.pad(nucleus, ((expansion,expansion),(expansion,expansion)), "constant", constant_values=(0))
                # contour
                poly_cv, _ = cv.findContours(nucleus, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                poly_cv=poly_cv[0][:,0,:]
                poly_cv=np.flip(poly_cv, axis=1)
                
                edfs.append(elliptic_fourier_descriptors(poly_cv, order=order, normalize=True))
                
        return edfs
    
    def get_img_paths(self, folder, ending):
        if "*." in ending:
            ending = ending
        elif "." in ending:
            ending = "*"+ending
        else:
            ending = "*."+ending
        return list(Path(folder).glob(ending))

def preprocess(benchmark_dataset_path, inverse=False):
    benchmark_dataset_path = Path(benchmark_dataset_path)
    for folder in benchmark_dataset_path.glob("**"):
        if folder.name == "labels":
            print(f"Preprocessing images in : {folder}")
            folder.parent.joinpath("labels_preprocessed").mkdir(exist_ok=True)
            for path in folder.glob("*.tif"):
                img = imread(path)
                img = ((img > 0)*np.iinfo(np.uint16).max).astype("uint16")
                if inverse:
                    img = (-1*(img-np.iinfo(np.uint16).max)).astype("uint16")
                imwrite(folder.parent.joinpath("labels_preprocessed", path.name), img)
                
def preprocess_lizard(benchmark_dataset_path):
    benchmark_dataset_path = Path(benchmark_dataset_path)
    for folder in benchmark_dataset_path.glob("**"):
        if folder.name == "labels":
            print(f"Preprocessing images in : {folder}")
            folder.parent.joinpath("labels_preprocessed").mkdir(exist_ok=True)
            for path in folder.glob("*.tif"):
                img = imread(path)
                img_pre = np.zeros([3, img.shape[0], img.shape[1]]).astype("uint8")
                img_pre[0,...] = 200
                img_pre[1,...] = 180
                img_pre[2,...] = 220
                
                img_pre[0,img>0] = 100
                img_pre[1,img>0] = 70
                img_pre[2,img>0] = 150
                imwrite(folder.parent.joinpath("labels_preprocessed", path.name), img_pre)    

if __name__ == "__main__":
    train_labels_folder = "datasets/bbbc039/preprocessed/train/labels/"  
    synthetic_labels_folder = "datasets/bbbc039/synthetic/"
    all_labels_parent_folder = "datasets/bbbc039"
    inverse = False

    #preprocess_lizard(all_labels_parent_folder)
    dataset1 = Dataset(label_size=[696, 520], number_of_labels=760)

    props = extract_properties(train_labels_folder, save_labels=False, dim=None, remove_percentiles=True)
    area = (np.mean(props["area"]), np.std(props["area"]))
    num_shapes = (np.mean(props["cells_per_image"]), np.std(props["cells_per_image"]))
    
    
    efd1 = EFD(img_folder=train_labels_folder, ending="tif", order=20, area=area, random_center=True,
               fully_visible=False, max_overlap=0.05, random_rotation=True, num_shapes=num_shapes)
    dataset1.add_to_label_pipeline(efd1)
    dataset1.run_label_pipeline()
    dataset1.save_LII(synthetic_labels_folder, split=[120, 40, 360, 120, 120])
    preprocess(all_labels_parent_folder, inverse)
