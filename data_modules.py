from pytorch_lightning.core.datamodule import LightningDataModule
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import os
from torch.utils.data import DataLoader


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.TIFF', '.tif', '.TIF']


class CycleGANDataset(Dataset):
    def __init__(self, images_folder_A, images_folder_B, transforms_A=None, transforms_B=None):
        self.images_folder_A = images_folder_A
        self.images_folder_B = images_folder_B
        self.transforms_A = transforms_A
        self.transforms_B = transforms_B

        # get paths of all images
        self.paths_A = self.get_image_paths(self.images_folder_A)
        self.paths_B = self.get_image_paths(self.images_folder_B)

        # save size of A and B for __getitem__ and __len__ method
        self.len_A = len(self.paths_A)
        self.len_B = len(self.paths_B)

        print(f"Initialized dataset with {self.len_A} images in A and {self.len_B} images in B")
        
        # this needs to be done because for images B, it can happen that multiple workers access the same file
        # https://stackoverflow.com/a/73289157
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')

    def __len__(self):
        return self.len_A

    def __getitem__(self, idx):
        try:
            A = Image.open(self.paths_A[idx])
        except:
            print(f"Not able to load {self.paths_A[idx]}")
        A = np.asarray(A).astype(np.float32)
        idx_B = np.random.randint(0, self.len_B-1)  # get random image to avoid fixed pairs
        try:
            B = Image.open(self.paths_B[idx_B])
        except:
            print(f"Not able to load {self.paths_B[idx]}")
        B = np.asarray(B).astype(np.float32)

        if self.transforms_A:
            A = self.transforms_A(image=A)["image"]
        if self.transforms_B:
            B = self.transforms_B(image=B)["image"]
        return {"A": A, "B": B, "A_stem": self.paths_A[idx].stem, "B_stem": self.paths_B[idx_B].stem}  # stem is only needed for inference

    def get_image_paths(self, folder):
        image_paths = []
        for (root, _, filenames) in os.walk(folder):
            for filename in filenames:
                if any(filename.endswith(extension) for extension in IMG_EXTENSIONS):
                    image_paths.append(Path(os.path.join(root, filename)))
        return image_paths


class ToTensor3D(ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0):
        super(ToTensor3D, self).__init__(always_apply, p)

    def apply(self, image, **params):
        img = torch.from_numpy(image)
        return img


class MinMaxRangeNormalize(ImageOnlyTransform):
    """Normalization is applied by the formula: `img = (max-min)/max_pixel_value*img+min`
    Standard parameters convert an image in range [0,255] to range [0,1]
    Args:
        min (float, list of float): min values of output image
        max  (float, list of float): max values of output image
        max_pixel_value (float): maximum possible pixel value of input image
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, min=0, max=1, max_pixel_value=255.0, always_apply=False, p=1.0):
        super(MinMaxRangeNormalize, self).__init__(always_apply, p)
        self.min = np.array(min, dtype=np.float32)
        self.max = np.array(max, dtype=np.float32)
        self.max_pixel_value = max_pixel_value
        # min_pixel_value is allways 0 for normal images not needed here
        # for fast calculations calculate a for a*img+min=(max-min)/max_pixel_value*img+min once
        self.a = (self.max-self.min)/self.max_pixel_value

    def apply(self, image, **params):
        img = self.a*image+self.min
        return img

    def get_transform_init_args_names(self):
        return ("min", "max", "max_pixel_value")


class GrayToRGB(ImageOnlyTransform):
    """ Expands the grayscale dimensions to RGB
    """

    def __init__(self, always_apply=False, p=1.0):
        super(GrayToRGB, self).__init__(always_apply, p)

    def apply(self, image, **params):
        img = np.concatenate([image[..., np.newaxis]]*3, axis=-1)
        return img

    def get_transform_init_args_names(self):
        return ("")


class CycleGANDataModule(LightningDataModule):
    def __init__(self, batch_size, batch_size_val, num_workers, image_folder_train_A, image_folder_train_B,
                 image_folder_val_A, image_folder_val_B, image_folder_test_A, image_folder_test_B,
                 transforms_train_A, transforms_train_B, transforms_val_A, transforms_val_B,
                 transforms_test_A, transforms_test_B, min_A, max_A, min_B, max_B):
        """ Generate LightningDataModule that prepares the dataloaders

        Args:
            batch_size (int): Batch size used for training
            batch_size_val (int): Batch size used for evaluation
            num_workers (int): Number of workers used for training and evaluation
            image_folder_train_A (Path): Pathlib Path to image folder train A
            image_folder_train_B (Path): Pathlib Path to image folder train B
            image_folder_val_A (Path): Pathlib Path to image folder val A
            image_folder_val_B (Path): Pathlib Path to image folder val B
            image_folder_test_A (Path): Pathlib Path to image folder test A
            image_folder_test_B (Path): Pathlib Path to image folder test B
            transforms_train_A (List): List of albumentations transformations used for training data A
            transforms_train_B (List): List of albumentations transformations used for training data B
            transforms_val_A (List): List of albumentations transformations used for val data A
            transforms_val_B (List): List of albumentations transformations used for val data B
            transforms_test_A (List): List of albumentations transformations used for test data A
            transforms_test_B (List): List of albumentations transformations used for test data B
        """
        super().__init__()
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.num_workers = num_workers
        self.image_folder_train_A = image_folder_train_A
        self.image_folder_train_B = image_folder_train_B
        self.image_folder_val_A = image_folder_val_A
        self.image_folder_val_B = image_folder_val_B
        self.image_folder_test_A = image_folder_test_A
        self.image_folder_test_B = image_folder_test_B
        self.transforms_train_A = transforms_train_A
        self.transforms_train_B = transforms_train_B
        self.transforms_val_A = transforms_val_A
        self.transforms_val_B = transforms_val_B
        self.transforms_test_A = transforms_test_A
        self.transforms_test_B = transforms_test_B
        self.min_A = min_A
        self.max_A = max_A
        self.min_B = min_B
        self.max_B = max_B

    def train_dataloader(self):
        train_dataset = CycleGANDataset(images_folder_A=self.image_folder_train_A,
                                        images_folder_B=self.image_folder_train_B,
                                        transforms_A=A.Compose(self.transforms_train_A),
                                        transforms_B=A.Compose(self.transforms_train_B))
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        val_dataset = CycleGANDataset(images_folder_A=self.image_folder_val_A,
                                      images_folder_B=self.image_folder_val_B,
                                      transforms_A=A.Compose(self.transforms_val_A),
                                      transforms_B=A.Compose(self.transforms_val_B))
        return DataLoader(val_dataset, batch_size=self.batch_size_val, num_workers=self.num_workers)   

    def test_dataloader(self):
        test_dataset = CycleGANDataset(images_folder_A=self.image_folder_test_A,
                                       images_folder_B=self.image_folder_test_B,
                                       transforms_A=A.Compose(self.transforms_test_A),
                                       transforms_B=A.Compose(self.transforms_test_B))
        return DataLoader(test_dataset, batch_size=1, num_workers=self.num_workers)

def get_data_module(args):   
    # expand crop_size if only one value is given. 3rd value not needed for 2D data
    crop_size = args.crop_size
    if crop_size == None:
        crop_size = [None, None, None]
    if len(crop_size) == 1:
        crop_size = [crop_size[0], crop_size[0], crop_size[0]]
    
    min_A = args.minmax_scale_A[0]
    max_A = args.minmax_scale_A[1]
    min_B = args.minmax_scale_B[0]
    max_B = args.minmax_scale_B[1]
    
    if "bbbc039" == args.data_module:
        transforms_train_A = [A.RandomCrop(crop_size[0], crop_size[1], p=1),
                              A.ToFloat(max_value=np.iinfo(np.uint16).max),  # output is in range [0,1]
                              A.GaussNoise(var_limit=(0.1, 0.1), p=1),
                              MinMaxRangeNormalize(min=min_A, max=max_A, max_pixel_value=1),
                              ToTensorV2()]
        transforms_train_B = [A.RandomCrop(crop_size[0], crop_size[1], p=1),
                              A.ToFloat(max_value=np.iinfo(np.uint16).max),  # output is in range [0,1]
                              MinMaxRangeNormalize(min=min_B, max=max_B, max_pixel_value=1),
                              ToTensorV2()]
        transforms_val_A = transforms_train_A
        transforms_val_B = transforms_train_B
        transforms_test_A = [A.ToFloat(max_value=np.iinfo(np.uint16).max),  # output is in range [0,1]
                             A.GaussNoise(var_limit=(0.1, 0.1), p=1),
                             MinMaxRangeNormalize(min=min_A, max=max_A, max_pixel_value=1),
                             ToTensorV2()]
        transforms_test_B = [A.ToFloat(max_value=np.iinfo(np.uint16).max),  # output is in range [0,1]
                             MinMaxRangeNormalize(min=min_B, max=max_B, max_pixel_value=1),
                             ToTensorV2()]
        image_folder_train_A = Path(args.path_train_A)
        image_folder_train_B = Path(args.path_train_B)
        image_folder_val_A = Path(args.path_val_A)
        image_folder_val_B = Path(args.path_val_B)
        image_folder_test_A = Path(args.path_test_A)
        image_folder_test_B = Path(args.path_test_A)
        return CycleGANDataModule(args.batch_size, args.batch_size_val, args.num_workers, image_folder_train_A, image_folder_train_B,
                                  image_folder_val_A, image_folder_val_B, image_folder_test_A, image_folder_test_B, transforms_train_A,
                                  transforms_train_B, transforms_val_A, transforms_val_B, transforms_test_A, transforms_test_B,
                                  min_A, max_A, min_B, max_B)
    if "lizard" == args.data_module:
        transforms_train_A = [A.RandomCrop(crop_size[0], crop_size[1], p=1),
                              A.ToFloat(max_value=np.iinfo(np.uint8).max),  # output is in range [0,1]
                              A.GaussNoise(var_limit=(0.01, 0.01), p=1),
                              MinMaxRangeNormalize(min=min_A, max=max_A, max_pixel_value=1),
                              ToTensorV2()]
        transforms_train_B = [A.RandomCrop(crop_size[0], crop_size[1], p=1),
                              A.ToFloat(max_value=np.iinfo(np.uint8).max),  # output is in range [0,1]
                              MinMaxRangeNormalize(min=min_B, max=max_B, max_pixel_value=1),
                              ToTensorV2()]
        transforms_val_A = transforms_train_A
        transforms_val_B = transforms_train_B
        transforms_test_A = [A.ToFloat(max_value=np.iinfo(np.uint8).max),  # output is in range [0,1]
                             A.GaussNoise(var_limit=(0.01, 0.01), p=1),
                             MinMaxRangeNormalize(min=min_A, max=max_A, max_pixel_value=1),
                             ToTensorV2()]
        transforms_test_B = [A.ToFloat(max_value=np.iinfo(np.uint8).max),  # output is in range [0,1]
                             MinMaxRangeNormalize(min=min_B, max=max_B, max_pixel_value=1),
                             ToTensorV2()]
        image_folder_train_A = Path(args.path_train_A)
        image_folder_train_B = Path(args.path_train_B)
        image_folder_val_A = Path(args.path_val_A)
        image_folder_val_B = Path(args.path_val_B)
        image_folder_test_A = Path(args.path_test_A)
        image_folder_test_B = Path(args.path_test_A)
        return CycleGANDataModule(args.batch_size, args.batch_size_val, args.num_workers, image_folder_train_A, image_folder_train_B,
                                  image_folder_val_A, image_folder_val_B, image_folder_test_A, image_folder_test_B, transforms_train_A,
                                  transforms_train_B, transforms_val_A, transforms_val_B, transforms_test_A, transforms_test_B,
                                  min_A, max_A, min_B, max_B)
    else:
        raise ValueError(f'{args.data_module} not implemented')
