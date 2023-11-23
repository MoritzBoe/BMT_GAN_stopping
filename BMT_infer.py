import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import torch
from PIL import Image
from pytorch_lightning import seed_everything

from lit_cyclegan import LitCycleGAN
from data_modules import get_data_module


def infer(model, device, test_img_folder, results_folder, result_dtype, args):
    """Infer using Stitching Aware Inference with overlapping patches from both domains.
    Can be used for simple tiling, when the overlap is set to zero.

    Args:
        model (LitCycleGAN): Trained LitCycleGAN model
        device (torch.device): Device (GPU or CPU). GPU is highly recommended.
        test_img_folder (Path): Path to the images used for prediction.
        results_folder (Path): Path to the results folder.
        result_dtype (str): Data type of results.
        args (Namespace): Dataset args (data_module, num_channels_A, num_channels_B)

    """
    print(f"Starting prediction")
    args.batch_size = 1
    args.batch_size_val = 1
    args.num_workers = 0
    args.crop_size = None
    args.minmax_scale_A = [-1,1]
    args.minmax_scale_B = [-1,1]
    # initialize paths for datamodule, while only path_test_A is used in inference
    args.path_train_A = test_img_folder
    args.path_train_B = test_img_folder
    args.path_val_A = test_img_folder
    args.path_val_B = test_img_folder
    args.path_test_A = test_img_folder
    args.path_test_B = test_img_folder
    args.result_dtype = result_dtype
    # seed to get same results for every run, since mandatory noise is added in augmentations
    # Alternatively, apply noise manually to the label images and remove it from augmentations
    seed_everything(1, workers=True)
    
    data_module = get_data_module(args)
    dataloader = data_module.test_dataloader()

    # get scaling
    scaling_B = [(model.hparams.minmax_scale_B[1]-model.hparams.minmax_scale_B[0])/2,
                 model.hparams.minmax_scale_B[0]-(-1)*((model.hparams.minmax_scale_B[1]-model.hparams.minmax_scale_B[0])/2)]
    model.gen_AB.eval()

    num_images = len(dataloader)
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            print(f"Processing image {idx+1} of {num_images}")
            batch["A"] = batch["A"].to(device)
           
            # infer and scale
            pred = model.gen_AB(batch["A"])*scaling_B[0]+scaling_B[1]
            pred_path = results_folder.joinpath(batch["A_stem"][0]+".tif")
            save_img(pred, model.hparams.minmax_scale_B[0], model.hparams.minmax_scale_B[1], pred_path, args.result_dtype)

def save_img(img, scale_min, scale_max, save_path, result_dtype):
    # convert img to numpy
    if torch.is_tensor(img):
        img = img.cpu().numpy()
    img = img.squeeze() 
    # move color channel to the back if necessary. Fails when img.shape = (3,X,3) and the last channel already is the color channel!!
    if img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.moveaxis(img, 0, -1)
    # scale image and convert to uint8
    img = np.round((img-scale_min)*np.iinfo(getattr(np, result_dtype)).max/(scale_max-scale_min)).astype(result_dtype)
    Image.fromarray(img).save(save_path)



def batch_infer(args):
    checkpoints = Path(args.run_folder).glob("*.ckpt")
    test_img_folders = list(Path(args.parent_test_images_A).glob("**/labels_preprocessed"))
    
    for checkpoint in checkpoints:
        model = LitCycleGAN.load_from_checkpoint(checkpoint)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        for test_img_folder in test_img_folders:
            results_folder = test_img_folder.parent.joinpath("samples", checkpoint.parts[-2],checkpoint.stem)
            results_folder.mkdir(exist_ok=True, parents=True)
            infer(model, device, test_img_folder, results_folder, args.result_dtype, args)

    print(list(checkpoints))





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--run_folder', '-rf', default="", type=str, help="Local run folder")
    parser.add_argument('--parent_test_images_A', default="datasets/bbbc039/synthetic/downstream/", type=str, help="Path to test images from domain A, when using the bbbc039 data module.")
    parser.add_argument('--num_channels_A', type=int, default=1, help="Number of channels of images in domain A.")
    parser.add_argument('--num_channels_B', type=int, default=1, help="Number of channels of images in domain B.")
    parser.add_argument('--data_module', default="bbbc039", type=str, help="Name of the datamodule used.")
    parser.add_argument('--result_dtype', default="uint16", type=str, help="Dtype of image saved.")
    args = parser.parse_args()
    batch_infer(args)