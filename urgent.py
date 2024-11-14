# Library Imports (General)
import pathlib
import os
import sys
import gc
import requests
import math
import random
import pickle
import argparse
import ipywidgets
import numpy as np
import pandas as pd

# Library Imports (Modelling)
import pydicom
import torch
import scipy

# Library Imports (Monitoring)
import matplotlib
import matplotlib.pyplot as plt
import time
import timeit
import warnings
import tqdm
import torchvision

# Function Imports (General)
from pathlib import Path
from ipywidgets import interactive, IntSlider

# Function Imports (Modelling)
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, utils
from skvideo import io

# Function Imports (Monitoring)
from PIL import Image, ImageSequence

# Non-Conditional MetaBrest Dataset Reader Class (V4)
class NCDataset(Dataset):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser,
        dataset: str = 'private',
        mode: str = 'train',
    ):  
        
        # Dataset Choice
        super(NCDataset).__init__(); self.settings = settings
        self.mode = mode; self.dataset = dataset
        if self.dataset == 'public': self.data_folderpath = self.settings.public_data_folderpath
        elif self.dataset == 'lung': self.data_folderpath = self.settings.lung_data_folderpath
        elif self.dataset == 'private': self.data_folderpath = self.settings.private_data_folderpath
        else: print("ERROR: Chosen Dataset / Directory does not exist!")
        
        # Subject Indexing (Existing or New Version)
        subj_listpath = Path(f"{self.settings.reader_folderpath}/V{self.settings.data_version}" +\
                             f"/{self.dataset}_{self.mode}_setV{self.settings.data_version}.txt")
        #print(subj_listpath)
        if subj_listpath.exists():
            #print(f"Reading {self.dataset} Dataset Save Files for {self.mode} Set | Version {settings.data_version}")
            self.subj_list = subj_listpath.read_text().splitlines()
        else:
            print(f"Generating New Save Files for {self.dataset} Dataset | Version {settings.data_version}")
            self.subj_list = os.listdir(self.data_folderpath)       # Complete List of Subjects in Dataset
            self.subj_list.remove('video_data')
            self.subj_list = self.subj_split(self.subj_list)        # Selected List of Subjects in Dataset
        #assert len(self.subj_list) == self.num_subj, f"WARNING: Number of subjs does not match Dataset Version!"

        # --------------------------------------------------------------------------------------------
        
        # Dataset Transformations Initialization
        self.transform = transforms.Compose([
                                        transforms.Resize(( self.settings.img_size,
                                                            self.settings.img_size)),
                                        transforms.ToTensor()])
        self.h_flip = transforms.Compose([transforms.RandomHorizontalFlip(p = 1)])
        self.v_flip = transforms.Compose([transforms.RandomVerticalFlip(p = 1)])

    # ============================================================================================

    # DataLoader Length / No. Subjects Computation Functionality
    def __len__(self): return len(self.subj_list)
    
    # Subject Splitting Functionality
    def subj_split(self, subj_list: list):

        # Dataset Splitting
        train_subj = len(subj_list) - self.settings.val_subj - self.settings.test_subj if self.settings.train_subj == 0 else self.settings.train_subj
        assert 0 < (train_subj + self.settings.val_subj + self.settings.test_subj) <= len(subj_list),\
               f"ERROR: Dataset does not contain {train_subj + self.settings.val_subj + self.settings.test_subj} Subjects!"
        if self.settings.val_subj != 0:
            val_subj = np.sort(np.array(random.sample(subj_list, self.settings.val_subj), dtype = 'str'))
            subj_list = [subj for subj in subj_list if subj not in val_subj]                                    # Validation Set Splitting
        if self.settings.test_subj != 0:
            test_subj = np.sort(np.array(random.sample(subj_list, self.settings.test_subj), dtype = 'str'))
            subj_list = [subj for subj in subj_list if subj not in test_subj]                                   # Test Set Splitting
        train_subj = np.sort(np.array(random.sample(subj_list, train_subj), dtype = 'str'))
        subj_list = [subj for subj in subj_list if subj not in train_subj]                                      # Training Set Splitting
        subj_list = np.sort(np.array(subj_list, dtype = 'str'))
        assert len(subj_list) + len(train_subj) + self.settings.val_subj + self.settings.test_subj == len(self.subj_list),\
               f"ERROR: Dataset Splitting went Wrong!"

        # Dataset Split Saving
        if not os.path.isdir(f"V{self.settings.data_version}"): os.mkdir(f"V{self.settings.data_version}")
        if len(train_subj) != 0: np.savetxt(f"{self.settings.reader_folderpath}/V{self.settings.data_version}/{self.dataset}_train_setV{self.settings.data_version}.txt", train_subj, fmt='%s')
        if len(subj_list) != 0: np.savetxt(f"{self.settings.reader_folderpath}/V{self.settings.data_version}/{self.dataset}_rest_setV{self.settings.data_version}.txt", subj_list, fmt='%s')
        if self.settings.test_subj != 0:
            if len(test_subj) != 0: np.savetxt(f"{self.settings.reader_folderpath}/V{self.settings.data_version}/{self.dataset}_test_setV{self.settings.data_version}.txt", test_subj, fmt='%s')
        if self.settings.val_subj != 0:
            if len(val_subj) != 0: np.savetxt(f"{self.settings.reader_folderpath}/V{self.settings.data_version}/{self.dataset}_val_setV{self.settings.data_version}.txt", val_subj, fmt='%s')
 
    # ============================================================================================
        
    # Single Batch / Subject Generation Functionality
    def __getitem__(self, idx: int = 0 or str, save: bool = False):
        subj_idx = idx if type(idx) == str else self.subj_list[idx]

        # MP4 Subject File Reading
        if self.settings.data_format == 'mp4':

            # Subject Data Access
            subj_folderpath = f"{self.data_folderpath}/video_data/V{self.settings.data_version}/{self.mode}/{subj_idx}.mp4"
            img_data = (torchvision.io.read_video(subj_folderpath, pts_unit = 'sec')[0][:, :, :, 0] / 255.0).type(torch.float32)

        # DICOM Subject File Reading
        elif self.settings.data_format == 'dicom':

            # Subject Folder Access
            subj_folderpath = f"{self.data_folderpath}/{subj_idx}"
            subj_filelist = os.listdir(subj_folderpath)
            for i, path in enumerate(subj_filelist):
                subj_folderpath = f"{self.data_folderpath}/{subj_idx}/{path}"
                subj_filelist = os.listdir(subj_folderpath)
                while os.path.splitext(subj_filelist[0])[1] not in ['.dcm', '.xlm']:
                    subj_folderpath = Path(f"{subj_folderpath}/{subj_filelist[0]}")
                    subj_filelist = os.listdir(subj_folderpath)
                if len(subj_filelist) >= 50: break
            subj_filelist = np.ndarray.tolist(np.sort(subj_filelist))
            
            # Subject General Information Access
            subj_filepath = Path((f"{subj_folderpath}/{subj_filelist[0]}"))
            while os.path.splitext(subj_filepath)[1] not in ['', '.dcm']:
                i += 1; subj_filepath = Path((f"{subj_folderpath}/{subj_filelist[i]}"))
            subj_info = pydicom.dcmread(subj_filepath, force = True)
            og_idx = int(subj_info[0x0020, 0x0013].value)
            subj_ori = subj_info[0x0020, 0x0037].value
            subj_v_flip = (np.all(subj_ori == [-1, 0, 0, 0, -1, 0]))
            subj_h_flip = (torch.rand(1) < (self.settings.h_flip / 100))

            # --------------------------------------------------------------------------------------------
                
            # Subject Slice Data Access
            og_frame = len(subj_filelist) + og_idx + 50 if self.dataset == 'lung' else 100
            img_data = torch.empty((og_frame, self.settings.img_size, self.settings.img_size)); slice_list = []
            for i, slice_filepath in enumerate(np.sort(subj_filelist)):
                if os.path.splitext(slice_filepath)[1] in ['', '.dcm']:
                    
                    # Slice Data Access
                    slice_filepath = Path(f"{subj_folderpath}/{slice_filepath}")
                    slice_data = pydicom.dcmread(slice_filepath, force = True)
                    slice_idx = int(slice_data[0x0020, 0x0013].value)
                    if slice_idx >= len(subj_filelist) + 1: slice_idx -= og_idx 
                    slice_list.append(slice_idx)
                    img_slice = slice_data.pixel_array.astype(float)

                    # Slice Image Pre-Processing | Rescaling, Resizing & Flipping
                    if self.settings.data_prep:
                        img_slice = np.uint8((np.maximum(img_slice, 0) / img_slice.max()) * 255)
                        img_slice = Image.fromarray(img_slice).resize(( self.settings.img_size,
                                                                        self.settings.img_size)) 
                        if subj_h_flip: img_slice = self.h_flip(img_slice)
                        if subj_v_flip: img_slice = self.v_flip(img_slice)
                        img_slice = np.array(self.transform(img_slice))
                    img_data[slice_idx, :, :] = torch.Tensor(img_slice); del img_slice
                else: subj_filelist.remove(slice_filepath)
            print(f"Accessing Subject {subj_idx}: {len(subj_filelist)} -> {self.settings.num_slice} Slices")
            img_data = img_data[np.sort(slice_list)]

            # --------------------------------------------------------------------------------------------
            
            # Slice Cropping | Spaced-Out Slices
            if self.settings.slice_spacing:
                s_array = slice_array = np.linspace(0 + self.settings.slice_bottom_margin,
                    len(subj_filelist) - self.settings.slice_top_margin - 1, self.settings.num_slice)
                slice_array[0 : int(np.floor(self.settings.num_slice / 2))] = np.ceil(s_array[0 : int(np.floor(self.settings.num_slice / 2))]).astype(int)
                slice_array[int(np.ceil(self.settings.num_slice / 2) + 1)::] = np.floor(s_array[int(np.ceil(self.settings.num_slice / 2) + 1)::]).astype(int)
                slice_array[int(np.floor(self.settings.num_slice / 2))] = np.round(s_array[int(np.floor(self.settings.num_slice / 2))])
                img_data = img_data[slice_array, :, :]

            # Slice Cropping | Middle Slices Only
            else:
                extra_slice = self.settings.num_slice - img_data.shape[0]
                if img_data.shape[0] < self.settings.num_slice:             # Addition of Repeated Peripheral Slices
                    for extra in range(extra_slice):
                        if extra % 2 == 0: img_data = torch.cat((img_data, img_data[-1].unsqueeze(0)), dim = 0)
                        else: img_data = torch.cat((img_data[0].unsqueeze(0), img_data), dim = 0)
                elif img_data.shape[0] > self.settings.num_slice:           # Removal of Emptier Peripheral Slices
                    img_data = img_data[int(np.ceil(-extra_slice / 2)) :\
                        int(len(img_data) - np.floor(-extra_slice / 2))]
            assert(img_data.shape[0] == self.settings.num_slice)
          
            # Item Dictionary Returning
            if save:
                print(f"Saving Patient Data for {subj_idx} into Video Format")
                if not os.path.isdir(f"{self.data_folderpath}/video_data/V{self.settings.data_version}"):
                    os.mkdir(f"{self.data_folderpath}/video_data/V{self.settings.data_version}")
                if not os.path.isdir(f"{self.data_folderpath}/video_data/V{self.settings.data_version}/{self.mode}"):
                    os.mkdir(f"{self.data_folderpath}/video_data/V{self.settings.data_version}/{self.mode}")
                torchvision.io.write_video(f"{self.data_folderpath}/video_data/V{self.settings.data_version}/{self.mode}/{subj_idx}.mp4",
                    (img_data.unsqueeze(3).repeat(1, 1, 1, 3) * 255).type(torch.uint8), fps = self.settings.num_fps)
        else: raise(NotImplementedError)
        return img_data.unsqueeze(0)
   

# Non-Conditional 3D Diffusion Model Parser Initialization
if True:
    ncdiff_parser = argparse.ArgumentParser(
        description = "Non-Conditional 3D Diffusion Model")
    ncdiff_parser.add_argument('--model_type', type = str,            # Chosen Model / Diffusion
                                choices =  {'video_diffusion',
                                            'blackout_diffusion',
                                            'gamma_diffusion'},
                                default = 'video_diffusion')
    ncdiff_parser.add_argument('--model_version', type = int,         # Model Version Index
                                default = 0)
    ncdiff_parser.add_argument('--data_version', type = int,          # Dataset Version Index
                                default = 2)
    ncdiff_parser.add_argument('--noise_type', type = str,            # Diffusion Noise Distribution
                                default = 'gaussian')
    settings = ncdiff_parser.parse_args("")

    # ============================================================================================

    # Directories and Path Arguments
    ncdiff_parser.add_argument('--reader_folderpath', type = str,         # Path for Dataset Reader Directory
                                default = '../../MetaBreast/data/non_cond')
                                #default = "X:/nas-ctm01/homes/pfsousa/MetaBreast/data/non_cond")
    ncdiff_parser.add_argument('--public_data_folderpath', type = str,    # Path for Private Dataset Directory
                                #default = "X:/nas-ctm01/datasets/public/MEDICAL/Duke-Breast-Cancer-T1")
                                default = "../../../../datasets/public/MEDICAL/Duke-Breast-Cancer-T1")
    ncdiff_parser.add_argument('--private_data_folderpath', type = str,   # Path for Private Dataset Directory
                                #default = "X:/nas-ctm01/datasets/private/METABREST/T1W_Breast")
                                default = '../../../../datasets/private/METABREST/T1W_Breast')
    ncdiff_parser.add_argument( '--lung_data_folderpath', type = str,     # Path for LUCAS Dataset Directory
                                #default = "X:/nas-ctm01/datasets/private/LUCAS/lidc/TCIA_LIDC-IDRI_20200921/LIDC-IDRI")
                                default = "../../../../datasets/private/LUCAS/lidc/TCIA_LIDC-IDRI_20200921/LIDC-IDRI")

    # Directory | Model-Related Path Arguments
    ncdiff_parser.add_argument('--model_folderpath', type = str,          # Path for Model Architecture Directory
                                default = f'../../models/{settings.model_type}')
    ncdiff_parser.add_argument('--script_folderpath', type = str,         # Path for Model Training & Testing Scripts Directory
                                default = f'../../scripts/{settings.model_type}')
    ncdiff_parser.add_argument('--logs_folderpath', type = str,           # Path for Model Saving Directory
                                default = f'../../logs/{settings.model_type}')
        
    # ============================================================================================

    # Dataset | Dataset General Arguments
    ncdiff_parser.add_argument('--data_format', type = str,           # Chosen Dataset Format for Reading
                                choices =  {'mp4', 'dicom'},
                                default = 'dicom')
    ncdiff_parser.add_argument('--img_size', type = int,              # Generated Image Resolution
                                default = 64)
    ncdiff_parser.add_argument('--num_slice', type = int,             # Number of 2D Slices in MRI
                                default = 30)
    ncdiff_parser.add_argument('--slice_spacing', type = bool,        # Usage of Linspace for Slice Spacing
                                default = True)
    ncdiff_parser.add_argument('--slice_bottom_margin', type = int,   # Number of 2D Slices to be Discarded in Bottom Margin
                                default = 5)
    ncdiff_parser.add_argument('--slice_top_margin', type = int,      # Number of 2D Slices to be Discarded in Top Margin
                                default = 15)
    ncdiff_parser.add_argument('--data_prep', type = bool,            # Usage of Dataset Pre-Processing Control Value
                                default = True)
    ncdiff_parser.add_argument('--h_flip', type = int,                # Percentage of Horizontally Flipped Subjects
                                default = 50)

    # Dataset | Dataset Splitting Arguments
    ncdiff_parser.add_argument('--train_subj', type = int,            # Number of Random Subjects in Training Set
                                default = 0)                          # PS: Input 0 for all Subjects in the Dataset
    ncdiff_parser.add_argument('--val_subj', type = int,              # Number of Random Subjects in Validation Set
                                default = 0)
    ncdiff_parser.add_argument('--test_subj', type = int,             # Number of Random Subjects in Test Set
                                default = 0)

    # Dataset | DataLoader Arguments
    ncdiff_parser.add_argument('--batch_size', type = int,            # DataLoader Batch Size Value
                                default = 1)
    ncdiff_parser.add_argument('--shuffle', type = bool,              # DataLoader Subject Shuffling Control Value
                                default = False)
    ncdiff_parser.add_argument('--num_workers', type = int,           # Number of DataLoader Workers
                                default = 8)
    ncdiff_parser.add_argument('--num_fps', type = int,               # Number of Video Frames per Second
                                default = 4)

    # ============================================================================================

    # Model | Architecture-Defining Arguments
    ncdiff_parser.add_argument('--seed', type = int,                  # Randomised Generational Seed
                                default = 0)
    ncdiff_parser.add_argument('--dim', type = int,                   # Input Dimensionality (Not Necessary)
                                default = 64)
    ncdiff_parser.add_argument('--num_channel', type = int,           # Number of Input Channels for Dataset
                                default = 1)
    ncdiff_parser.add_argument('--mult_dim', type = tuple,            # Dimensionality for all Conditional Layers
                                default = (1, 2, 4, 8))

    # Model | Training & Diffusion Arguments
    #ncdiff_parser.add_argument('--num_epochs', type = int,            # Number of Training Epochs
    #                            default = 30)
    ncdiff_parser.add_argument('--num_ts', type = int,                # Number of Scheduler Timesteps
                                default = 300)
    ncdiff_parser.add_argument('--num_steps', type = int,             # Number of Diffusion Training Steps
                                default = 500000)
    ncdiff_parser.add_argument('--lr_base', type = float,             # Base Learning Rate Value
                                default = 1e-4)
    ncdiff_parser.add_argument('--save_interval', type = int,         # Number of Training Step Interval inbetween Image Saving
                                default = 1000)
    ncdiff_parser.add_argument('--save_img', type = int,              # Square Root of Number of Images Saved for Manual Evaluation
                                default = 2)

    # ============================================================================================

    settings = ncdiff_parser.parse_args("")
    settings.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


# Checkerboard / Salt & Pepper Noise Addition (VideoDiffusion)
def sp_noise(img, sp = [10, 10]):
    row , col = img.shape
    img_sp = img.clone() + 0.1255

    # https://www.geeksforgeeks.org/add-a-salt-and-pepper-noise-to-an-image-with-python/
      
    # Black Pixel Coloring
    #number_of_pixels = random.randint(int(sp[0] / 100 * row * col), int(sp[1] / 100 * row * col))
    #for i in range(number_of_pixels):
    #    y_coord=random.randint(0, row - 1)
    #    x_coord=random.randint(0, col - 1)
    #    img_sp[y_coord][x_coord] = 1
          
    # White Pixel Coloring
    number_of_pixels = random.randint(int(sp[0] / 100 * row * col), int(sp[1] / 100 * row * col))
    for i in range(number_of_pixels):
        y_coord=random.randint(0, row - 1)
        x_coord=random.randint(0, col - 1)
        img_sp[y_coord][x_coord] = 0
    
    img_sp = (img_sp - img_sp.min()) / img_sp.max()
    return img_sp

# Gaussian Noise Addition (MedicalDiffusion)
def g_noise(img, factor = 40):

    row , col = img.shape
    img_g = img.clone() + 0.062
    #print(img_g.min()); print(img_g.max())
    noise = ((torch.randn((img_g.shape[0], img_g.shape[1])) * 1. + 0.) / np.pi + 1.) / 2.
    noise = ((noise / np.pi + 1.) / 2.) * (factor / 100)
    #print(noise); print(noise.min()); print(noise.max())
    img_g = img_g + noise
    #print(img_g.min()); print(img_g.max())
    img_g = (img_g - img_g.min()) / (img_g.max() - img_g.min())
    return img_g

"""
print("start")
dataset = NCDataset(settings,
                    mode = "train",
                    dataset = 'public')
print("dataset")
data = dataset.__getitem__(8)
print("data read")
data_sp = torch.empty_like(data)
data_g = torch.empty_like(data)
for slice in range(data.shape[1]):
    data_sp[0, slice, :, :] = sp_noise(data[0, slice, :, :])
    data_g[0, slice, :, :] = g_noise(data[0, slice, :, :])

io.vwrite("sample_10.gif", data[0])
io.vwrite("sample_10_sp.gif", data_sp[0])
io.vwrite("sample_10_gaussian.gif", data_g[0])
"""
"""
filepath = "../../MetaBreast/logs/video_diffusion/V2/sample/sample_10.gif"
img = Image.open(filepath); sample_img1 = torch.empty((1, 30, 64, 64))
img = [frame.convert('L') for frame in ImageSequence.Iterator(img)]
for i in range(sample_img1.shape[1]):
    sample_img1[0, i, :, :] = torch.Tensor(np.array(img[i])) / 255.0

filepath = "../../MedDiff/evaluation/V1/sampling_0/sample_0.pt"
sample_img2 = torch.load(filepath, map_location = torch.device('cpu'))
#sample_img2_norm = (sample_img2 - sample_img2.min()) / (sample_img2.max() - sample_img2.min())
sample_img2 = sample_img2 - sample_img2.min(); sample_img2 /= sample_img2.max()
io.vwrite("sample_meddiff.gif", sample_img2)
io.vwrite("sample_videodiff.gif", sample_img1)
"""

"""
print("start")
filepath_sr = "../../SuperResolution/super_resolution/Real-ESRGAN/experiments/videodiff_V2/sample_3_sr.pt"
img_sr = torch.load(filepath_sr)
print("#1")
filepath = "../../MetaBreast/logs/video_diffusion/V2/sample/sample_3.gif"
img = Image.open(filepath); sample_img = torch.empty((1, 30, 64, 64))
img = [frame.convert('L') for frame in ImageSequence.Iterator(img)]
print("#2")
for i in range(30):
    sample_img[0, i, :, :] = torch.Tensor(np.array(img[i]))
print(sample_img.shape)
io.vwrite("sample_vid.gif", sample_img)
io.vwrite("sample_vid_sr.gif", img_sr)
"""

mcvd1 = "../../MCVD/mcvd_V4/logs/samples/samples_260000.pt"
mcvd1 = torch.load(mcvd1).reshape((1, 1, 128, 30, 128)).permute((0, 1, 3, 2, 4)) * 255.
mcvd2 = "../../MCVD/mcvd_V4/logs/samples/samples_265000.pt"
mcvd2 = torch.load(mcvd2).reshape((1, 1, 128, 30, 128)).permute((0, 1, 3, 2, 4)) * 255.
meddiff1 = "../../MedDiff/evaluation/V1/sampling_0/sample_12.pt"
meddiff1 = torch.load(meddiff1, map_location=torch.device('cpu')) * 255.
meddiff2 = "../../MedDiff/evaluation/V1/sampling_0/sample_4.pt"
meddiff2 = torch.load(meddiff2, map_location=torch.device('cpu')) * 255.
io.vwrite("mcvd_260000.gif", mcvd1[0, 0])
io.vwrite("mcvd_265000.gif", mcvd2[0, 0])
io.vwrite("meddiff_12.gif", meddiff1[0, 0])
io.vwrite("meddiff_4.gif", meddiff2[0, 0])
