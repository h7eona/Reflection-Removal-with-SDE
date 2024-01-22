import numpy as np
import os
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import random
import torchvision.transforms.functional as TF
import cv2

import glob

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.
    return img

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

def ShiftData(image, shift_range, width):
    shift_image = image.clone()

    to_shift = image[ :, :, 0:shift_range]
    rest = image[:,  :, shift_range:width]

    shift_image[:, :, 0:(width-shift_range)] = rest
    shift_image[:, :, (width-shift_range):width] = to_shift

    return shift_image

class PanoramaDataLoader(Dataset):
    def __init__(self, mode, data_dir, opt):
        super(PanoramaDataLoader, self).__init__()
        
        self.mode = mode
        
        gt_dir = 'T' 
        input_dir = 'M'
        gt_R_dir = 'R'
        
        self.gt_filenames = glob.glob(f'{data_dir}/{gt_dir}/*.png')
        self.input_filenames = glob.glob(f'{data_dir}/{input_dir}/*.png')
        self.R_filenames = glob.glob(f'{data_dir}/{gt_R_dir}/*.png')

        self.opt=opt

        self.tar_size = len(self.gt_filenames)


    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        # gt_filename = os.path.split(self.gt_filenames[tar_index])[-1][:-4]
        # input_filename = os.path.split(self.input_filenames[tar_index])[-1][:-4]
        gt_path = self.gt_filenames[tar_index]
        input_path = self.input_filenames[tar_index]
        R_path = self.R_filenames[tar_index]
        
        gt_np = np.float32(load_img(self.gt_filenames[tar_index]))
        inp_np = np.float32(load_img(self.input_filenames[tar_index]))
        r_np = np.float32(load_img(self.R_filenames[tar_index]))

        ########################### TRAINING ###########################
        if self.mode == 'train':            
            # numpy to tensor
            gt = torch.from_numpy(gt_np).permute(2,0,1)
            inp = torch.from_numpy(inp_np).permute(2,0,1)
            r = torch.from_numpy(r_np).permute(2,0,1)

            # Resize to pre-defined size
            gt = TF.resize(gt, (self.opt["GT_size_h"], self.opt["GT_size_w"]))            
            inp = TF.resize(inp, (self.opt["LR_size_h"], self.opt["LR_size_w"]))
            r = TF.resize(r, (self.opt["LR_size_h"], self.opt["LR_size_w"]))

            ## DATA AUG
            # Random horizontal flipping
            if random.random() > 0.5:
                gt = TF.hflip(gt)
                inp = TF.hflip(inp)
                r = TF.hflip(r)

            return {"LQ": inp, "GT": gt, "R": r, "LQ_path": input_path, "GT_path": gt_path, "R_path": R_path}

        
        ########################### VALIDATION ###########################
        elif self.mode == 'val':
            tar_index = index % self.tar_size

            # gt_filename = os.path.split(self.gt_filenames[tar_index])[-1][:-4]
            # input_filename = os.path.split(self.input_filenames[tar_index])[-1][:-4]
            gt_path = self.gt_filenames[tar_index]
            input_path = self.input_filenames[tar_index]
            R_path = self.R_filenames[tar_index]
            
            # numpy to tensor
            gt = torch.from_numpy(gt_np).permute(2,0,1)
            inp = torch.from_numpy(inp_np).permute(2,0,1)
            r = torch.from_numpy(r_np).permute(2,0,1)
            
            # Resize to pre-defined size
            gt = TF.resize(gt, (256, 256))
            inp = TF.resize(inp, (256, 256))
            r = TF.resize(r, (256, 256))

            return {"LQ": inp, "GT": gt, "R": r, "LQ_path": input_path, "GT_path": gt_path, "R_path": R_path}




def get_training_data(data_dir, opt):
    assert os.path.exists(data_dir)
    return PanoramaDataLoader('train', data_dir, opt)

def get_validation_data(data_dir, opt):
    assert os.path.exists(data_dir)
    return PanoramaDataLoader('val', data_dir, opt)