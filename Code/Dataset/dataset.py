import argparse
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.image as pm
import torch.nn as nn
import tifffile
from data import utils
import scipy.ndimage
from PIL import Image
import torch.nn.functional as F
from scipy import interpolate
from skimage.transform import resize
from torch.utils.data import DataLoader
import io
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
gpu_ids = [0]
output_device = gpu_ids[0]

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(device)
else:
    device = torch.device('cpu')
    print(device)


############################
# Process data
############################
def inter(array, size):
    h, w = array.shape
    new_h, new_w = np.floor_divide((h, w), size)
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    new_x = np.linspace(0, w - 1, new_w)
    new_y = np.linspace(0, h - 1, new_h)
    f = interpolate.interp2d(x, y, array, kind='linear')
    array_down = f(new_x, new_y)
    # array_down = resize(array, (new_h, new_w), order=1, anti_aliasing=True)
    return array_down

# Parameters
g = torch.tensor(9.80616, dtype=torch.float64)
# input data
dem_tif_path = '/mnt/SSD1/qinqsong/ML4Earth_Hackathon/Dataset/Input_data/dem_mask.tif'
pre_path = '/mnt/SSD1/qinqsong/ML4Earth_Hackathon/Dataset/Rainfall'
man_path = '/mnt/SSD1/qinqsong/ML4Earth_Hackathon/Dataset/Input_data/manning.npy'
ini_height = '/mnt/SSD1/qinqsong/ML4Earth_Hackathon/Dataset/Input_data/waterDepth.tif'
# Training and validation Dataset
train_path = '/mnt/SSD1/qinqsong/ML4Earth_Hackathon/Dataset/Training_dataset'
val_path = '/mnt/SSD1/qinqsong/ML4Earth_Hackathon/Dataset/Validation_dataset'

dem_map = tifffile.imread(dem_tif_path)
dem_map = dem_map[37:-33, 170:-57]
dem_map_down = inter(dem_map, 16)
print('dem_map_down', dem_map_down.shape)
TILE_SIZE_X = 2000
TILE_SIZE_Y = 2000
ALLOWED_MASKED_PERCENTAGE = 0
MAX_TOPOGRAPHY_DIFFERENCE = 2000

# DEM z
def process_dem(dem_map):
    # with open(dem_tif_path, 'rb') as f:
    #     tiffbytes = f.read()
    np_ma_map = np.ma.masked_array(dem_map, mask=(dem_map < -2000))
    np_ma_map = utils.fix_missing_values(np_ma_map)
    dem = torch.from_numpy(np_ma_map)
    return dem.float()
# Precipitation
def precip(pre_path, i, t0, downsampling = True):
    pre_list = os.listdir(pre_path)
    pre_list.sort()
    lens = len(pre_list)
    pre_list_train = []
    for j in range(i*(t0),(i+1)*(t0)):
        mp = int(j / 60)
        if mp >= lens:
            mp = lens-1
        pre_list_train.append(pre_list[mp])
    print('pre_list', pre_list_train)
    count = 0
    for f in pre_list_train:
        f_path = os.path.join(pre_path,f)
        imag = tifffile.imread(f_path)
        imag = imag[37: -33, 170: -57]
        if downsampling == True:
            imag = inter(imag, 16)
        imag[imag<0] = 0
        imag = torch.from_numpy(imag)
        imag = torch.unsqueeze(imag, dim=0)
        if count == 0:
            tiles_time = imag
        else:
            tiles_time = torch.cat((tiles_time,imag), dim=0)
        count = count + 1
    # tiles_time = tiles_time[i*t0:(i+1)*t0,: ,:]
    tiles_time = torch.unsqueeze(tiles_time, dim=0)
    tiles_time = tiles_time.float()
    return tiles_time

# Manning Coefficient
def manning(man_path, downsampling = True):
    img = np.load(man_path)
    np_map = np.array(img)
    np_map = np_map[37: -33, 170: -57]
    if downsampling == True:
        np_map = inter(np_map, 16)
    print('manning', np_map.shape)
    man = torch.from_numpy(np_map)
    man = torch.unsqueeze(man, dim=0)
    man = torch.unsqueeze(man, dim=0)
    return man.float()

# initial water level h
def gen_init(ini_height, ini_discharge, downsampling = True):
    imag = tifffile.imread(ini_height)
    imag_river = tifffile.imread(ini_discharge)
    rows, cows = imag.shape
    imag = imag[37: -33, 170: -57]
    imag_river = imag_river[37: -33, 170: -57]
    if downsampling == True:
        imag = inter(imag, 16)
        imag_river = inter(imag_river, 16)
    imag[imag < 0] = 0
    image = torch.from_numpy(imag)
    return image.float()
# If boundary conditions are to be considered in your designed model,
# the study area only considers the discharges at the inflow boundary, which is 13236m3/s.
def boundary_r(dem_map, mm):
    rows, cols = dem_map.shape
    dem_shape = [rows, cols]
    influx = (128, -1, 10)
    # discharge in influx
    dischargein = [13236]  # [m^3/s] cubic meters per second
    dischargeiout = [0]
    dischargein_a = np.array(dischargein[mm])
    dischargeout_a = np.array(dischargeiout[mm])
    discharge = [dischargein_a, dischargeout_a]
    boundary_conditions = boundary.FluxBoundaryConditions(
        dem_shape, influx, outflux, discharge)
    return boundary_conditions

def data(i, path):
    t0 = 16
    dt0 = 30
    t00, tfinal = (i * (t0)) * dt0, ((i+1)*(t0)) * dt0
    h_gt = []
    for i in range(t00,tfinal,dt0):
        current_time = torch.tensor(i, dtype=torch.float).to(device)
        print('current_time', current_time)
        path_h = os.path.join(path, str(current_time)+".tiff")
        h_current = Image.open(path_h)
        h_current = np.array(h_current)
        h_current = torch.from_numpy(h_current)
        h_current = h_current.float()
        h_gt.append(h_current)
    h_gt = torch.stack(h_gt, 0)
    print('len_supervised of water height', h_gt.size())
    return h_gt

# Process data into input to a neural network
class train_data():
    def __init__(self, train=True):
        super().__init__()
        self.load(train=train)

    def load(self, train=True):
        if train:
            t0 = 16
            lmbdleft, lmbdright = 0, (dem_map_down.shape[0] - 1) * 30 * 16
            thtlower, thtupper = 0, (dem_map_down.shape[1] - 1) * 30 * 16
            dt = 30
            l = (7 * 12 * 6 * 60) / dt
            t00, tfinal = 0, (t0 - 1) * dt
            nt = int(l / (t0))
            print('Number of training sequences', nt)
            m = dem_map_down.shape[0]
            n = dem_map_down.shape[1]

            t = np.linspace(t00, tfinal, t0)
            x = np.linspace(lmbdleft, lmbdright, m)
            y = np.linspace(thtlower, thtupper, n)
            data_star = np.hstack((x.flatten(), y.flatten(), t.flatten()))
            # Data normalization
            lb = data_star.min(0)
            ub = data_star.max(0)
            input_data_list = []
            h_gt_list = []
            z_list = []
            Rain_list = []
            Manning_list = []
            mm_data = []
            for i in range(nt):
                mm = int((i * (t0)) / 2880)
                mm = np.array(mm)
                print("Calculate the number of days:", mm)
                mm = torch.from_numpy(mm)
                mm = mm.float()
                mm = torch.unsqueeze(mm, dim=0)
                # model.train()
                h_gt = data(i, train_path)
                gridx = torch.from_numpy(x)
                gridx = gridx.reshape(1, m, 1, 1, 1).repeat([1, 1, n, t0, 1])
                gridy = torch.from_numpy(y)
                gridy = gridy.reshape(1, 1, n, 1, 1).repeat([1, m, 1, t0, 1])
                gridt = torch.from_numpy(t)
                gridt = gridt.reshape(1, 1, 1, t0, 1).repeat([1, m, n, 1, 1])
                h_init = h_gt[0, :, :]
                h_init = h_init.reshape(1, m, n, 1, 1).repeat([1, 1, 1, t0, 1])
                input_data = torch.cat((gridt, gridx, gridy), dim=-1)
                input_data = 2.0 * (input_data - lb) / (ub - lb) - 1.0
                input_data = torch.cat((input_data, h_init.cpu()), dim=-1)
                input_data = input_data.float()
                # initial condition
                # if i == 0:
                h_gt = torch.unsqueeze(h_gt, dim=0)
                z = process_dem(dem_map_down)
                z = torch.unsqueeze(z, dim=0)
                # rainfall
                Rain = precip(pre_path, i, t0, downsampling=True)
                Rain = torch.unsqueeze(Rain, dim=0)
                Manning = manning(man_path, downsampling=True)
                Manning = torch.unsqueeze(Manning, dim=0)
                input_data_list.append(input_data)
                h_gt_list.append(h_gt)
                z_list.append(z)
                Rain_list.append(Rain)
                Manning_list.append(Manning)
                mm_data.append(mm)
            data_input = torch.cat(input_data_list, dim=0)
            gt_h = torch.cat(h_gt_list, dim=0)
            data_z = torch.cat(z_list, dim=0)
            data_Rain = torch.cat(Rain_list, dim=0)
            data_Manning = torch.cat(Manning_list, dim=0)
            data_mm = torch.cat(mm_data, dim=0)
            self.data_input = data_input
            self.gt_h = gt_h
            self.data_z = data_z
            self.data_Rain = data_Rain
            self.data_Manning = data_Manning
            self.data_mm = data_mm
        else:
            t0 = 16
            lmbdleft, lmbdright = 0, (dem_map_down.shape[0] - 1) * 30 * 16
            thtlower, thtupper = 0, (dem_map_down.shape[1] - 1) * 30 * 16
            dt = 30
            t00, tfinal = 0, (t0 - 1) * dt
            l0 = int((7 * 12 * 6 * 60) / dt)
            l1 = int((12 * 60 * 60) / dt)
            nt0 = int(l0 / t0)
            nt1 = int(l1 / t0)

            print('Number of validation sequences', (nt1-nt0))
            m = dem_map_down.shape[0]
            n = dem_map_down.shape[1]

            t = np.linspace(t00, tfinal, t0)
            x = np.linspace(lmbdleft, lmbdright, m)
            y = np.linspace(thtlower, thtupper, n)
            data_star = np.hstack((x.flatten(), y.flatten(), t.flatten()))
            # Data normalization
            lb = data_star.min(0)
            ub = data_star.max(0)
            input_data_list = []
            h_gt_list = []
            z_list = []
            Rain_list = []
            Manning_list = []
            mm_data = []
            for i in range(nt0, nt1):
                mm = int((i * (t0)) / 2880)
                mm = np.array(mm)
                print("Calculate the number of days:", mm)
                mm = torch.from_numpy(mm)
                mm = mm.float()
                mm = torch.unsqueeze(mm, dim=0)
                h_gt = data(i, val_path)
                gridx = torch.from_numpy(x)
                gridx = gridx.reshape(1, m, 1, 1, 1).repeat([1, 1, n, t0, 1])
                gridy = torch.from_numpy(y)
                gridy = gridy.reshape(1, 1, n, 1, 1).repeat([1, m, 1, t0, 1])
                gridt = torch.from_numpy(t)
                gridt = gridt.reshape(1, 1, 1, t0, 1).repeat([1, m, n, 1, 1])
                h_init = h_gt[0, :, :]
                h_init = h_init.reshape(1, m, n, 1, 1).repeat([1, 1, 1, t0, 1])
                input_data = torch.cat((gridt, gridx, gridy), dim=-1)
                input_data = 2.0 * (input_data - lb) / (ub - lb) - 1.0
                input_data = torch.cat((input_data, h_init.cpu()), dim=-1)
                input_data = input_data.float()
                # initial condition
                h_gt = torch.unsqueeze(h_gt, dim=0)
                z = process_dem(dem_map_down)
                z = torch.unsqueeze(z, dim=0)
                # rainfall
                Rain = precip(pre_path, i, t0, downsampling=True)
                Rain = torch.unsqueeze(Rain, dim=0)
                Manning = manning(man_path, downsampling=True)
                Manning = torch.unsqueeze(Manning, dim=0)
                input_data_list.append(input_data)
                h_gt_list.append(h_gt)
                z_list.append(z)
                Rain_list.append(Rain)
                Manning_list.append(Manning)
                mm_data.append(mm)
            data_input = torch.cat(input_data_list, dim=0)
            gt_h = torch.cat(h_gt_list, dim=0)
            data_z = torch.cat(z_list, dim=0)
            data_Rain = torch.cat(Rain_list, dim=0)
            data_Manning = torch.cat(Manning_list, dim=0)
            data_mm = torch.cat(mm_data, dim=0)
            self.data_input = data_input
            self.gt_h = gt_h
            self.data_z = data_z
            self.data_Rain = data_Rain
            self.data_Manning = data_Manning
            self.data_mm = data_mm


    def __getitem__(self, idx):
        return self.data_input[idx], self.gt_h[idx], self.data_z[idx], self.data_Rain[idx], self.data_Manning[idx], self.data_mm[idx]

    def __len__(self, ):
        return self.data_input.shape[0]

if __name__ == '__main__':
    # trainset = train_data(train=True)
    # train_loader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=4)
    # print('lens of train dataset', len(train_loader))

    valset = train_data(train=False)
    train_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=4)
    print('lens of train dataset', len(train_loader))
