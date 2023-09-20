import argparse
import numpy as np
import random
import torch
from systems_pbc import *
import torch.backends.cudnn as cudnn
from utils import *
from visualize import *
import matplotlib.pyplot as plt
from models import FNN3d
from train_utils import Adam
from tqdm import tqdm
from train_utils.losses import GeoPC_loss
import matplotlib.image as pm
import torch.nn as nn
# from mpl_toolkits.basemap import Basemap
import tifffile
from data import utils
import scipy.ndimage
import boundary
from PIL import Image
from pyMesh import visualize2D, setAxisLabel
import train_utils.tensorboard as tb
from AWL import AutomaticWeightedLoss
import torch.nn.functional as F
from hydraulics import saint_venant
from scipy import interpolate
from skimage.transform import resize
from train_utils.losses import *
from torch.utils.data import DataLoader
from pyMesh import visualize2D
import imageio
import io
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
gpu_ids = [0]
output_device = gpu_ids[0]



################
# Arguments
################
parser = argparse.ArgumentParser(description='FNO')
parser.add_argument('--loss_style', default='mean', help='Loss for the network (MSE, vs. summing).')

parser.add_argument('--visualize', default=True, help='Visualize the solution.')
parser.add_argument('--save_model', default=True, help='Save the model for analysis later.')
# PINO_model
parser.add_argument('--layers', nargs='+', type=int, default=[16, 24, 24, 32, 32], help='Dimensions/layers of the NN')
parser.add_argument('--modes1', nargs='+', type=int, default=[8, 8, 12, 12], help='')
parser.add_argument('--modes2', nargs='+', type=int, default=[8, 8, 12, 12], help='')
parser.add_argument('--modes3', nargs='+', type=int, default=[8, 8, 8, 8], help='')
parser.add_argument('--fc_dim', type=int, default=128, help='')
parser.add_argument('--epochs', type=int, default=15000)
parser.add_argument('--activation', default='gelu', help='Activation to use in the network.')
#train
parser.add_argument('--base_lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--milestones', nargs='+', type=int, default=[500, 1000, 2000, 3000, 4000, 5000], help='')
parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='')
parser.add_argument('--theta', type=float, default=0.7, help='q centered weighting. [0,1].')

args = parser.parse_args()
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
            t0 = 80
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
            t0 = 80
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



def train():
    # # model
    model = FNN3d(modes1=args.modes1, modes2=args.modes2, modes3=args.modes3, fc_dim=args.fc_dim,
                  layers=args.layers).to(device)
    model = nn.DataParallel(model, device_ids=gpu_ids, output_device=output_device)
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=args.base_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.scheduler_gamma)

    model.train()
    # input data
    epochs = 5000
    pbar = range(epochs)
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.05)
    model.train()
    trainset = train_data(train=True)
    train_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
    for e in range(epochs):
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            input_data, gt_h, z, Rain, Manning, mm = data
            input_data, gt_h, z, Rain, Manning = input_data.to(device), gt_h.to(device), z.to(device), Rain.to(device), Manning.to(device)
            h_init = input_data[..., 0, -1]
            init_condition = [h_init]
            data_condition = [gt_h]
            out = model(input_data)
            # print(out.shape)
            # boundary
            output = out.permute(0, 3, 1, 2, 4)
            outputH = output[:, :, :, :, 0].clone()
            loss_d, loss_c = GeoPC_loss(input_data, outputH, data_condition, init_condition)
            total_loss = loss_c + loss_d
            total_loss.backward(retain_graph=True)
            optimizer.step()

            total_loss = total_loss.item()
            scheduler.step()
            pbar.set_description(
                (
                    f'Epoch {e} '
                    f'loss_d: {loss_d:.5f} '
                    f'loss_c: {loss_c:.5f} '
                )
            )
            # loss
            tb.log_scalars(e, write_hparams=True,
                           loss_d=loss_d)
        if e%500 == 0:
            log_dir = '/mnt/SSD1/qinqsong/ML4Earth_Hackathon/FNO/Results/'
            eval(model, log_dir)

            if args.save_model == True:
                state_dict = model.state_dict()
                torch.save({'epoch': e, 'state_dict': state_dict},
                           log_dir + f'pretrain/checkpoint_%d.pth'%(e))
            # torch.cuda.empty_cache()
    print('Done!')

def eval(model, log_dir):
    model.eval()
    valset = train_data(train=False)
    val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)
    key = 0
    for i, data in enumerate(val_loader):
        with torch.no_grad():
            out = model(input_data)
        h = out[:, :, :, :, :1]
        h, h01 = F.threshold(h, threshold=0, value=0), F.threshold(h, threshold=0.1, value=0)
        # h_gt, qx_gt, qy_gt = data(i)
        h, h01 = torch.squeeze(h), torch.squeeze(h01)
        h, h01 = h.permute(2, 0, 1), h01.permute(2, 0, 1)
        h_g = torch.squeeze(gt_h)
        h_p = h.detach().cpu().numpy()
        # h_p = h_p.reshape(-1, 1)
        u_p = u.detach().cpu().numpy()
        # u_p = u_p.reshape(-1, 1)
        v_p = v.detach().cpu().numpy()
        # v_p = v_p.reshape(-1, 1)

        h_gt = h_g.detach().cpu().numpy()
        print('h_gt', h_gt.shape)
        print('h_p', h_p.shape)
        # MSE error
        error_h_abs = np.mean(np.abs(h_gt - h_p))

        print('error_h_abs',error_h_abs)


if __name__ == '__main__':
    train()