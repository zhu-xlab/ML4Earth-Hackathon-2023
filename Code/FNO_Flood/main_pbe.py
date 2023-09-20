import argparse
import numpy as np
import os
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
from mpl_toolkits.basemap import Basemap
from skimage.external import tifffile
from data import utils
import scipy.ndimage
import boundary
import io
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


################
# Arguments
################
parser = argparse.ArgumentParser(description='GeoPINS')
parser.add_argument('--loss_style', default='mean', help='Loss for the network (MSE, vs. summing).')

parser.add_argument('--visualize', default=True, help='Visualize the solution.')
parser.add_argument('--save_model', default=False, help='Save the model for analysis later.')
# PINO_model
parser.add_argument('--layers', nargs='+', type=int, default=[16, 24, 24, 32, 32], help='Dimensions/layers of the NN')
parser.add_argument('--modes1', nargs='+', type=int, default=[12, 12, 9, 9], help='')
parser.add_argument('--modes2', nargs='+', type=int, default=[12, 12, 9, 9], help='')
parser.add_argument('--modes3', nargs='+', type=int, default=[4, 4, 4, 4], help='')
parser.add_argument('--fc_dim', type=int, default=128, help='')
parser.add_argument('--epochs', type=int, default=15000)
parser.add_argument('--activation', default='gelu', help='Activation to use in the network.')
#train
parser.add_argument('--base_lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--milestones', nargs='+', type=int, default=[500, 1500, 3000, 4000, 6000], help='')
parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='')

args = parser.parse_args()
# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


############################
# Process data
############################

# Parameters
g = torch.tensor(9.80616, dtype=torch.float64)
dem_tif_path = 'E:/1_COPY/Flood/Pakistan/input/dem_utm_mask.tif'
pre_path = 'E:/1_COPY/Flood/Pakistan/input/rainfall_utm_tif'
man_path = 'E:/1_COPY/Flood/Pakistan/input/manning.tif'
dem_map = tifffile.imread(io.BytesIO(dem_tif_path))
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
    tiles = utils.divide_to_tiles(np_ma_map, (TILE_SIZE_X, TILE_SIZE_Y))
    tiles_filtered = []
    for tile in tiles:
        if (utils.masked_values_percentage(
                tile) <= ALLOWED_MASKED_PERCENTAGE) and (
                utils.topography_difference(
                    tile) <= MAX_TOPOGRAPHY_DIFFERENCE):
            tiles_filtered.append(tile)
    return tiles_filtered
# Precipitation
def precip(pre_path):
    pre_list = os.listdir(pre_path)
    tiles_time = []
    for f in pre_list:
        imag = tifffile.imread(f)
        # imag_new = np.zeros([dem_map.shape[0], dem_map.shape[1]])
        # scale_h = int(dem_map.shape[0] / imag.shape[0]) + 1
        # scale_w = int(dem_map.shape[1] / imag.shape[1]) + 1
        # label_new = scipy.ndimage.zoom(imag, zoom=[scale_h, scale_w], order=0)
        # imag_new =
        tiles = utils.divide_to_tiles(imag, (TILE_SIZE_X, TILE_SIZE_Y))
        tiles_time.append(tiles)
    return tiles_time

# Manning Coefficient
def manning(man_path):
    with open(man_path, 'rb') as f:
        tiffbytes = f.read()
    np_map = tifffile.imread(io.BytesIO(tiffbytes))
    tiles = utils.divide_to_tiles(np_map, (TILE_SIZE_X, TILE_SIZE_Y))
    tiles_filtered = []
    for tile in tiles:
        tiles_filtered.append(tile)
    return tiles_filtered

# initial water level h
def gen_init():
    source = torch.zeros((TILE_SIZE_X, TILE_SIZE_Y), dtype=torch.float)
    # source = source.unsqueeze(dim=0).unsqueeze(dim=0)
    return source

def boundary(dem_map, output):
    # streamflow (influx and outflux)
    INFLUX_LENGTH = 400
    OUTFLUX_LENGTH = 400
    # outflux
    min_location = utils.find_lowest_point(dem_map)
    x, y = min_location
    rows, cols = dem_map.shape
    dem_shape = [rows, cols]
    outflux = (x, y, OUTFLUX_LENGTH)
    # influx (should replace with real boundary)
    if x > 0:
        influx_axis = 0 if y < 0 else -1
        influx = (rows // 2, influx_axis, INFLUX_LENGTH)
    else:
        influx_axis = 0 if x < 0 else -1
        influx = (influx_axis, cols // 2, INFLUX_LENGTH)
    # discharge in influx and outflux
    dischargein = 250 # [m^3/s] cubic meters per second
    dischargeout = 1
    # FluxBoundaryConditions
    boundary_conditions = boundary.FluxBoundaryConditions(
        dem_shape, influx, outflux, dischargein, dischargeout)
    return boundary_conditions

def train():
    # physical domain
    days = 61
    T = 86400
    lmbdleft, lmbdright = 0, dem_map.shape[0] * 30
    thtlower, thtupper = 0, dem_map.shape[1] * 30
    t0, tfinal = 0, days * 24
    l = 1464
    m = dem_map.shape[0]
    n = dem_map.shape[1]

    t = np.linspace(t0, tfinal, l)
    x = np.linspace(lmbdleft, lmbdright, m)
    y = np.linspace(thtlower, thtupper, n)
    T_p, X_p, Y_p = np.meshgrid(t, x, y, indexing='ij')

    # # model
    model = FNN3d(modes1=args.modes1, modes2=args.modes2, modes3=args.modes3, fc_dim=args.fc_dim,
                  layers=args.layers).to(device)
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=args.base_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.scheduler_gamma)
    model.train()
    # # input data
    numPatchRow = int(x / TILE_SIZE_X)
    numPatchCol = int(y / TILE_SIZE_Y)
    if (x.shape > TILE_SIZE_X * numPatchRow):
        x = np.pad(x, (0, TILE_SIZE_X - (x.shape - TILE_SIZE_X * numPatchRow)), 'symmetric')
    if y.shape > TILE_SIZE_Y * numPatchCol:
        y = np.pad(y, (0, TILE_SIZE_Y - (y.shape - TILE_SIZE_Y * numPatchCol)), 'symmetric')
    Col = int(x / TILE_SIZE_X)
    Row = int(y / TILE_SIZE_Y)
    inputs_data = []
    for i in range(0, Col * Row):
        inputs_data.append([])
    count = 0
    for i in range(Col):
        for j in range(Row):
            mm = ((i + 1) * TILE_SIZE_X)
            nn = ((j + 1) * TILE_SIZE_Y)
            x0 = x[i * TILE_SIZE_X:mm]
            y0 = y[j * TILE_SIZE_Y:nn]
            gridx = torch.from_numpy(x0)
            gridx = gridx.reshape(1, TILE_SIZE_X, 1, 1, 1).repeat([1, 1, TILE_SIZE_Y, l, 1])
            gridy = torch.from_numpy(y)
            gridy = gridy.reshape(1, 1, TILE_SIZE_Y, 1, 1).repeat([1, TILE_SIZE_X, 1, l, 1])
            gridt = torch.from_numpy(t)
            gridt = gridt.reshape(1, 1, 1, l, 1).repeat([1, TILE_SIZE_X, TILE_SIZE_Y, 1, 1])
            # gridt = 2.0 * (gridt - torch.min(gridt)) / (torch.max(gridt) - torch.min(gridt)) - 1.0
            input_data = torch.cat((gridt, gridx, gridy), dim=-1)
            inputs_data[count] = input_data
            count = count + 1

    input_data = input_data.float().to(device)
    # train
    # initial condition
    h_init = gen_init()
    # rainfall
    Rain = precip(pre_path)
    #manning
    Manning = manning(man_path)

    pbar = range(args.epochs)
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.05)
    train_pino = 0.0
    train_loss = 0.0
    model.train()
    for e in pbar:
        for batch_num, image in enumerate((zip(image1,inputs_data))):
        out = model(input_data)
        loss_c, loss_f = GeoPC_loss(input_data, out, hs0, f, h_init, qx_init, qy_init)
        total_loss = loss_c + loss_f

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_pino += loss_f.item()
        train_loss += total_loss.item()
        if e % 50 == 0:
            scheduler.step()
        pbar.set_description(
            (
                f'Epoch {e}, loss_c: {loss_c:.5f} '
                f'loss_f: {loss_f:.5f}'
            )
        )
    print('Done!')







    










# Radius of Earth
a = torch.tensor(6.37122e6, dtype=torch.float64)

# Angular velocity
Omega = torch.tensor(7.292e-5, dtype=torch.float64)

# Scales of problem
T = 86400
L = a.numpy()
U = L/T
HH = a.numpy()**2/(T**2*g.numpy())
print('HH',HH)

days = 5
c = 1.0
H = c * HH
alpha, R, lmbd0, tht0 = 0.0, a.numpy() / 3, 3 * np.pi / 2, 0.0
u00, h00 = 2 * np.pi * a.numpy() / (12 * 86400), 2.94e4
v00 = u00

u0 = lambda lmbd, tht: u00 * (np.cos(tht) * np.cos(alpha) + np.sin(tht) * np.cos(lmbd) * np.sin(alpha)) / U
v0 = lambda lmbd, tht: -v00 * np.sin(lmbd) * np.sin(alpha) / U
h0 = lambda lmbd, tht: (h00 - (a.numpy() * Omega.numpy() * u00 + u00 ** 2 / 2) * (
            -np.cos(lmbd) * np.cos(tht) * np.sin(alpha) + np.sin(tht) * np.cos(alpha)) ** 2) / g.numpy() / H
hs0 = lambda lmbd, tht: torch.sin(0.0 * lmbd) + torch.sin(0.0 * tht)

# Plot initial conditions
# Boundaries of the computational domain
lmbdleft, lmbdright = -np.pi, np.pi
thtlower, thtupper = -np.pi/2, np.pi/2
t0, tfinal = 0, days*86400/T
############################
# Train the model
############################
# # model
model = FNN3d(modes1=args.modes1, modes2=args.modes2, modes3=args.modes3, fc_dim=args.fc_dim, layers=args.layers).to(device)
optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=args.base_lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.scheduler_gamma)
model.train()
#
# train
l, m, n = days+1, 500, 500
t = np.linspace(t0, tfinal, l)
x = np.linspace(lmbdleft, lmbdright, m)
y = np.linspace(thtlower, thtupper, n)
T_p, X_p, Y_p  = np.meshgrid(t,x,y, indexing='ij')

dt = t[1]-t[0]
h_init = h0(X_p[0,], Y_p[0,])
u_init = u0(X_p[0,], Y_p[0,])
v_init = v0(X_p[0,], Y_p[0,])

fig1 = plt.figure()
lon, lat = np.meshgrid(np.linspace(-180, 180, 500), np.linspace(-90, 90, 500))
map = Basemap(projection='ortho', lat_0=45, lon_0=15)
xx, yy = map(lon, lat)
plt.subplot(221)
map.drawmeridians(np.arange(-180, 180, 30))
map.drawparallels(np.arange(-90, 90, 30))
cs = map.contourf(xx, yy, H * h_init, np.linspace(990, 3010, 21), cmap=plt.cm.coolwarm,
                  vmin=H * h_init.min(), vmax=H * h_init.max())
plt.colorbar(ticks=[1000 + 500 * j for j in range(8)])
plt.title('h')
# plt.xlabel(r'$\lambda$')
# plt.ylabel(r'$\theta$')
plt.subplot(222)
map.drawmeridians(np.arange(-180, 180, 30))
map.drawparallels(np.arange(-90, 90, 30))
cs = map.contourf(xx, yy, U * u_init, np.linspace(0, 40, 21), cmap=plt.cm.coolwarm,
                  vmin=U * u_init.min(), vmax=U * u_init.max())
plt.colorbar(ticks=[10 * j for j in range(5)])
plt.title('u')
plt.subplot(223)
map.drawmeridians(np.arange(-180, 180, 30))
map.drawparallels(np.arange(-90, 90, 30))
cs = map.contourf(xx, yy, U * v_init, 21, cmap=plt.cm.coolwarm,
                  vmin=U * v_init.min(), vmax=U * v_init.max())
plt.colorbar()
plt.title('v')
# cs = map.contourf(xx, yy, U*v0(X,Y), 100, cmap=plt.cm.rainbow)
# plt.colorbar()
plt.tight_layout()
plt.show()
fig1.savefig('Initial.jpg', bbox_inches='tight')
plt.close(fig1)

print('h_init',h_init.shape)
h_init = torch.from_numpy(h_init)
h_init = h_init.float().to(device)

u_init = torch.from_numpy(u_init)
u_init = u_init.float().to(device)

v_init = torch.from_numpy(v_init)
v_init = v_init.float().to(device)

gridx = torch.from_numpy(x)
gridx = gridx.reshape(1, m, 1, 1, 1).repeat([1, 1, n, l, 1])
gridy = torch.from_numpy(y)
gridy = gridy.reshape(1, 1, n, 1, 1).repeat([1, m, 1, l, 1])
gridt = torch.from_numpy(t)
gridt = gridt.reshape(1, 1, 1, l, 1).repeat([1, m, n, 1, 1])
gridt = 2.0*(gridt-torch.min(gridt))/(torch.max(gridt)-torch.min(gridt))-1.0
input_data = torch.cat((gridt, torch.cos(gridy)*torch.cos(gridx), torch.cos(gridy)*torch.sin(gridx), torch.sin(gridy)), dim=-1)

input_data = input_data.float().to(device)
f = 2 * Omega * T * torch.sin(gridy)
f = f.float().to(device)

pbar = range(args.epochs)
pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.05)
train_pino = 0.0
train_loss = 0.0
model.train()
for e in pbar:
    out = model(input_data)
    loss_c, loss_f = GeoPC_loss(input_data, out, hs0, f, h_init, u_init, v_init)
    total_loss = loss_c + loss_f

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    train_pino += loss_f.item()
    train_loss += total_loss.item()
    if e % 50 == 0:
        scheduler.step()
    pbar.set_description(
        (
            f'Epoch {e}, loss_c: {loss_c:.5f} '
            f'loss_f: {loss_f:.5f}'
        )
    )
print('Done!')

#eval
model.eval()
out = model(input_data)
u, v, h = out[:,:,:,:,:1], out[:,:,:,:,1:2], out[:,:,:,:,2:3]
u, v, h = torch.squeeze(u), torch.squeeze(v), torch.squeeze(h)
u, v, h = u.permute(2,0,1), v.permute(2,0,1), h.permute(2,0,1)
u, v, h = u.cpu().detach().numpy(), v.cpu().detach().numpy(), h.cpu().detach().numpy()
u, v, h = np.reshape(u, (l, m, n)), np.reshape(v, (l, m, n)), np.reshape(h, (l, m, n))
#
plot_steps = [i for i in range(l)]
for i in plot_steps:
    fig0 = plt.figure()
    lon, lat = np.meshgrid(np.linspace(-180, 180, 500), np.linspace(-90, 90, 500))
    map = Basemap(projection='ortho', lat_0=45, lon_0=15)
    xx, yy = map(lon, lat)
    plt.subplot(221)
    map.drawmeridians(np.arange(-180, 180, 30))
    map.drawparallels(np.arange(-90, 90, 30))
    cs = map.contourf(xx, yy, H*h[i,], np.linspace(990,3010, 21), cmap=plt.cm.coolwarm,
               vmin=H*h[i,].min(), vmax=H*h[i,].max())
    plt.colorbar(ticks=[1000 + 500 * j for j in range(8)])
    plt.title('h on day{: 2.0f}'.format(i))

    plt.subplot(222)
    map.drawmeridians(np.arange(-180, 180, 30))
    map.drawparallels(np.arange(-90, 90, 30))
    cs = map.contourf(xx, yy, U*u[i,], np.linspace(0,40, 21), cmap=plt.cm.coolwarm,
               vmin=U*u[i,].min(), vmax=U*u[i,].max())
    plt.colorbar(ticks=[10*j for j in range(5)])
    plt.title('u on day{: 2.0f}'.format(i))

    plt.subplot(223)
    map.drawmeridians(np.arange(-180, 180, 30))
    map.drawparallels(np.arange(-90, 90, 30))
    cs = map.contourf(xx, yy, U*v[i,], 21, cmap=plt.cm.coolwarm,
               vmin=U*v[i,].min(), vmax=U*v[i,].max())
    plt.colorbar()
    plt.title('v on day{: 2.0f}'.format(i))
    plt.tight_layout()
    plt.show()
    fig0.savefig(str(i) + 'day.jpg', bbox_inches='tight')
    plt.close(fig0)
htrue = h0(X_p[0,], Y_p[0,])
utrue = u0(X_p[0,], Y_p[0,])
vtrue = v0(X_p[0,], Y_p[0,])
l2_h = np.zeros(l)
linfty_h = np.zeros(l)
l2_v = np.zeros(l)
linfty_v = np.zeros(l)

for i in range(l):
  l2_h[i] = np.sqrt(np.sum(np.square(h[i,] - htrue)*np.cos(Y_p[0,]))/np.sum(np.square(htrue)*np.cos(Y_p[0,])))
  linfty_h[i] = np.max(np.abs(h[i,] - htrue))/np.max(np.abs(htrue))

  l2_v[i] = np.sqrt(np.sum( (np.square(u[i,] - utrue) + np.square(v[i,] - vtrue) )*np.cos(Y_p[0,]) )/np.sum( (np.square(utrue) + np.square(vtrue) )*np.cos(Y_p[0,])))
  linfty_v[i] = np.max(np.sqrt( np.square(u[i,] - utrue) + np.square(v[i,] - vtrue) ))/np.max(np.sqrt(np.square(utrue) + np.square(vtrue)))

print("Final l_2 error h: {:4.2e}".format(l2_h[-1]))
print("Final l_infty error h: {:4.2e}".format(linfty_h[-1]))
print("Final l_2 error v: {:4.2e}".format(l2_v[-1]))
print("Final l_infty error v: {:4.2e}".format(linfty_v[-1]))