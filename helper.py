import math
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import torch

def grads2img(gradients):
    mG = gradients.detach().squeeze(0).cpu()

    # assumes mG is [row,cols,2]
    nRows = mG.shape[0]
    nCols = mG.shape[1]
    mGr = mG[:, :, 0]
    mGc = mG[:, :, 1]
    mGa = np.arctan2(mGc, mGr)
    mGm = np.hypot(mGc, mGr)
    mGhsv = np.zeros((nRows, nCols, 3), dtype=np.float32)
    mGhsv[:, :, 0] = (mGa + math.pi) / (2. * math.pi)
    mGhsv[:, :, 1] = 1.

    nPerMin = np.percentile(mGm, 5)
    nPerMax = np.percentile(mGm, 95)
    mGm = (mGm - nPerMin) / (nPerMax - nPerMin)
    mGm = np.clip(mGm, 0, 1)

    mGhsv[:, :, 2] = mGm
    mGrgb = colors.hsv_to_rgb(mGhsv)
    return torch.from_numpy(mGrgb).permute(2, 0, 1)

def get_plot_single_tensor(tensor, spatial_sidelen):
    # assumes mG is [row*cols, 1]
    fig = plt.figure()
    ax = fig.gca()
    pcolormesh = ax.pcolormesh(tensor.view(spatial_sidelen, spatial_sidelen).detach().cpu())
    fig.colorbar(pcolormesh, ax=ax)
    return fig

def get_plot_grid_tensor(tensor_1, tensor_2, tensor_3, tensor_4, spatial_sidelen):
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0.22)
    (ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')
    pcolormesh1 = ax1.pcolormesh(tensor_1.view(spatial_sidelen, spatial_sidelen).detach().cpu())
    pcolormesh2 = ax2.pcolormesh(tensor_2.view(spatial_sidelen, spatial_sidelen).detach().cpu())
    pcolormesh3 = ax3.pcolormesh(tensor_3.view(spatial_sidelen, spatial_sidelen).detach().cpu())
    pcolormesh4 = ax4.pcolormesh(tensor_4.view(spatial_sidelen, spatial_sidelen).detach().cpu())
    fig.colorbar(pcolormesh1, ax=ax1)
    fig.colorbar(pcolormesh2, ax=ax2)
    fig.colorbar(pcolormesh3, ax=ax3)
    fig.colorbar(pcolormesh4, ax=ax4)
    return fig