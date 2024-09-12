import math
import matplotlib.pyplot as plt
'''
Both functions below assume the tensor input is [row*cols, 1]. They expect the samples to be nicely arranged in a grid.
Therefore, these functions would not work well for random samples (MeshSampler) but for contiguous samples (GridSampler), such as an analytical surface.
'''

def get_plot_single_tensor(tensor):
    fig = plt.figure()
    ax = fig.gca()
    spatial_sidelen = math.isqrt(tensor.shape[0])
    pcolormesh = ax.pcolormesh(tensor.view(spatial_sidelen, spatial_sidelen).detach().cpu())
    fig.colorbar(pcolormesh, ax=ax)
    return fig

def get_plot_grid_tensor(tensor_1, tensor_2, tensor_3, tensor_4):
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0.22)
    (ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')
    spatial_sidelen = math.isqrt(tensor_1.shape[0])
    pcolormesh1 = ax1.pcolormesh(tensor_1.view(spatial_sidelen, spatial_sidelen).detach().cpu())
    pcolormesh2 = ax2.pcolormesh(tensor_2.view(spatial_sidelen, spatial_sidelen).detach().cpu())
    pcolormesh3 = ax3.pcolormesh(tensor_3.view(spatial_sidelen, spatial_sidelen).detach().cpu())
    pcolormesh4 = ax4.pcolormesh(tensor_4.view(spatial_sidelen, spatial_sidelen).detach().cpu())
    fig.colorbar(pcolormesh1, ax=ax1)
    fig.colorbar(pcolormesh2, ax=ax2)
    fig.colorbar(pcolormesh3, ax=ax3)
    fig.colorbar(pcolormesh4, ax=ax4)
    return fig