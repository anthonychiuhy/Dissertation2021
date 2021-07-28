import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

def save_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_object(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)

def make_grid_data(x_interval, y_interval):
    n_x, n_y = len(x_interval), len(y_interval)
    grid_x, grid_y = np.meshgrid(x_interval, y_interval)
    x, y = grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)
    return grid_x, grid_y, np.concatenate((x, y), axis=1)
    
def plot_flow2D(flow, x_interval=np.linspace(-7, 7, 500), y_interval=np.linspace(-7, 7, 500), cutoff=None):
    grid_x, grid_y, X = make_grid_data(x_interval, y_interval)
    X = torch.from_numpy(X).float()
    
    log_pX = flow.log_likelihood(X)
    pX = torch.exp(log_pX) 
    pX[~pX.isfinite()] = 0. # Make -inf of nan values to have 0 density
    pX = pX.numpy().reshape(len(x_interval), len(y_interval))
    
    if cutoff is not None:
        max_val = cutoff * pX.max()
        idx = pX > max_val
        pX[idx] = max_val
    
    plt.pcolormesh(grid_x, grid_y, pX, shading='auto')

def plot_object2D(obj, x_interval=np.linspace(-7, 7, 500), y_interval=np.linspace(-7, 7, 500)):
    grid_x, grid_y, X = make_grid_data(x_interval, y_interval)

    log_pX = obj.logpdf_multiple(X)
    pX = np.exp(log_pX).reshape(len(x_interval), len(y_interval))
    
    plt.pcolormesh(grid_x, grid_y, pX, shading='auto')

def plot_outrange(pX, pX_ensemble, grid_x, grid_y, conf_interval=50, cutoff=None):
    pX_outrange = np.zeros_like(pX)
    
    gap = (100 - conf_interval) / 2
    pX_down = np.percentile(pX_ensemble, gap, axis=0)
    pX_up = np.percentile(pX_ensemble, 100-gap, axis=0)

    idx_up = pX > pX_up
    idx_down = pX < pX_down

    pX_outrange[idx_up] = (pX - pX_up)[idx_up]
    pX_outrange[idx_down] = (pX_down - pX)[idx_down]
    
    if cutoff is not None:
        max_val = cutoff * pX_outrange.max()
        idx = pX_outrange > max_val
        pX_outrange[idx] = max_val

    plt.pcolormesh(grid_x, grid_y, pX_outrange, shading='auto')
    
def flow_density(flow, X):
    X = torch.tensor(X).float()
    log_pX = flow.log_likelihood(X)
    pX = torch.exp(log_pX)
    pX[~pX.isfinite()] = 0. # Make -inf of nan values to have 0 density
    return pX.numpy()

def object_density(obj, X):
    log_pX = obj.logpdf_multiple(X)
    pX = np.exp(log_pX)
    return pX