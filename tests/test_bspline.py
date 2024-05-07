import pytest
from jax import random
import jax.numpy as jnp
import numpy as np
import torch

from splinx.spline import B_batch, extend_grid, coef2curve

__author__ = "Jamison Moody"
__copyright__ = "Jamison Moody"
__license__ = "GPLv3"

"""
torch functions pulled directly from https://github.com/KindXiaoming/pykan/blob/baf5f7e44ec219894531c6655b211a0b8509526e/kan/spline.py
"""

def B_batch_torch(x, grid, k=0, extend=True, device='cpu'):
    '''
    evaludate x on B-spline bases
    
    
    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of splines, number of samples)
        grid : 2D torch.tensor
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
        device : str
            devicde
    
    Returns:
    --------
        spline values : 3D torch.tensor
            shape (number of splines, number of B-spline bases (coeffcients), number of samples). The numbef of B-spline bases = number of grid points + k - 1.
      
    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> B_batch(x, grids, k=k).shape
    torch.Size([5, 13, 100])
    '''

    # x shape: (size, x); grid shape: (size, grid)
    def extend_grid(grid, k_extend=0):
        # pad k to left and right
        # grid shape: (batch, grid)
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

        for i in range(k_extend):
            grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
        grid = grid.to(device)
        return grid

    if extend == True:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2).to(device)
    x = x.unsqueeze(dim=1).to(device)

    if k == 0:
        value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    else:
        B_km1 = B_batch_torch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False, device=device)
        value = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * B_km1[:, :-1] + (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * B_km1[:, 1:]
    return value

def coef2curve_torch(x_eval, grid, coef, k, device="cpu"):
    '''
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).
    
    Args:
    -----
        x_eval : 2D torch.tensor)
            shape (number of splines, number of samples)
        grid : 2D torch.tensor)
            shape (number of splines, number of grid points)
        coef : 2D torch.tensor)
            shape (number of splines, number of coef params). number of coef params = number of grid intervals + k
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde
        
    Returns:
    --------
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        
    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> coef = torch.normal(0,1,size=(num_spline, num_grid_interval+k))
    >>> coef2curve(x_eval, grids, coef, k=k).shape
    torch.Size([5, 100])
    '''
    # x_eval: (size, batch), grid: (size, grid), coef: (size, coef)
    # coef: (size, coef), B_batch: (size, coef, batch), summer over coef
    y_eval = torch.einsum('ij,ijk->ik', coef, B_batch_torch(x_eval, grid, k, device=device))
    return y_eval

@pytest.mark.parametrize("seed, k", [
    (0, 2), (0, 3), (0, 4),    # Different values of k with the same seed
    (1, 3), (2, 3), (3, 3)     # Different seeds with the same k value
])
def test_B_batch(seed, k):
    num_spline = 5
    num_sample = 100
    num_grid_interval = 10

    key = random.PRNGKey(seed)
    x = random.normal(key, shape=(num_spline, num_sample))
    grids = jnp.einsum('i,j->ij', jnp.ones(num_spline), jnp.linspace(-1, 1, num_grid_interval+1))

    # Converting JAX Arrays to NumPy arrays
    x_torch = torch.tensor(np.array(x))
    grids_torch = torch.tensor(np.array(grids))

    extended_grid = extend_grid(grids, k)
    batch_jax = B_batch(x, extended_grid, k=k)

    # Comparison logic
    assert jnp.allclose(batch_jax, jnp.array(B_batch_torch(x_torch, grids_torch, k=k))), "JAX output does not match expected output"


@pytest.mark.parametrize("seed, k", [
    (10, 2), (10, 3), (10, 4),    # Different values of k with the same seed
    (11, 3), (12, 3), (13, 3)     # Different seeds with the same k value
])
def test_coef2curve(seed, k):

    num_spline = 5
    num_sample = 100
    num_grid_interval = 10
    k = 3
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    x_eval = random.normal(key, shape=(num_spline, num_sample))
    grids = jnp.einsum('i,j->ij', jnp.ones(num_spline,), jnp.linspace(-1,1,num_grid_interval+1))
    coef = random.normal(subkey, (num_spline, num_grid_interval+k))
    extended_grids = extend_grid(grids, k)

    x_torch = torch.tensor(np.array(x_eval))
    grids_torch = torch.tensor(np.array(grids))
    coef_torch = torch.tensor(np.array(coef))

    assert jnp.allclose(coef2curve(x_eval, extended_grids, coef, k=k),
                        jnp.array(coef2curve_torch(x_torch, grids_torch, coef_torch, k=k)), rtol=1e-4), "JAX output does not match expected output"

