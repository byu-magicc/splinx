from jax import jit, random
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt

# x shape: (size, x); grid shape: (size, grid)
def extend_grid(grid, k_extend=0):
    # pad k to left and right
    # grid shape: (batch, grid)
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

    for i in range(k_extend):
        grid = jnp.concatenate([grid[:, [0]] - h, grid], axis=1)
        grid = jnp.concatenate([grid, grid[:, [-1]] + h], axis=1)

    return grid

@partial(jit, static_argnames=["k"])
def B_batch(x, grid, k=0):
    '''
    evaludate x on B-spline bases
    
    Args:
    -----
        x : 2D jax.numpy.array
            inputs, shape (number of splines, number of samples)
        grid : 2D jax.numpy.array
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
    
    Returns:
    --------
        spline values : 3D jax.numpy.array
            shape (number of splines, number of B-spline bases (coeffcients), number of samples). The numbef of B-spline bases = number of grid points + k - 1.
      
    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k=3
    >>> key = random.PRNGKey(0)
    >>> x = random.normal(key, shape=(num_spline, num_sample))
    >>> grids = jnp.einsum('i,j->ij', jnp.ones(num_spline), jnp.linspace(-1, 1, num_grid_interval+1))
    >>> extended_grid = extend_grid_jax(grids, k)
    >>> batch_jax = B_batch_jax(x, extended_grid, k=k)
    >>> batch_jax.shape
    (5, 13, 100)
    '''

    value = (x[:,None,:] >= grid[:,:-1,None]) * (x[:,None,:] < grid[:,1:,None])

    for i in range(1, k+1):
        value = (x[:, None, :] - grid[:, :-(i + 1), None]) / (grid[:, i:-1, None] - grid[:, :-(i + 1), None]) * value[:, :-1] + (grid[:, i + 1:, None] - x[:, None, :]) / (grid[:, i + 1:, None] - grid[:, 1:(-i), None]) * value[:, 1:]
    
    return value

@partial(jit, static_argnames=["k"])
def coef2curve(x_eval, grid, coef, k):
    '''
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).
    
    Args:
    -----
        x_eval : 2D jax.numpy.array
            shape (number of splines, number of samples)
        grid : 2D jax.numpy.array
            shape (number of splines, number of grid points)
        coef : 2D jax.numpy.array
            shape (number of splines, number of coef params). number of coef params = number of grid intervals + k
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde
        
    Returns:
    --------
        y_eval : 2D jax.numpy.array
            shape (number of splines, number of samples)
        
    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> key = random.PRNGKey(0)
    >>> key, subkey = random.split(key)
    >>> x_eval = random.normal(key, shape=(num_spline, num_sample))
    >>> grids = jnp.einsum('i,j->ij', jnp.ones(num_spline,), jnp.linspace(-1,1,num_grid_interval+1))
    >>> coef = random.normal(subkey, size=(num_spline, num_grid_interval+k))
    >>> extended_grids = extend_grid(grids, k)
    >>> coef2curve(x_eval, extended_grids, coef, k=k).shape
    (5, 100)
    '''
    # x_eval: (size, batch), grid: (size, grid), coef: (size, coef)
    # coef: (size, coef), B_batch: (size, coef, batch), summer over coef
    y_eval = jnp.einsum('ij,ijk->ik', coef, B_batch(x_eval, grid, k))
    return y_eval

@partial(jit, static_argnames=["k"])
def curve2coef(x_eval, y_eval, grid, k):
    '''
    converting B-spline curves to B-spline coefficients using least squares.
    
    Args:
    -----
        x_eval : 2D jax.numpy.array
            shape (number of splines, number of samples)
        y_eval : 2D jax.numpy.array
            shape (number of splines, number of samples)
        grid : 2D jax.numpy.array
            shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines
        
    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x_eval = random.normal(0,1,size=(num_spline, num_sample))
    >>> y_eval = random.normal(0,1,size=(num_spline, num_sample))
    >>> grids = jnp.einsum('i,j->ij', jnp.ones(num_spline,), jnp.linspace(-1,1,num_grid_interval+1))
    >>> curve2coef(x_eval, y_eval, grids, k=k).shape
    torch.Size([5, 13])
    '''
    # x_eval: (size, batch); y_eval: (size, batch); grid: (size, grid); k: scalar
    mat = B_batch(x_eval, grid, k).permute(0, 2, 1)
    coef = jnp.linalg.lstsq(mat, y_eval[:,:,None]).solution[:, :, 0]
    return coef

if __name__=="__main__":

    # How to plot a 2D spline with control points
    num_sample = 100
    k = 3
    x_eval = jnp.repeat(jnp.linspace(-1, 1, num_sample)[None, :], 2, axis=0)
    
    coef = jnp.array([[1,2,1,-1,5,3,6],
                      [1,2,3,4,5,10,15]])
    num_grid_interval = coef.shape[1] - k
    grids = jnp.einsum('i,j->ij', jnp.ones(2,), jnp.linspace(-1,1,num_grid_interval+1))
    extended_grids = extend_grid(grids, k)

    # Multiply the coeficients by the different basis functions
    y_eval = coef2curve(x_eval, extended_grids, coef, k=k)

    plt.figure(figsize=(10, 6))
    plt.plot(y_eval[0, :], y_eval[1, :], label='Spline')

    # Extract control points for each spline
    # Here, extended_grids would represent the x positions of control points if it aligns with the spline's domain
    ctrl_points_x = coef[0, :]  # You might need to adjust this based on your extend_grid function
    ctrl_points_y = coef[1, :]
    
    # Plot control points
    plt.scatter(ctrl_points_x, ctrl_points_y, marker='s', s=50, color='red', label="Control Points")  # Red circles for control points
    # Optionally connect control points with lines to better visualize the influence
    plt.plot(ctrl_points_x, ctrl_points_y, 'r--', alpha=0.5)  # Dashed lines connecting control points

    plt.title('2D B-Spline Curve with Control Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()