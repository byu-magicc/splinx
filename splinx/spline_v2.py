"""
Refactored code to be easier to understand.
"""

from jax import jit, random, vmap
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
from jax import vmap, lax, debug
from jax import custom_jvp


@partial(jit, static_argnames=["n"])
def basis_0(t, knots, *, n):
    """
    Given a knot vector and the number of control points n, this 
    computes all values of N_{i,0}(t) for i = 0, 1, ..., n-1.
    """
    condition = (knots[:-1] <= t) & (t < knots[1:])
    return jnp.where(condition, 1.0, 0.0)
    # return 1




if __name__ == "__main__":

    n = 3
    k = 1

    t = 1.0

    # Define the knot vector
    knots = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

    print(f"condition: {(knots[:-1] <= t) & (t < knots[1:])}")

    print(f"basis: {basis_0(t, knots, n=3)}")


    # Plot the basis functions
    # t = jnp.linspace(0.0, 3.0, 1000)
    # y = jnp.array([basis_0(t, knots, i) for i in range(5)])
    # for i in range(5):
    #     plt.plot(t, y[i], label=f"$N_{{i,0}}(t)$")
    # plt.legend()
    # plt.show()

    