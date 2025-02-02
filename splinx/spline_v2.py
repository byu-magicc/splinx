"""
Refactored code to be easier to understand.
"""

import jax
from jax import jit, random, vmap
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
from jax import vmap, lax, debug
from jax import custom_jvp


# @partial(jit, static_argnames=["n"])
@jit
def basis_0(t, knots):
    """
    Given a knot vector and the number of control points n, this 
    computes all values of N_{i,0}(t) for i = 0, 1, ..., n-1.
    """
    condition = (knots[:-1] <= t) & (t < knots[1:])
    return jnp.where(condition, 1.0, 0.0)


# @jit
@partial(jit, static_argnames=["k"])
def basis_i(t, knots, i, *, k):
    if k == 0:
        return basis_0(t, knots)
    else:
        t0 = (t - knots[i]) / (knots[i+k] - knots[i])
        t1 = (knots[i+k+1] - t) / (knots[i+k+1] - knots[i+1])
        return t0 * basis_i(t, knots, i, k=k-1) + t1 * basis_i(t, knots, i+1, k=k-1)
    # jax.lax.cond(
    #     k == 0, 
    #     lambda: basis_0(t, knots),
    #     lambda: (t - knots[i]) / (knots[i+k] - knots[i]) * basis_i(t, knots, i, k=k-1) + (knots[i+k+1] - t) / (knots[i+k+1] - knots[i+1]) * basis_i(t, knots, i+1, k=k-1)
    #     # return t0 * basis_i(t, knots, i, k-1) + t1 * basis_i(t, knots, i+1, k-1)
    # )

# @jit
# def basis(t, knots, k):
#     jax.lax.cond(
#         k == 0,
#         lambda: basis_0(t, knots),
#         lambda: (knots - t) / (basis(t, knots, k-1),
#         operand_stack=[]
#     )


if __name__ == "__main__":

    n = 3
    k = 1

    t = 1.0

    # Define the knot vector
    knots = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

    print(f"condition: {(knots[:-1] <= t) & (t < knots[1:])}")

    print(f"basis_0: {basis_0(t, knots)}")

    k = 2
    print(f"basis_{k}: {basis_i(t, knots, 0, k=k)}")


    # Plot the basis functions
    # t = jnp.linspace(0.0, 3.0, 1000)
    # y = jnp.array([basis_0(t, knots, i) for i in range(5)])
    # for i in range(5):
    #     plt.plot(t, y[i], label=f"$N_{{i,0}}(t)$")
    # plt.legend()
    # plt.show()

    