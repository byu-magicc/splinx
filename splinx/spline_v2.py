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
from timeit import timeit


# @jit
@partial(jit, static_argnums=(3,))
def basis_i(t, knots, i, k):
    if k == 0:
        condition = (knots[i] <= t) & (t < knots[i+1])
        return jnp.where(condition, 1.0, 0.0)
    else:
        denominator_0 = knots[i+k] - knots[i]
        denominator_1 = knots[i+k+1] - knots[i+1]

        t0 = jnp.where(denominator_0 != 0.0, (t - knots[i]) / (knots[i+k] - knots[i]), 0.0)
        t1 = jnp.where(denominator_1 != 0.0, (knots[i+k+1] - t) / (knots[i+k+1] - knots[i+1]), 0.0)

        return t0 * basis_i(t, knots, i, k-1) + t1 * basis_i(t, knots, i+1, k-1)

basis_vmap = vmap(basis_i, in_axes=(None, None, 0, None), out_axes=0)

# @jit
# def basis(t, knots, k):
#     jax.lax.cond(
#         k == 0,
#         lambda: basis_0(t, knots),
#         lambda: (knots - t) / (basis(t, knots, k-1),
#         operand_stack=[]
#     )


if __name__ == "__main__":

    n = 40
    k = 5 # Compile time gets slow at about 10

    t = 1.5

    # Define the knot vector
    # knots = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]) # Good for n=3, k=1
    # knots = jnp.array([0.0, 0.0, 1.0, 2.0, 3.0, 3.0]) # good for n=3, k=2
    # knots = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]) # good for n=3, k=2
    knots = jnp.arange(n+k+1)

    print(f"condition: {(knots[:-1] <= t) & (t < knots[1:])}")

    print(f"basis_0: {basis_i(t, knots, 0, 0)}")

    print(f"basis_{k}: {basis_i(t, knots, 0, k)}")

    indices = jnp.arange(n)
    print(f"basis: {basis_vmap(t, knots, indices, k)}")

    get_bases = vmap(vmap(basis_i, in_axes=(0, None, None, None)), in_axes=(None, None, 0, None))

    # Plot the basis functions
    t = jnp.linspace(knots[0], knots[-1], 1000).reshape(-1, 1)
    # vals = vmap(vmap(basis_i, in_axes=(0, None, None, None)), in_axes=(None, None, 0, None))(t, knots, indices, k)
    vals = get_bases(t, knots, indices, k)
    for i in range(len(vals)):
        plt.plot(t, vals[i], label=f"$N_{{i,k}}(t)$")
    plt.legend()
    plt.show()

    print(f"Time: {timeit(lambda: get_bases(t, knots, indices, k).block_until_ready(), number=1)}")
    
    for i in range(10):
        print(f"Time at {i}: {timeit(lambda: get_bases(t, knots, indices, k).block_until_ready(), number=1)}")