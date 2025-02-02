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
def basis(t, knots, i, k):
    if k == 0:
        condition = (knots[i] <= t) & (t < knots[i+1])
        return jnp.where(condition, 1.0, 0.0)
    else:
        denominator_0 = knots[i+k] - knots[i]
        denominator_1 = knots[i+k+1] - knots[i+1]

        t0 = jnp.where(denominator_0 != 0.0, (t - knots[i]) / (knots[i+k] - knots[i]), 0.0)
        t1 = jnp.where(denominator_1 != 0.0, (knots[i+k+1] - t) / (knots[i+k+1] - knots[i+1]), 0.0)

        return t0 * basis(t, knots, i, k-1) + t1 * basis(t, knots, i+1, k-1)

basis_vmap = vmap(basis, in_axes=(None, None, 0, None), out_axes=0)



if __name__ == "__main__":

    n = 40
    k = 5 # Compile time gets slow at about 10

    t = 1.5

    # Define the knot vector
    knots = jnp.arange(n+k+1)

    indices = jnp.arange(n)
    print(f"basis: {basis_vmap(t, knots, indices, k)}")


    # Function that vmaps over both all basis functions and an input array of t values
    get_bases = vmap(vmap(basis, in_axes=(0, None, None, None)), in_axes=(None, None, 0, None))

    # Plot the basis functions
    t = jnp.linspace(knots[0], knots[-1], 1000).reshape(-1, 1)
    # vals = vmap(vmap(basis_i, in_axes=(0, None, None, None)), in_axes=(None, None, 0, None))(t, knots, indices, k)
    vals = get_bases(t, knots, indices, k)
    for i in range(len(vals)):
        plt.plot(t, vals[i], label=f"$N_{{{i},{k}}}(t)$")
    plt.legend()
    plt.show()

    # Benchmark the runtime _after_ it is JIT compiled.
    print(f"Time: {timeit(lambda: get_bases(t, knots, indices, k).block_until_ready(), number=1)}")
    
    for i in range(10):
        print(f"Time at {i}: {timeit(lambda: get_bases(t, knots, indices, k).block_until_ready(), number=1)}")