from jax import jit
from functools import partial
from jax import numpy as jnp
from jax import custom_jvp

@custom_jvp
def f(x, k):
    return k*jnp.sin(x)

@f.defjvp
def f_jvp(primals, tangents):
    x, k = primals
    x_dot, _ = tangents

    y_eval = f(x, k)

    y_eval_dot = k*jnp.cos(x)*x_dot

    return y_eval, y_eval_dot

f_jit = jit(f, static_argnames=["k"])

if __name__=='__main__':
    x = jnp.linspace(-1,1,100)
    f_jit(x=x, k=2)
