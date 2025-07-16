import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from functools import partial

@jax.jit
def t_logpdf(x, loc, shape, dof):
    dim = len(x)
    dev = x - loc
    L = jax.scipy.linalg.cholesky(shape, lower=True)
    log_pdet = 2 * jnp.sum(jnp.log(jnp.diag(L)))
    y = jnp.vectorize(
        partial(jax.lax.linalg.triangular_solve, lower=True, transpose_a=True),
        signature="(n,n),(n)->(n)"
      )(L, dev)
    maha = jnp.sum(jnp.square(y))

    t = 0.5 * (dof + dim)
    A = gammaln(t)
    B = gammaln(0.5 * dof)
    C = dim/2. * jnp.log(dof * jnp.pi)
    D = 0.5 * log_pdet
    E = -t * jnp.log(1 + (1. / dof) * maha)

    return A - B - C - D + E