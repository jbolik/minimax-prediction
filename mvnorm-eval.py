import jax
import jax.numpy as jnp
from jax.numpy.linalg import slogdet
from jax.scipy.stats import multivariate_normal as mvtn
import matplotlib.pyplot as plt
import numpy as np
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

@jax.jit
def log_pred(xp, yp, x, y):
    assert len(x) == len(y)
    o = len(y)
    cons = jnp.log(o-2) - jnp.log(2*jnp.pi) + (o-1)/2*jnp.log(o+1) - (o-2)/2*jnp.log(o)
    ones = jnp.ones((o,))
    Gxy1 = jnp.vstack((x, y, ones)) @ jnp.vstack((x, y, ones)).T
    Gy1 = jnp.vstack((y, ones)) @ jnp.vstack((y, ones)).T
    xp, yp = jnp.hstack((x, xp)), jnp.hstack((y, yp))
    onesp = jnp.ones((o + 1,))
    Gxy1p = jnp.vstack((xp, yp, onesp)) @ jnp.vstack((xp, yp, onesp)).T
    Gy1p = jnp.vstack((yp, onesp)) @ jnp.vstack((yp, onesp)).T
    cons_xy = slogdet(Gy1)[1] + (o-2)/2*slogdet(Gxy1)[1]
    main = -(o-1)/2*slogdet(Gxy1p)[1] - slogdet(Gy1p)[1]
    return cons + cons_xy + main

def log_pred_ij(xp, yp, x, y):
    assert len(x) == len(y)
    n = len(y)
    X = jnp.vstack((x, y))
    xyp = jnp.array([xp, yp])
    mean = jnp.mean(X, 1)
    S = (n-1) * jnp.cov(X)
    return t_logpdf(xyp, mean, (n+1)*S/(n*(n-2)), n-2)

def log_pred_j(xp, yp, x, y):
    assert len(x) == len(y)
    n = len(y)
    X = jnp.vstack((x, y))
    xyp = jnp.array([xp, yp])
    mean = jnp.mean(X, 1)
    S = (n-1) * jnp.cov(X)
    return t_logpdf(xyp, mean, (n+1)*S/(n*(n-1)), n-1)

@jax.jit
def log_pred_plug_in(xp, yp, x, y, ddof):
    assert len(x) == len(y)
    X = jnp.vstack((x, y))
    xyp = jnp.array([xp, yp])
    mean = jnp.mean(X, 1)
    S = jnp.cov(X, ddof=ddof)
    return mvtn.logpdf(xyp, mean, S)

def log_pdf(xp, yp, m, n, a, b, c):
    mean = jnp.array([m, n])
    U = jnp.array([[a, b], [0, c]])
    return mvtn.logpdf(jnp.array([xp, yp]), mean, U @ U.T)

def sample(key, m, n, a, b, c):
    mean = jnp.array([m, n])
    U = jnp.array([[a, b], [0, c]])
    return jax.random.multivariate_normal(key, mean, U @ U.T)

o = 3
@jax.jit
def mc_estimate_risk(key, m, n, a, b, c):
    samples = jax.vmap(sample, (0, None, None, None, None, None))(jax.random.split(key, o+1), m, n, a, b, c)
    xp, yp = samples[0, 0], samples[0, 1]
    x, y = samples[1:, 0], samples[1:, 1]
    # return log_pred(xp, yp, x, y) - log_pred_ij(xp, yp, x, y)
    # return log_pred(xp, yp, x, y) - log_pred_plug_in(xp, yp, x, y)
    return (
        # log_pdf(xp, yp, m, n, a, b, c) - log_pred(xp, yp, x, y),
        # log_pdf(xp, yp, m, n, a, b, c) - log_pred_ij(xp, yp, x, y),
        # log_pdf(xp, yp, m, n, a, b, c) - log_pred_j(xp, yp, x, y),
        log_pdf(xp, yp, m, n, a, b, c) - log_pred_plug_in(xp, yp, x, y, 1),
        log_pdf(xp, yp, m, n, a, b, c) - log_pred_plug_in(xp, yp, x, y, 0))

# samples = 8192*256
# keys = jax.random.split(jax.random.PRNGKey(0), samples)
# # results = jax.vmap(mc_estimate_risk, (0, None, None, None, None, None))(keys, 1, 2, 1, -3, 4)
# results = jax.vmap(mc_estimate_risk, (0, None, None, None, None, None))(keys, -3, -2, 5, 2, 2)
# # results = jax.vmap(mc_estimate_risk, (0, None, None, None, None, None))(keys, 7, -5, 8, -8, 3)
# # print(jnp.mean(results))
# for result in results:
#     print("{0:.3g}".format(jnp.nanmean(result)))
# # print(jnp.mean(jax.vmap(mc_estimate_risk, (0, None, None, None, None, None))(keys, -3, 5, 2, 7, 1)))
# # print(jnp.sum(jnp.isnan(jax.vmap(mc_estimate_risk, (0, None, None, None, None, None))(keys, -3, 5, 2, 7, 1))))

x = jnp.array([1, 2, 3], float)
y = jnp.array([-2, 3, 1], float)
X = jnp.vstack((x, y))
mean = jnp.mean(X, 1)

# Generate x and y values
xpts = np.linspace(-10, 10, 500)
ypts = np.linspace(-10, 10, 500)
X, Y = np.meshgrid(xpts, ypts)

# Compute the function values
def f(xp, yp):
    return log_pred(xp, yp, x, y)
Z = jax.vmap(jax.vmap(f, (0, 0)), (0, 0))(X, Y)

# Plot the function
plt.figure(figsize=(7.5, 6))
contour = plt.contourf(X, Y, Z, levels=10, cmap='viridis')
# plt.colorbar(contour, label='log predictive density')
plt.colorbar(contour)
plt.scatter(x, y, c='black')
# plt.scatter(mean[0], mean[1], c='red')
plt.xlabel('x')
plt.ylabel('y')
# plt.axis('equal')  # To maintain aspect ratio
plt.show()