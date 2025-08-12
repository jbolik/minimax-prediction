# %%
import jax
import jax.numpy as jnp
from jax.numpy.linalg import slogdet
from jax.scipy.stats import multivariate_normal as mvtn
import matplotlib.pyplot as plt
import numpy as np
from t import t_logpdf
from tqdm import tqdm

# %%
def rbf_kernel(X1: jnp.ndarray,
                X2: jnp.ndarray,
                lengthscale: float) -> jnp.ndarray:
    sqdist = jnp.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
    return jnp.exp(-0.5 * sqdist / (lengthscale ** 2))

# %%
# GP model

def log_pdf(y, X, lengthscale, amplitude, beta):
    return mvtn.logpdf(y, X @ beta, amplitude**2 * rbf_kernel(X, X, lengthscale))

def sample(key, X, lengthscale, amplitude, beta):
    return jax.random.multivariate_normal(key, X @ beta, amplitude**2 * rbf_kernel(X, X, lengthscale))

# %%
# we are working with a fixed lengthscale
lengthscale = 1.0

# %%
from jax.numpy.linalg import inv

# dof=n-p for right-invariant prior and dof=n for Jeffreys prior
def log_pred(yp, Xp, yo, Xo, dof):
    n_obs = len(yo)
    Xo = jnp.vstack((Xo, Xp))
    K = rbf_kernel(Xo, Xo, lengthscale)
    K_inv = inv(K)
    A = K_inv - K_inv @ Xo @ inv(Xo.T @ K_inv @ Xo) @ Xo.T @ K_inv
    Aoo, Aop, Apo, App = A[:n_obs, :n_obs], A[:n_obs, n_obs:], A[n_obs:, :n_obs], A[n_obs:, n_obs:]
    App_inv = inv(App)
    Sigma = (yo.T @ (Aoo - Aop @ App_inv @ Apo) @ yo / dof) * App_inv
    mu = - App_inv @ Apo @ yo
    return t_logpdf(yp, mu, Sigma, dof)

# dof=n-p for unbiased and dof=n for MLE
def log_pred_plug_in(yp, Xp, yo, Xo, dof):
    K = rbf_kernel(Xo, Xo, lengthscale)
    K_inv = inv(K)
    beta_hat = inv(Xo.T @ K_inv @ Xo) @ Xo.T @ K_inv @ yo
    amp_hat = jnp.sqrt((yo - Xo @ beta_hat).T @ K_inv @ (yo - Xo @ beta_hat) / dof)
    # print(beta_hat, amp_hat)
    return log_pdf(yp, Xp, lengthscale, amp_hat, beta_hat)

# %%
# numerically evaluate predictive procedures against knowing true parameters
n, p = 4, 2
Xot = jax.random.uniform(jax.random.PRNGKey(0), (n, p))
Xpt = jax.random.uniform(jax.random.PRNGKey(0), (1, p))
# a, b = jnp.array(2, float), jax.random.uniform(jax.random.PRNGKey(42), (p,))
a, b = jnp.array(3, float), jnp.array([4, -5], float)

@jax.jit
def mc_estimate_risk(key, Xo, Xp, amplitude, beta):
    X = jnp.vstack((Xo, Xp))
    samples = sample(key, X, lengthscale, amplitude, beta)
    yo, yp = samples[:n], samples[n:]
    # conditional covariance in last dimension
    cov = amplitude**2 * rbf_kernel(X, X, lengthscale)
    mean_cond = Xp @ beta + cov[n:, :n] @ inv(cov[:n, :n]) @ (yo - Xo @ beta)
    cov_cond = cov[n:, n:] - cov[n:, :n] @ inv(cov[:n, :n]) @ cov[:n, n:]
    true_pdf = mvtn.logpdf(yp, mean_cond, cov_cond)
    return jnp.array([
        true_pdf - log_pred(yp, Xp, yo, Xo, dof=n-p),
        true_pdf - log_pred(yp, Xp, yo, Xo, dof=n),
        true_pdf - log_pred_plug_in(yp, Xp, yo, Xo, dof=n-p),
        true_pdf - log_pred_plug_in(yp, Xp, yo, Xo, dof=n)])

# a, b = jnp.array(20, float), jnp.array([11, 31], float)


# Xot = jnp.array([[0, 0], [-10, 0], [0, 1], [2, 3]], float)
# Xpt = jnp.array([[1, 1]], float)

# # check invariance of predictive procedures
# yo = jnp.array([1, -2, 3, 5], float)
# yp = jnp.array([2.])
# print(jnp.exp(log_pred_plug_in(yp, Xp, yo, Xo, n-p)))
# print(jnp.exp(log_pred_plug_in(a * yp + Xp @ b, Xp, 2 * yo + Xo @ b, Xo, n-p)))
# print(jnp.exp(log_pred(yp, Xp, yo, Xo, n-p)))
# print(jnp.exp(log_pred(a * yp + Xp @ b, Xp, 2 * yo + Xo @ b, Xo, n-p)))
# # conclusion: the predictive procedures are invariant!

samples = 8192 * 4
iters = 2 ** 12
key = jax.random.PRNGKey(42)
scores, nans = jnp.zeros((4,)), jnp.zeros((4,))
for i in tqdm(range(iters)):
    key_now, key = jax.random.split(key, 2)
    keys = jax.random.split(key_now, samples)
    results = jax.vmap(mc_estimate_risk, (0, None, None, None, None))(keys, Xot, Xpt, a, b)
    scores += jnp.nanmean(results, axis=0)
    nans += jnp.mean(jnp.isnan(results), axis=0)
print(scores / iters, nans / iters)

# %% [markdown]
# The predictive risk doesn't seem to be invariant in the parameters
# For amp=3 beta=[4, -5]: [ 0.42137426  0.7708848   6.611105   13.481195  ] [2.3841858e-07 2.3841858e-07 0.0000000e+00 0.0000000e+00]
# For amp=3 beta=[4, 15]: [1.203939   0.99705505 1.5386127  1.7187254 ] [0. 0. 0. 0.]


