from t import t_logpdf
from scipy.stats import multivariate_t
import numpy as np

def test_t_logpdf():
    # Test parameters
    x = np.array([0.5, -1.2])
    loc = np.array([-1.0, 3.0])
    shape = np.array([[2.0, 0.5], [0.5, 1.0]])
    dof = 5

    # Compute using t_logpdf
    result = t_logpdf(x, loc, shape, dof)

    # Compute using scipy's multivariate t
    scipy_result = multivariate_t.logpdf(x, loc=loc, shape=shape, df=dof)

    # Assert that the results are close
    np.testing.assert_allclose(result, scipy_result, rtol=1e-5, atol=1e-8)

def test_t_logpdf_single_dimension():
    x = np.array([1.0])
    loc = np.array([0.0])
    shape = np.array([[5.0]])
    dof = 10

    result = t_logpdf(x, loc, shape, dof)
    scipy_result = multivariate_t.logpdf(x, loc=loc, shape=shape, df=dof)

    np.testing.assert_allclose(result, scipy_result, rtol=1e-5, atol=1e-8)

def test_t_logpdf_high_dof():
    x = np.array([0.3, -0.7])
    loc = np.array([0.0, 0.0])
    shape = np.array([[2.0, 0.5], [0.5, 1.0]])
    dof = 1000

    result = t_logpdf(x, loc, shape, dof)
    scipy_result = multivariate_t.logpdf(x, loc=loc, shape=shape, df=dof)

    np.testing.assert_allclose(result, scipy_result, rtol=2e-5, atol=1e-8)
