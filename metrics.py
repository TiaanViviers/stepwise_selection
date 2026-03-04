import numpy as np


def _as_1d_float_array(x, name):
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _validate_scalar(value, name, positive=False):
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")


def _validate_model_size(n, k):
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}.")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}.")
    if k >= n:
        raise ValueError(f"k must be < n, got k={k}, n={n}.")


def RSS(Y, Yhat):
    """Residual sum of squares.

    Use case:
    - Compare in-sample fit of models on the same response vector.
    - Lower is better.
    """
    y = _as_1d_float_array(Y, "Y")
    yhat = _as_1d_float_array(Yhat, "Yhat")
    if y.shape != yhat.shape:
        raise ValueError("Y and Yhat must have the same shape.")
    return float((y - yhat).T @ (y - yhat))


def R2(Y, Yhat):
    """Coefficient of determination (in-sample).

    Use case:
    - Standard OLS-style in-sample goodness-of-fit.
    - Can be negative for poor models.
    - Undefined when Y has zero variance.
    """
    y = _as_1d_float_array(Y, "Y")
    yhat = _as_1d_float_array(Yhat, "Yhat")
    if y.shape != yhat.shape:
        raise ValueError("Y and Yhat must have the same shape.")

    y_bar = np.mean(y)
    tss = float((y - y_bar).T @ (y - y_bar))
    if tss <= 0:
        raise ValueError("R2 is undefined when TSS <= 0 (Y has no variance).")

    rss = float((y - yhat).T @ (y - yhat))
    return float(1 - (rss / tss))


def TSS(Y):
    """Total sum of squares around mean(Y).

    Use case:
    - Companion quantity for R2/adjusted R2.
    """
    y = _as_1d_float_array(Y, "Y")
    yc = y - np.mean(y)
    return float(yc @ yc)


def adjusted_R2(rss, tss, n, k):
    """Adjusted R-squared.

    Use case:
    - Model comparison with same response and sample size.
    - Requires n - k - 1 > 0 and tss > 0.
    """
    _validate_model_size(n, k)
    _validate_scalar(rss, "rss")
    _validate_scalar(tss, "tss", positive=True)
    denom = n - k - 1
    if denom <= 0:
        raise ValueError(f"adjusted_R2 undefined for n-k-1 <= 0, got {denom}.")

    r2 = 1 - rss / tss
    return float(1 - (1 - r2) * (n - 1) / denom)


def aic(rss, n, k):
    """Akaike Information Criterion for Gaussian linear models.

    Use case:
    - Relative comparison between models fit to the same data.
    - Lower is better.
    - Undefined when rss <= 0.
    """
    _validate_model_size(n, k)
    _validate_scalar(rss, "rss", positive=True)
    return float(n * np.log(rss / n) + 2 * (k + 1))


def bic(rss, n, k):
    """Bayesian Information Criterion for Gaussian linear models.

    Use case:
    - Relative comparison between models fit to the same data.
    - Lower is better.
    - Undefined when rss <= 0.
    """
    _validate_model_size(n, k)
    _validate_scalar(rss, "rss", positive=True)
    return float(n * np.log(rss / n) + np.log(n) * (k + 1))


def Cp(rss, sigma2, n, k):
    """Mallow's Cp using full-model variance estimate.

    Use case:
    - Subset comparison in linear regression when sigma2 comes from a valid full model.
    - Lower is better.
    - Requires sigma2 > 0.
    """
    _validate_model_size(n, k)
    _validate_scalar(rss, "rss")
    _validate_scalar(sigma2, "sigma2", positive=True)
    return float(rss / sigma2 - (n - 2 * (k + 1)))
