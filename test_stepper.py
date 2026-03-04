import numpy as np
from sklearn.linear_model import LinearRegression

from stepwise_regressor import Stepwise_regression_selector


def main():
    X, Y = generate_data()
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    
    ols = LinearRegression()
    
    aic_fstepper = Stepwise_regression_selector('RSS', 'AIC')
    aic_fstepper.step_forward(X, Y, ols)
    faic = aic_fstepper.selected_subset
    aic_bstepper = Stepwise_regression_selector('RSS', 'AIC')
    aic_bstepper.step_backward(X, Y, ols)
    baic = aic_bstepper.selected_subset
    
    print(f"Forward best subset:\n{faic}\n")
    print(f"Backward best subset:\n{baic}")
    
    
    
    
    
def generate_data():
    # Set up Random number generator
    rng = np.random.default_rng(seed=1)

    # Create epsilon vector
    eps = rng.standard_normal(size=1000)

    # Create X vector
    mu = rng.integers(low=0, high=20, endpoint=True)
    sigma = rng.integers(low=1, high=4, endpoint=True)
    x = rng.normal(loc=mu, scale=sigma, size=1000)

    # Create Y vector
    B_0 = rng.uniform(low=-5, high=5)
    B_1 = rng.uniform(low=-3, high=7)
    B_2 = rng.uniform(low=0, high=4)
    B_3 = rng.uniform(low=-1, high=6)
    
    Y = B_0 + B_1*x + B_2*(x**2) + B_3*(x**3) + eps
    X = np.array([x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10]).T
    
    return X, Y
    

if __name__ == "__main__":
    main()
