import numpy as np

from model import biexponential_2d

def generate_clean_data(TI, TE, params):
    """
    Generate noiseless synthetic 2D biexpnential Data
    """

    D_clean = biexponential_2d(TI, TE, params)
    return D_clean

def add_gaussian_noise(D_clean, sigma, seed=None):
    """
    Add iid Gaussian noise to clean data.
    """

    rng = np.random.default_rng(seed)

    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size = D_clean.shape
    )

    D_noisy = D_clean + noise

    return D_noisy

if __name__ == "__main__":
    from config import TI, TE, TRUE_PARAMS, SIGMA, RANDOM_SEED

    D_clean = generate_clean_data(TI, TE, TRUE_PARAMS)
    D_noisy = add_gaussian_noise(
        D_clean,
        SIGMA,
        RANDOM_SEED
    )


    print(D_noisy.shape)