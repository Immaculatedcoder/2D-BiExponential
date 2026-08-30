import numpy as np

# Build the grids
TI = np.logspace(np.log10(20.0), np.log10(4000.0),70)
TE = np.logspace(np.log10(5.0), np.log10(500.0), 30)

# Groudn-truth 
TRUE_PARAMS = {
    "k1": 120.0,
    "k2": 80.0,
    "beta1": 1.95,
    "beta2": 1.95,
    "T11": 1000.0,
    "T12": 2200.0,
    "T21": 80.0,
    "T22": 250.0,
}

# sigma = (k1 + k2) / SNR
SNR = 5000
SIGMA = (TRUE_PARAMS["k1"] + TRUE_PARAMS["k2"]) / SNR

RANDOM_SEED = 42

# Initial Start 
THETA0 = np.array([980.0, 2250.0, 75.0, 260.0], dtype=float)

# Global bounds for theta
THETA_LOWER = np.array([
    500.0,     # T11
    1500.0,    # T12
    20.0,      # T21
    150.0      # T22
], dtype=float)

THETA_UPPER = np.array([
    1500.0,    # T11
    3000.0,    # T12
    150.0,     # T21
    400.0      # T22
], dtype=float)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--print-TE", action="store_true")
    args = parser.parse_args()

    if args.print_TE:
        print(TE)