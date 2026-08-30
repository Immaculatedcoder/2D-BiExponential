"""
    S(TI, TE) =     k1 * (1.0 - beta1 * np.exp(-TI_grid/ T11)) 
        * np.exp(-TE_grid / T21) + 
        k2 * (1.0 - beta2 * np.exp(-TI_grid/ T12)) 
                * np.exp(-TE_grid / T22)
"""

from __future__ import annotations

import numpy as np

def biexponential_2d(TI, TE, params):
    """
        Evaluate the 2D biexponetial MRI Model

        Parameters
        ----------
        TI : array
            Inversion times
        TE : array
            Echo times
        params: dict
            Model parameters:
            k1, k2, beta1, beta2,
            T11, T12, T21, T22
        
        Returns
        ------
        S: ndarray
            Signals evaluated on the TI x TE grid.
    """

    k1 = params["k1"]
    k2 = params["k2"]

    beta1 = params["beta1"]
    beta2 = params["beta2"]

    T11 = params["T11"]
    T12 = params["T12"]

    T21 = params["T21"]
    T22 = params["T22"]

    TI_grid, TE_grid = np.meshgrid(TI, TE,indexing="ij")

    component1 = (
        k1 * (1.0 - beta1 * np.exp(-TI_grid/ T11)) 
        * np.exp(-TE_grid / T21)
    )
    component2 = (
        k2 * (1.0 - beta2 * np.exp(-TI_grid/ T12)) 
        * np.exp(-TE_grid / T22)
    )

    S = component1 + component2

    return S

if __name__ == "__main__":
    import argparse
    from config import TI, TE, TRUE_PARAMS


    parser = argparse.ArgumentParser()
    parser.add_argument("--print-S", action="store_true")
    args = parser.parse_args()

    if args.print_S:
        S = biexponential_2d(TI, TE, TRUE_PARAMS)
        print(S.shape)