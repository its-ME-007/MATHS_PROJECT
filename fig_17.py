# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 22:38:07 2024

@author: Admin
"""


from scipy.integrate import solve_bvp
import numpy as np
import matplotlib.pyplot as plt

def eqn13(y, phi):
    return np.vstack((phi[1], ALPHA*phi[0]))

def boundaryCond(phi_a, phi_b):
    return np.array([phi_a[0] - 1, phi_b[0]])

for ALPHA in (0, 0.5, 1, 1.5):
    y = np.linspace(-1, 1, 100)
    phiInitialGuess = np.zeros((2, y.size))
    solution = solve_bvp(eqn13, boundaryCond, y, phiInitialGuess)
    finalPhiResult = solution.sol(y)[0]
    plt.plot(y, finalPhiResult, label=f"α = {ALPHA}")

plt.xlabel("y")
plt.ylabel(r"$\phi(y)$")
plt.legend()
plt.title(r"Solution of $\frac{d^2 \phi}{dy^2} - \alpha \phi = 0$")
plt.show()
