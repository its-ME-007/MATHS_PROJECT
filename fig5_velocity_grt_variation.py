# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:16:13 2024

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# Parameters
bv = 0.5
sigma = 2
GRT_values = [1, 5, 10, 15]  # Varying GRT values
GRC = 1
m = 1

# Boundary conditions
def theta(y):
    """Linear interpolation of theta based on the boundary conditions"""
    return 1 + (m / 2) * (1 - y)

def phi(y):
    """Linear interpolation of phi based on the boundary conditions"""
    return 0.5 * (1 - y)

# Differential equation system
def odes(y, u, GRT):
    """
    u[0] = u1 (original function u)
    u[1] = u2 (first derivative of u)
    """
    dtheta_dy = -m / 2  # derivative of theta with respect to y
    f = np.zeros_like(u)
    f[0] = u[1]
    f[1] = bv * dtheta_dy * u[1] + sigma**2 * u[0] - (1 + bv * theta(y)) * (GRT * theta(y) + GRC * phi(y))
    return f

# Boundary conditions for u
def bc(ua, ub):
    return np.array([ua[0], ub[0]])

# Initial mesh and guess for u
y = np.linspace(-1, 1, 100)
u_guess = np.ones((2, y.size)) * 0.1  # Positive initial guess for u

# Plotting
plt.figure(figsize=(8, 12))  # Adjust figure size for stretched appearance

colors = ['black', 'red', 'green', 'blue']
for i, GRT in enumerate(GRT_values):
    # Solve the boundary value problem
    solution = solve_bvp(lambda y, u: odes(y, u, GRT), bc, y, u_guess)

    # Check if the solution is successful
    if solution.success:
        plt.plot(solution.x, solution.y[0], color=colors[i], label=f'GRT = {GRT}', linewidth=1.5)
        # Add GRT value as text at the peak of each curve
        plt.text(solution.x[np.argmax(solution.y[0])], max(solution.y[0]) + 0.5, f'{GRT}', color=colors[i], fontsize=12)

# Customize plot aesthetics to match the provided image
plt.xlabel('y', fontsize=14)
plt.ylabel('u', fontsize=14, rotation=0, labelpad=20)
plt.title('Equation 5: Dimensionless velocity for varying GRT values', fontsize=14, pad=30)  # Increased 'pad' to raise the title

plt.grid(True)

# Parameter annotations
plt.text(0.6, 10, 'bk=0.2, bv=0.2, m=1\nBr=0.01, GRc=1\nα=0.5, σ=2', fontsize=12)

# Adjust the aspect ratio and scale for a more exaggerated appearance
plt.gca().set_aspect(0.1, 'box')  # Change the aspect ratio for a stretched effect

plt.show()
