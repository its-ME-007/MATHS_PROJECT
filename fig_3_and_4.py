# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:31:34 2024

@author: Admin
"""

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# Constants
bv = 0.2  # Keeping bv constant
Br = 0.01
sigma = 2
GR_T = 1
GR_C = 1

# List of bk values to iterate over
bk_values = [-0.5, -0.2, 0, 0.2, 0.5]

# Define the system of differential equations
def equations(y, vars, bk):
    theta, dtheta_dy, u, du_dy = vars
    d2theta_dy2 = (
        bk * (dtheta_dy**2) - Br * (du_dy**2) - (bk - bv) * Br * theta * ((du_dy)**2 + sigma**2 * u**2)
        - bk * bv * Br * theta**2 * ((du_dy)**2 + sigma**2 * u**2) - Br * sigma**2 * u**2
    )
    d2u_dy2 = (
        bv * dtheta_dy * du_dy + sigma**2 * u - (1 + bv * theta) * (GR_T * theta + GR_C * 0.2)
    )
    return [dtheta_dy, d2theta_dy2, du_dy, d2u_dy2]

# Boundary conditions
def boundary_conditions(vars_a, vars_b):
    theta_a, dtheta_dy_a, u_a, du_dy_a = vars_a
    theta_b, dtheta_dy_b, u_b, du_dy_b = vars_b
    m = 1  # Example value for m
    return [u_a, u_b, theta_a - (1 + m), theta_b - 1]

# Initial guess for theta and u profiles
y = np.linspace(-1, 1, 100)
theta_guess = 1 + (y + 1) * 0.1  # Linear guess between boundary values
u_guess = y * 0.0  # Guess u as zero
initial_guess = np.vstack((theta_guess, theta_guess * 0, u_guess, u_guess * 0))

# Plot setup for Theta(y)
plt.figure(figsize=(10, 5))

# Iterate through the different bk values
for bk in bk_values:
    # Solve the boundary value problem
    solution = solve_bvp(lambda y, vars: equations(y, vars, bk), boundary_conditions, y, initial_guess)

    # Check if the solution was successful
    if solution.success:
        # Plot Theta(y)
        plt.plot(solution.x, solution.y[0], label=f"Theta(y) with bk = {bk}")

# Show plot for Theta(y)
plt.xlabel("y")
plt.ylabel("Theta(y)")
plt.title("Solution for Theta as a function of y for different bk values")
plt.grid()
plt.legend()
plt.show()

# Plot setup for u(y)
plt.figure(figsize=(10, 5))

# Iterate again for u(y)
for bk in bk_values:
    # Solve the boundary value problem
    solution = solve_bvp(lambda y, vars: equations(y, vars, bk), boundary_conditions, y, initial_guess)

    # Check if the solution was successful
    if solution.success:
        # Plot u(y)
        plt.plot(solution.x, solution.y[2], label=f"u(y) with bk = {bk}")

# Show plot for u(y)
plt.xlabel("y")
plt.ylabel("u(y)")
plt.title("Solution for phi as a function of y for different bk values")
plt.grid()
plt.legend()
plt.show()
