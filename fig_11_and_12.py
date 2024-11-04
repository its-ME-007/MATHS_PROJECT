# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 19:16:22 2024

@author: Admin
"""

from scipy.integrate import solve_bvp
import numpy as np
import matplotlib.pyplot as plt

# Constants
bk = 0.2
bv = 0.2
Br = 0.01
GR_T = 1
GR_C = 1
alpha = 0.5
sigma = 2

# Define a list of m values to iterate over
m_values = [-2, -1, 0, 1, 2]

# Define the system of differential equations
def eqn_system(y, phi):
    theta, dtheta_dy, u, du_dy = phi
    d2theta_dy2 = (
        bk * (dtheta_dy**2) - Br * (du_dy**2) - (bk - bv) * Br * theta * ((du_dy)**2 + sigma**2 * u**2)
        - bk * bv * Br * theta**2 * ((du_dy)**2 + sigma**2 * u**2) - Br * sigma**2 * u**2
    )
    d2u_dy2 = (
        bv * dtheta_dy * du_dy + sigma**2 * u - (1 + bv * theta) * (GR_T * theta + GR_C * alpha)
    )
    return np.vstack((dtheta_dy, d2theta_dy2, du_dy, d2u_dy2))

# Define boundary conditions
def boundary_conditions(phi_a, phi_b):
    return np.array([phi_a[2], phi_b[2], phi_a[0] - (1 + m), phi_b[0] - 1])

# Create lists to store solutions for different m values
theta_solutions = []
u_solutions = []

# Loop through m values
for m in m_values:
  # Update m value
  
  # Initial guess for the solution
  y = np.linspace(-1, 1, 100)
  phi_initial_guess = np.zeros((4, y.size))  # Initial guess for [theta, dtheta_dy, u, du_dy]

  # Solve the boundary value problem
  solution = solve_bvp(eqn_system, boundary_conditions, y, phi_initial_guess)

  # Extract the solution for theta and u
  theta_solution = solution.sol(y)[0]
  u_solution = solution.sol(y)[2]

  theta_solutions.append(theta_solution)
  u_solutions.append(u_solution)

# Plot Theta(y) variations
plt.figure()
for i, m in enumerate(m_values):
  plt.plot(y, theta_solutions[i], label=f"m={m}")
plt.xlabel("y")
plt.ylabel(r"$\theta$")
plt.title(r"Temperature Profiles for Different m Values")
plt.legend()
plt.grid(True)
plt.show()

# Plot u(y) variations
plt.figure()
for i, m in enumerate(m_values):
  plt.plot(y, u_solutions[i], label=f"m={m}")
plt.xlabel("y")
plt.ylabel("u")
plt.title(r"Velocity Profiles for Different m Values")
plt.legend()
plt.grid(True)
plt.show()