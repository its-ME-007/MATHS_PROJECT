import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
global alpha_iter

# Define the system of differential equations
def eqn_system(y, phi):
    theta, dtheta_dy, u, du_dy = phi
    d2theta_dy2 = (
        bk * (dtheta_dy**2) - Br * (du_dy**2) - (bk - bv) * Br * theta * ((du_dy**2) + sigma**2 * u**2)
        - bk * bv * Br * theta**2 * ((du_dy)**2 + sigma**2 * u**2) - Br * sigma**2 * u**2
    )
    d2u_dy2 = (
        bv * dtheta_dy * du_dy + sigma**2 * u - (1 + bv * theta) * (GR_T * theta + GR_C * alpha)
    )
    return np.vstack((dtheta_dy, d2theta_dy2, du_dy, d2u_dy2))

# Define boundary conditions
def boundary_conditions(phi_a, phi_b):
    return np.array([phi_a[2], phi_b[2], phi_a[0] - (1 + m), phi_b[0] - 1])

# Parameters
bk = 0.2
bv = 0.2
m = 1
Br = 0.01
GR_T = 1
GR_C = 1
sigma = 2
alpha_values = [0, 0.5, 1, 1.5]

# Solve the BVP for different alpha values and store solutions
theta_solutions = []
u_solutions = []
for alpha in alpha_values:
    # Set the alpha value for the current iteration
    alpha_iter = alpha

    # Initial guess for the solution
    y = np.linspace(-1, 1, 100)
    phi_initial_guess = np.zeros((4, y.size))  # Initial guess for [theta, dtheta_dy, u, du_dy]

    # Solve the boundary value problem
    sol = solve_bvp(eqn_system, boundary_conditions, y, phi_initial_guess)

    # Extract the solution for theta and u
    theta_solution = sol.sol(y)[0]
    u_solution = sol.sol(y)[2]

    theta_solutions.append(theta_solution)
    u_solutions.append(u_solution)

# Plot temperature profiles
plt.figure(figsize=(10, 6))
for i, alpha in enumerate(alpha_values):
    plt.plot(y, theta_solutions[i], label=f"alpha={alpha}")
plt.xlabel("y")
plt.ylabel("θ")
plt.title("Temperature Profiles for Different Alpha Values")
plt.legend()
plt.grid(True)
plt.show()

# Plot velocity profiles
plt.figure(figsize=(10, 6))
for i, alpha in enumerate(alpha_values):
    plt.plot(y, u_solutions[i], label=f"alpha={alpha}")
plt.xlabel("y")
plt.ylabel("u")
plt.title("Velocity Profiles for Different Alpha Values")
plt.legend()
plt.grid(True)
plt.show()