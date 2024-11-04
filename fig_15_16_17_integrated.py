
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

def equations(y, X):
    u, du_dy, theta, dtheta_dy, phi, dphi_dy = X

    d2u_dy2 = bv * dtheta_dy * du_dy + sigma**2 * u - (1 + bv * theta) * (GRT * theta + GRC * phi)
    d2theta_dy2 = bk * (dtheta_dy**2) - Br * (du_dy**2) - (bk - bv) * Br * theta * ((du_dy**2) + sigma**2 * u**2) \
                  + bk * bv * Br * theta**2 * ((du_dy**2) + sigma**2 * u**2) - Br * sigma**2 * u**2
    d2phi_dy2 = alpha * phi

    return [du_dy, d2u_dy2, dtheta_dy, d2theta_dy2, dphi_dy, d2phi_dy2]

def boundary_conditions(Xa, Xb):
    u_a, du_dy_a, theta_a, dtheta_dy_a, phi_a, dphi_dy_a = Xa
    u_b, du_dy_b, theta_b, dtheta_dy_b, phi_b, dphi_dy_b = Xb

    return [
        u_a,         # u(-1) = 0
        u_b,         # u(1) = 0
        theta_a - (1 + m),  # theta(-1) = 1 + m
        theta_b - 1, # theta(1) = 1
        phi_a - 1,   # phi(-1) = 1
        phi_b        # phi(1) = 0
    ]

y = np.linspace(-1, 1, 100)
initial_guess = np.zeros((6, y.size))

# Set initial parameters
bk = 0.2
bv = 0.2
m = 1
Br = 0.01
GRT = 1  
GRC = 1  
sigma = 2

# Values of alpha to be varied
alpha_values = [0, 0.5, 1, 1.5]
tolerance = 1e-5

# Containers for solutions
u_results = []
theta_results = []
phi_results = []

# Loop over alpha values and solve the BVP
for alpha in alpha_values:
    solution = solve_bvp(equations, boundary_conditions, y, initial_guess, tol=tolerance)
    if solution.success:
        u_results.append((alpha, solution.y[0]))
        theta_results.append((alpha, solution.y[2]))
        phi_results.append((alpha, solution.y[4]))
    else:
        print("Failed for alpha =", alpha)

# Plot u(y) for different alpha values
for alpha, u in u_results:
    plt.plot(solution.x, u, label=f'alpha = {alpha}')
plt.xlabel("y")
plt.ylabel("u(y)")
plt.title("Solution for u(y) with varying alpha")
plt.grid(True)
plt.legend()
plt.show()

# Plot theta(y) for different alpha values
for alpha, theta in theta_results:
    plt.plot(solution.x, theta, label=f'alpha = {alpha}')
plt.xlabel("y")
plt.ylabel("theta(y)")
plt.title("Solution for theta(y) with varying alpha")
plt.grid(True)
plt.legend()
plt.show()

# Plot phi(y) for different alpha values
for alpha, phi in phi_results:
    plt.plot(solution.x, phi, label=f'alpha = {alpha}')
plt.xlabel("y")
plt.ylabel("phi(y)")
plt.title("Solution for phi(y) with varying alpha")
plt.grid(True)
plt.legend()
plt.show()
