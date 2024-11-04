import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# Parameters
sigma = 2
GRT = 1
GRC = 1
m = 1
bv_values = [-0.5, -0.2, 0, 0.2, 0.5]  # Different values of bv to iterate over

# Boundary conditions
def theta(y):
    """ Linear interpolation of theta based on the boundary conditions """
    return 1 + (m / 2) * (1 - y)

def phi(y):
    """ Linear interpolation of phi based on the boundary conditions """
    return 0.5 * (1 - y)

# Differential equation system
def odes(y, u, bv):
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
u_guess = np.ones((2, y.size)) * 0.1  # Initial guess for u

# Plot setup
plt.figure(figsize=(10, 6))
plt.xlabel('y')
plt.ylabel('u(y)')
plt.title('Solutions for Different Values of bv')
plt.grid()

# Solve and plot for each bv value
for bv in bv_values:
    # Define the ODE function specific to the current bv
    def odes_current(y, u):
        return odes(y, u, bv)

    # Solve the boundary value problem
    solution = solve_bvp(odes_current, bc, y, u_guess)

    # Plot if the solution is successful
    if solution.success:
        plt.plot(solution.x, solution.y[0], label=f'bv = {bv}')
    else:
        print(f"Solver did not converge for bv = {bv}")

# Add legend and show plot
plt.legend()
plt.show()
