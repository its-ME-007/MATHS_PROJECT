import numpy as np
from scipy.integrate import solve_bvp
#import matplotlib.pyplot as plt

# Define constants
bv = 0.2  
Br = 0
sigma = 2
GR_T = 1
GR_C = 1
bk_values = [0, 0.3, 0.6]

# Define differential equations
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

# Define boundary conditions
def boundary_conditions(vars_a, vars_b):
    theta_a, dtheta_dy_a, u_a, du_dy_a = vars_a
    theta_b, dtheta_dy_b, u_b, du_dy_b = vars_b
    m = 1  
    return [u_a, u_b, theta_a - (1 + m), theta_b - 1]

# Points at which to evaluate and print the solution
evaluation_points = np.array([1, 0.6, 0.2, -0.2, -0.6, -1])

# Function to print values at specified points
def print_values_at_points(solution, points):
    theta_values = solution.sol(points)[0]  # Theta (temperature) values at specified points
    u_values = solution.sol(points)[2]      # u (velocity) values at specified points
    
    print(f"\nValues at specified points (bk = {bk}):")
    for i, point in enumerate(points):
        print(f"At y = {point}: Theta = {theta_values[i]:.4f}", end="")
        if bk == 0:
            print(f", u = {u_values[i]:.4f}")  # Print u only when bk = 0
        else:
            print("")  # Just print theta for other bk values
# Discretize y and initial guess
y = np.linspace(-1, 1, 100)
theta_guess = 1 + (y + 1) * 0.1  
u_guess = y * 0.0
initial_guess = np.vstack((theta_guess, theta_guess * 0, u_guess, u_guess * 0))

# Solve and plot for different bk values
for bk in bk_values:
    solution = solve_bvp(lambda y, vars: equations(y, vars, bk), boundary_conditions, y, initial_guess)
    
    if solution.success:
        print(f"\nResults for bk = {bk}:")
        print_values_at_points(solution, evaluation_points)  # Print theta and u at specified points

        '''# Plot Theta
        plt.figure(1)
        plt.plot(solution.x, solution.y[0], label=f"Theta(y) with bk = {bk}")

        # Plot u
        plt.figure(2)
        plt.plot(solution.x, solution.y[2], label=f"u(y) with bk = {bk}")

# Final plot adjustments for Theta
plt.figure(1)
plt.xlabel("y")
plt.ylabel("Theta(y)")
plt.title("Solution for Theta as a function of y for different bk values")
plt.grid()
plt.legend()
plt.show()

# Final plot adjustments for u
plt.figure(2)
plt.xlabel("y")
plt.ylabel("u(y)")
plt.title("Solution for u as a function of y for different bk values")
plt.grid()
plt.legend()
plt.show()
'''