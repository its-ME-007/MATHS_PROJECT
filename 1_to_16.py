import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# Differential equations
def equations(y, X):
    u, du_dy, theta, dtheta_dy, phi, dphi_dy = X
    d2u_dy2 = bv * dtheta_dy * du_dy + sigma**2 * u - (1 + bv * theta) * (GRT * theta + GRC * phi)
    d2theta_dy2 = bk * (dtheta_dy**2) - Br * (du_dy**2) - (bk - bv) * Br * theta * ((du_dy**2) + sigma**2 * u**2) \
                  + bk * bv * Br * theta**2 * ((du_dy**2) + sigma**2 * u**2) - Br * sigma**2 * u**2
    d2phi_dy2 = alpha * phi
    return [du_dy, d2u_dy2, dtheta_dy, d2theta_dy2, dphi_dy, d2phi_dy2]

# Boundary conditions
def boundary_conditions(Xa, Xb):
    u_a, du_dy_a, theta_a, dtheta_dy_a, phi_a, dphi_dy_a = Xa
    u_b, du_dy_b, theta_b, dtheta_dy_b, phi_b, dphi_dy_b = Xb
    return [
        u_a,           # u(-1) = 0
        u_b,           # u(1) = 0
        theta_a - (1 + m),  # theta(-1) = 1 + m
        theta_b - 1,   # theta(1) = 1
        phi_a - 1,     # phi(-1) = 1
        phi_b          # phi(1) = 0
    ]

# Parameters
y = np.linspace(-1, 1, 100)
initial_guess = np.zeros((6, y.size))

# Variable configurations with added value 20 for GRT
variable_data = {
    'bv': [-0.5, -0.2, 0, 0.2, 0.5],
    'bk': [-0.5, -0.2, 0, 0.2, 0.5],
    'GRT': [1, 5, 10, 15], 
    'GRC': [1, 5, 10, 15, 20],
    'Br': [0.1, 0.5, 1, 1.5, 2],
    'm': [-2, -1, 0, 1, 2],
    'sigma': [0.5, 1, 2, 4],
    'alpha': [0, 0.5, 1, 1.5]
}

# Default parameters
default_values = {
    'bk': 0.2,
    'bv': 0.2,
    'm': 1,
    'Br': 0.01,
    'GRT': 1,
    'GRC': 1,
    'sigma': 2,
    'alpha': 0.5,
    'tolerance': 1e-5
}

# Iterate over variable configurations
for param_name, values in variable_data.items():
    # Reset parameters to defaults at the start of each loop
    for key, val in default_values.items():
        locals()[key] = val
    tolerance = default_values['tolerance']

    u_results = []
    theta_results = []
    phi_results = []

    for value in values:
        # Set the parameter dynamically and reset the initial guess
        locals()[param_name] = value
        initial_guess = np.zeros((6, y.size))

        # Adjust tolerance if necessary
        if param_name == 'GRT':
            tolerance = 1e-2  # May need to loosen tolerance for larger GRT values
        elif param_name == 'Br':
            tolerance = 1e-1
            bk, bv = 0.5, -0.2

       # print(f"Solving for {param_name} = {value} with tolerance = {tolerance}")

        solution = solve_bvp(equations, boundary_conditions, y, initial_guess, tol=tolerance)

        if solution.success:
            u_results.append((value, solution.y[0]))
            print(f"du/dy values for {param_name} = {value} ")
            print(" "*6 + "at y=-1" + " "*13 +"at y=1" + " "*6)
            print(solution.y[1,0], end = '  ')
            print(solution.y[1,-1])
            print(f"dtheta/dy values for {param_name} at {value}")
            print(" "*6 + "at y=-1" + " "*13 +"at y=1" + " "*6)
            print(solution.y[3,0], end = '  ')
            print(solution.y[3,-1])
            print ("-"*20)
            theta_results.append((value, solution.y[2]))
            #phi_results.append((value, solution.y[4]))
        else:
            print(f"Solution failed for {param_name} = {value}: {solution.message}")
            continue

    # Plotting
    for results, ylabel, title in [
        (u_results, "u(y)", f"Solution for u(y) with varying {param_name}"),
        (theta_results, "theta(y)", f"Solution for theta(y) with varying {param_name}"),
        #(phi_results, "phi(y)", f"Solution for phi(y) with varying {param_name}")
    ]:
        plt.figure(figsize=(8, 5))
        for value, result in results:
            if len(solution.x) == len(result):  # Ensure dimensions match
                plt.plot(solution.x, result, label=f'{param_name} = {value}')
        plt.xlabel("y")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()
