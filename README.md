# Runge-Kutta Shooting Method for Solving Nonlinear Differential Equations

This project demonstrates the application of the **Runge-Kutta shooting method** for solving complex nonlinear boundary value problems (BVPs) involving coupled differential equations. The equations are solved using Python's `scipy` library, and results are visualized with `matplotlib`.

## Problem Description

The study involves solving equations that describe the **combined effects of variable viscosity and thermal conductivity** on double-diffusive convection flow of a permeable fluid in a vertical channel. These coupled nonlinear differential equations stem from the governing laws of conservation of momentum, energy, and species concentration.

The equations include:

1. **Momentum Equation**:
   \[
   \frac{d}{dy}\left(\mu \frac{dU}{dy}\right) - \mu \frac{U}{\kappa} + \rho_0 g \beta_T (T - T_0) + \rho_0 g \beta_C (C - C_0) = 0
   \]

2. **Energy Equation**:
   \[
   \frac{d}{dy}\left(K \frac{dT}{dy}\right) + \mu \left(\frac{dU}{dy}\right)^2 + \mu \frac{U^2}{\kappa} = 0
   \]

3. **Concentration Equation**:
   \[
   D \frac{d^2C}{dy^2} - \gamma C = 0
   \]

Boundary conditions are applied to ensure physical relevance at the channel walls.

## Key Features of the Project

1. **Transformation to Initial Value Problems**:
   The shooting method is employed to convert boundary value problems into initial value problems by guessing unknown boundary conditions iteratively.

2. **Runge-Kutta Method**:
   The fourth-order Runge-Kutta method is used for numerical integration of the resulting IVPs.

3. **Parameter Exploration**:
   The effects of physical parameters such as variable viscosity, thermal conductivity, thermal Grashof number, mass Grashof number, Brinkman number, and chemical reaction rates are analyzed.

## Python Implementation

The implementation is divided into three scripts:

1. **`1_to_16.py`**:
   - Solves BVPs for multiple test cases, iterating over a range of parameter values.
   - Employs `scipy.integrate.solve_ivp` for numerical integration.

2. **`17_only.py`**:
   - Focuses on a specific complex case with unique parameter settings.

3. **`table2_verify.py`**:
   - Verifies numerical solutions against tabulated or analytical values to ensure accuracy.

## How to Run

1. Install dependencies:
   ```bash
   pip install numpy scipy matplotlib
   ```

2. Run the scripts:
   ```bash
   python 1_to_16.py
   python 17_only.py
   python table2_verify.py
   ```

   Each script outputs results to the console and generates plots illustrating the solution profiles for velocity, temperature, and concentration.

## Visualization

The results include:
- Profiles of velocity, temperature, and concentration as functions of spatial coordinate.
- Analysis of the effects of changing physical parameters on flow behavior.

## Insights

This study provides insights into:
1. How variable viscosity and thermal conductivity impact convection flows in porous media.
2. The role of Brinkman and Grashof numbers in enhancing or suppressing flow and heat transfer.

## References

This work draws on numerical methods for solving BVPs and builds on theoretical insights from heat and mass transfer in fluid systems. Notable references include:
- Attia (2006) on temperature-dependent viscosity and thermal conductivity.
- Seddeek (2005) on chemical reactions and variable fluid properties.


