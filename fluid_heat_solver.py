import numpy as np
import matplotlib.pyplot as plt

def solve_1d_convection_diffusion(L, N, u, alpha, T_left, T_right):
    """
    Solves 1D steady-state advection-diffusion equation for temperature distribution:
    u * (dT/dx) = alpha * (d^2T/dx^2)
    
    Parameters:
    L       : Length of domain (m)
    N       : Number of grid points
    u       : Fluid velocity (m/s)
    alpha   : Thermal diffusivity (m^2/s)
    T_left  : Boundary condition at x = 0 (deg C)
    T_right : Boundary condition at x = L (deg C)
    """
    dx = L / (N - 1)
    x = np.linspace(0, L, N)
    
    # Initialize Coefficient Matrix A and Right-Hand Side Vector b
    A = np.zeros((N, N))
    b = np.zeros(N)
    
    # Peclet Number calculation
    Pe = u * dx / alpha
    print(f"Grid Peclet Number: {Pe:.4f}")
    
    # Boundary Conditions
    A[0, 0] = 1.0
    b[0] = T_left
    A[-1, -1] = 1.0
    b[-1] = T_right
    
    # Interior Nodes (Central Difference Scheme)
    for i in range(1, N - 1):
        # Convection term coefficients
        a_E = alpha / (dx**2) - u / (2 * dx)
        a_W = alpha / (dx**2) + u / (2 * dx)
        a_P = -2 * alpha / (dx**2)
        
        A[i, i-1] = a_W
        A[i, i] = a_P
        A[i, i+1] = a_E
        b[i] = 0.0
        
    # Solve System of Equations
    T = np.linalg.solve(A, b)
    return x, T

if __name__ == "__main__":
    # Parameters definition
    DOMAIN_LENGTH = 1.0      # meters
    GRID_POINTS = 50         # nodes
    FLUID_VELOCITY = 0.5     # m/s
    THERMAL_DIFFUSIVITY = 0.02 # m^2/s
    TEMP_INLET = 100.0       # deg C
    TEMP_OUTLET = 20.0       # deg C

    x, T = solve_1d_convection_diffusion(
        DOMAIN_LENGTH, GRID_POINTS, FLUID_VELOCITY, 
        THERMAL_DIFFUSIVITY, TEMP_INLET, TEMP_OUTLET
    )
    
    # Plotting Results
    plt.figure(figsize=(8, 5))
    plt.plot(x, T, 'b-o', label='Numerical Solution (Central Difference)')
    plt.title('1D Fluid Convection-Diffusion Temperature Profile')
    plt.xlabel('Domain Length x (m)')
    plt.ylabel('Temperature T (°C)')
    plt.grid(True)
    plt.legend()
    plt.savefig('temperature_profile.png')
    plt.show()
