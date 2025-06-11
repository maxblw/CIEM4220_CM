import numpy as np
import matplotlib.pyplot as plt

def goda_original(h, beta, Hmax, eta, k):
    """
    Calculate the Goda pressure distribution based on the original formula.
    beta = angle of the wave direction
    labda1 = empirical wave coefficient
    labda2 = empirical wave coefficient
    Hmax = wave height
    L = wave length
    rho = density of the water
    eta = surface elevation
    """
    rho = 1025 # density of salt water
    g = 9.81 #m/s^2

    beta = 0
    h_b = h
    h_ = h

    alpha1 = 0.6 + 0.5 * ((2 * k * h) / np.sinh(2 * k * h)) ** 2
    alpha2 = min(((h_b - h) / (3* h_b)) * (Hmax / h)**2, (2*h) / Hmax)
    alpha3 = 1 - (h_ / h) * (1 - (1 / np.cosh(k * h)))
   
    p1 = 1/2 * (np.cos(beta) + 1) * (alpha1 + alpha2 *(np.cos(beta)) ** 2) * rho * g * Hmax
    p3 = alpha3 * p1
    return [p1, p3]

def array_Goda(h_tot, h1, h2, eta, goda, k, H, printing=False):
    """
    Create an array of pressure values based on the Goda formula.
    l: curvature length                 [m]
    h_tot: total height of the caisson  [m]
    h1: SWL                             [m]
    h2: curvature height                [m]  
    eta: surface elevation              [m]
    goda: array with Goda parameters [p1, p3]
    k: wave number                      [1/m]
    """
    # define parameters
    p1 = goda[0]
    # p2 = goda[1]
    p3 = goda[1]
    d = eta + h1

    z = np.linspace(0, h_tot, 1000) # z coordinate from bottom to top of the caisson

    # define the correction factors
    L = (2* np.pi) / k  # wave length
    l = 1.25 # Horizontal curvature length

    p3til = 5848 * H * l / (L ** 2) + 9.75
    p2til = 0.45 * p3til
    p1til = 0
    if printing == True:
        print("h1", h1, "h2", h2, "d", d)
        print("p1", p1, "p3", p3)
        print("p1til", p1til, "p2til", p2til, "p3til", p3til)
    # Define the pressure distribution 
    q_goda = np.zeros(1000)

    Pv = np.zeros(len(z))  # Initialize pressure array
    Pr = np.zeros(len(z))  # Initialize corrected pressure array
    ptil = np.zeros(len(z))  # Initialize pressure correction array

    if d > h2:
        p4 = p1 * (h2 - h1) / (eta)
        Pv[np.where(z < h1)] = np.linspace(p3, p1, np.sum(z < h1))  # linear interpolation from P1 to P2
        Pv[np.where(z >= h1)] = np.linspace(p1, p4, np.sum(z >= h1))  # linear interpolation from P2 to P3

        ptil[np.where(z < h1)] = 1+p1til
        ptil[np.where((z >= h1) & (z < h2))] = np.linspace(1+p1til, 1+p2til, np.sum((z >= h1) & (z < h2)))
        ptil[np.where(z >= h2)] = np.linspace(1+p2til, 1+p3til, np.sum(z >= h2))

        Pr = Pv * ptil

        if printing == True:
            plt.plot(ptil, z, label='Pressure Correction Factor')
            plt.ylabel('Height (m)')
            plt.xlabel('Pressure Correction Factor')
            plt.title('Pressure Correction Factor along the Caisson')
            plt.grid(True)
            plt.show()

            plt.figure(figsize=(5, 5))
            plt.plot(Pr, z, label='Corrected Goda Pressure Distribution')
            plt.ylabel('Height (m)')
            plt.xlabel('Pressure (Pa)')
            plt.title('Corrected Pressure Distribution along the Caisson')
            plt.grid(True)
            plt.legend()

    if d < h2:
        Pv[np.where(z < h1)] = np.linspace(p3, p1, np.sum(z < h1))
        Pv[np.where((z >= h1) & (z < d))] = np.linspace(p1, 0, len(Pv[np.where((z >= h1) & (z < d))]))
        print("Pv", np.linspace(p1, 0, len(np.where((z >= h1) & (z < d)))))
        Pr = Pv.copy()  # No correction for this case

    if printing == True:
        plt.plot(Pv, z, 'r', linewidth = 1, label='Goda Pressure Distribution')
        plt.ylabel('Height (m)')
        plt.xlabel('Pressure (Pa)')
        plt.title('Pressure Distribution along the Caisson')
        plt.grid(True)
        plt.legend()
        plt.savefig('goda_pressure_distribution.png')

    return Pr

def find_dz(z_coords):    
    dz_list = np.zeros(len(z_coords))
    z_coords_temp = points[sea_dofs][:, 1]
    while z_coords_temp.shape[0] > 0:
        z_coord = z_coords_temp.min()
        z_coords_temp = np.delete(z_coords_temp, np.where(z_coords_temp == z_coord)[0][0])
        if z_coords_temp.shape[0] > 0:
            z_coord_i = z_coords_temp.min()
        dz = z_coord_i - z_coord
        dz_list[np.where(z_coords == z_coord)] = dz
    return dz_list

def interp_force(z_coords, q, L):
    n = len(q)
    z = np.linspace(0, L, n)
    q_interp = np.interp(z_coords, z, q)

