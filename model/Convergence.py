import gmsh
import meshio
import numpy as np
from scipy.sparse import linalg
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

import functions as func
import Model
import Load

def mesh_size(lc, time_steps, seaside, roof, top, wall, tip, C, density, body_force, k_bottom, load, amplification, dt, gamma, beta, top_left):
    displacement_lc = []

    # re-run the model for decreasing mesh size (lc)
    for i in range(len(lc)):
        Minterior_tag, sea_side_tag, bottom_side_tag = Model.Mesh(lc[i], lc[i], seaside, roof, top, wall, tip)
        points, triangles, group_bottom, group_sea, bottom_edges, sea_edges, mesh = Model.form_mesh(bottom_side_tag, sea_side_tag)
        K, M, f = Model.BC(mesh, triangles, points, C, density, body_force, bottom_edges, sea_edges, k_bottom, roof, load)
        u_hist = Model.time_discretization(amplification, f, K, M, time_steps, dt, gamma, beta)
        value_left = u_hist[3].flatten()[top_left] # select the deformation at same timestep for ervery lc
        displacement_lc.append(value_left) # storing the values
    return(displacement_lc)

def time_step(T, dt_list, amplification, f, K, M, gamma, beta, top_left):
    displacement_dt = []

    # re-run the model for decreasing timestep dt
    for dt in dt_list:
        time_steps = np.arange(0, T, dt)
        u_hist = Model.time_discretization(amplification, f, K, M, time_steps, dt, gamma, beta)
        value_left = u_hist[-1].flatten()[top_left]
        displacement_dt.append(value_left)
    return displacement_dt