import gmsh
import meshio
import functions as func
import numpy as np
from scipy.sparse import linalg
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

def Mesh(lc_small, lc_big, seaside, roof, top, wall, tip):
    gmsh.initialize()
    gmsh.model.add("Breakwater")

    # Tag offset
    tag = 1

    p_center = gmsh.model.geo.addPoint(seaside+tip, roof-2*tip, 0, lc_small, tag); tag += 1

    # Outer rectangle points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc_big, tag); tag += 1
    p2 = gmsh.model.geo.addPoint(0, top, 0, lc_big, tag); tag += 1
    p3 = gmsh.model.geo.addPoint(wall, top, 0, lc_small, tag); tag += 1
    p4 = gmsh.model.geo.addPoint(wall, roof, 0, lc_small, tag); tag += 1
    p5 = gmsh.model.geo.addPoint(seaside+tip, roof, 0, lc_small, tag); tag += 1
    p6 = gmsh.model.geo.addPoint(seaside+tip, roof-tip, 0, lc_small, tag); tag += 1
    p7 = gmsh.model.geo.addPoint(seaside, roof-2*tip, 0, lc_small, tag); tag += 1
    p8 = gmsh.model.geo.addPoint(seaside, 0, 0, lc_big, tag); tag += 1

    # Outer rectangle lines
    l_a = gmsh.model.geo.addLine(p1, p2)
    l_b = gmsh.model.geo.addLine(p2, p3)
    l_c = gmsh.model.geo.addLine(p3, p4)
    l_d = gmsh.model.geo.addLine(p4, p5)
    l_e = gmsh.model.geo.addLine(p5, p6)
    l_f = gmsh.model.geo.addCircleArc(p6, p_center, p7)
    l_g = gmsh.model.geo.addLine(p7, p8)
    l_h = gmsh.model.geo.addLine(p8, p1)


    # Define surface with a hole
    outer_loop = gmsh.model.geo.addCurveLoop([l_a, l_b, l_c, l_d, l_e, l_f, l_g, l_h])
    interior = gmsh.model.geo.addPlaneSurface([outer_loop])

    gmsh.model.geo.synchronize()

    # # Define physical group for the interior domain
    interior_tag = 3  # Assign a physical group ID
    gmsh.model.addPhysicalGroup(2, [interior], interior_tag)  # 2 corresponds to surface dimension

    sea_side_tag = 1  # Assign a physical group ID
    gmsh.model.addPhysicalGroup(1, [l_f, l_g, l_e], sea_side_tag)  # 1 to the line segment

    # # Define physical boundary for the left vertical side (l4)
    bottom_side_tag = 2  # Assign a physical group ID
    gmsh.model.addPhysicalGroup(1, [l_h], bottom_side_tag)  # 1 to the line segment

    # Generate the mesh
    # gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write("Breakwater.msh")
    return interior_tag, sea_side_tag, bottom_side_tag

def form_mesh(bottom_side_tag, sea_side_tag):
    mesh = meshio.read("Breakwater.msh")

    # Extract node coordinates and triangle connectivity
    points = mesh.points[:, :2]  # (x, y)
    triangles = mesh.get_cells_type("triangle")
    group_bottom = mesh.get_cell_data("gmsh:physical", "line")
    group_sea = mesh.get_cell_data("gmsh:physical", "line")
    # edges = mesh.get_cells_type("line") # edges on the left boundary
    # edges = mesh.cells_dict.get(2, [])  # edges on the left boundary
    bottom_edges = mesh.get_cells_type("line")[group_bottom == bottom_side_tag]
    sea_edges = mesh.get_cells_type("line")[group_sea == sea_side_tag]

    return points, triangles, group_bottom, group_sea, bottom_edges, sea_edges, mesh

def BC(mesh, triangles, points, C, density, body_force, bottom_edges, sea_edges, k_bottom, roof, q_goda ,printing=False):
    # Assemble the global stiffness matrix, mass matrix, and force vector
    K, M, f = func.assemble(mesh, triangles, points, C, density, body_force)

    # Find the nodes on the foundation (bottom) boundary
    bottom_Bnodes = np.where(np.isin(points[:, 1], points[bottom_edges[:, 1], 0]))[0]
    bottom_dofs = np.array([[2*n, 2*n+1] for n in bottom_Bnodes]).flatten()

    # Find the nodes on the seaside boundary
    sea_Bnodes = np.where(np.isin(points[:, 0], points[sea_edges[:, 0], 0]))[0]
    sea_Bnodes = [int(x) for x in sea_Bnodes if x not in bottom_Bnodes] # exlcude bottom node(s) from seaside nodes
    sea_dofs = np.array([[2*n, 2*n+1] for n in sea_Bnodes]).flatten()
    sea_dofs_x = []
    sea_dofs_y = []

    for node in sea_Bnodes:
        sea_dofs_x.append(2 * node)  # x-direction dof
        sea_dofs_y.append(2 * node + 1)  # y-direction dof


    # Apply the stiffness, mass
    for node in bottom_Bnodes:
        dof_x = 2 * node
        dof_y = 2 * node + 1
        K[dof_x, :] = 0  # Set the row to zero
        K[dof_x, dof_x] = 1 # add the bottom stiffness
        M[dof_x, :] = 0  # Set the row to zero
        M[dof_x, dof_x] = 1  # Set the diagonal to 1
        f[dof_x] = 0  # Set the force vector to

        K[dof_y, dof_y] += k_bottom  # Add the bottom stiffness

    n_fp = 1000
    z = np.linspace(0, roof, n_fp)  # Length of the breakwater
    q = -np.array(q_goda)
    

    idx_coords = [int(dof / 2) for dof in sea_dofs_x]  # Convert to indices for points
    z_coords = points[idx_coords][:, 1]  # Extract z-coordinates of the seaside nodes

    q_interp = np.interp(z_coords, z, q)
    if printing == True:
        plt.plot(z, q, label='Force over length')
        plt.plot(z_coords, q_interp, 'ro', label='Interpolated Force')
    dz = z_coords[1] - z_coords[0]

    for dof in sea_dofs_x:
        z_coord = points[int(dof / 2), 1]
        i = np.where(z_coords == z_coord)[0]
        q_comp = q_interp[i]
        F_comp = q_comp * dz
        f[dof] = F_comp 

    return K, M, f

def plot_all(time_steps, u_hist, points):
    # Set number of subplots (adjust cols/rows to your preference)
    num_steps = len(time_steps)
    cols = 4
    rows = int(np.ceil(num_steps / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()  # Flatten in case of 2D array

    for i in range(num_steps):
        ax = axes[i]

        # Reshape displacement
        u_step = u_hist[i].reshape(points.shape)

        # Deformed shape
        deform = points + u_step

        # Plot original and deformed mesh
        ax.scatter(points[:, 0], points[:, 1], c='gray', s=5, alpha=0.3)
        ax.scatter(deform[:, 0], deform[:, 1], c='blue', s=5, alpha=0.8)

        ax.set_title(f'Time Step {i}')
        ax.set_xlim(points[:, 0].min() - 2, points[:, 0].max() + 2)
        ax.set_ylim(points[:, 1].min() - 2, points[:, 1].max() + 2)
        ax.set_aspect('equal')
        ax.axis('off')

    # Hide unused subplots if any
    for i in range(num_steps, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def time_discretization(amplification, f, K, M, time_steps, dt, gamma, beta):
    n_dof = f.shape[0]
    n_steps = len(time_steps)

    u_n = np.zeros(n_dof)
    v_n = np.zeros(n_dof)
    a_n = linalg.spsolve(M, f - K @ u_n)

    bdt2 = beta * dt**2
    inv_bdt2 = 1.0 / bdt2
    inv_beta_dt = 1.0 / (beta * dt)
    inv_2beta = 1.0 / (2 * beta)

    # Precompute effective stiffness matrix
    K_eff = csr_matrix(K + M / bdt2)

    # Preallocate output array
    u_hist = np.empty((n_steps, n_dof // 2, 2))

    for i, t in enumerate(time_steps):
        # Effective force
        f_eff = f + M @ (u_n * inv_bdt2 + v_n * inv_beta_dt + (inv_2beta - 1.0) * a_n)

        # Solve for next displacement
        u_np1 = linalg.spsolve(K_eff, f_eff)

        # Update acceleration and velocity
        a_np1 = inv_bdt2 * (u_np1 - u_n) - inv_beta_dt * v_n - (1.0 - 1.0/(2*beta)) * a_n
        v_np1 = v_n + dt * ((1 - gamma) * a_n + gamma * a_np1)

        # Store results
        u_hist[i] = (u_np1.reshape(-1, 2) * amplification)

        # Prepare for next iteration
        u_n, v_n, a_n = u_np1, v_np1, a_np1

    return u_hist