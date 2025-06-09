import numpy as np

def GaussPoints(points=1):
    if points == 1:
        # 1-point quadrature
        qp = np.array([[1/3, 1/3]])
        weights = np.array([0.5])
    elif points == 3:
        qp = np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]])
        weights = np.array([1/6, 1/6, 1/6]) 
    else:
        raise NotImplementedError("Only order 1 and 2 implemented")
    return qp, weights

# Defining af function for the creation of the shape functions
def shape_functions(qp):
    # Linear shape functions on triangle
    N = []
    dN_dxi = []
    for xi, eta in qp:
        N.append(np.array([1 - xi - eta, xi, eta]))
        dN_dxi.append(np.array([[-1, -1], [1, 0], [0, 1]]))
    return np.array(N), np.array(dN_dxi)

# Function to define the elsaticity tensor
def elasticity_tensor(E, nu, plane_stress=True):
    if plane_stress:
        C = E / (1 - nu**2) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu)/2]
        ])
    else:  # Plane strain
        C = E / ((1 + nu)*(1 - 2*nu)) * np.array([
            [1 - nu, nu, 0],
            [nu, 1 - nu, 0],
            [0, 0, (1 - 2*nu)/2]
        ])
    return C

# Function to define the element stiffness matrix
def element_stiffness(coords, C, qp, weights, dN_dxi):
    Ke = np.zeros((6, 6))
    for i, (xi_eta, w) in enumerate(zip(qp, weights)):
        J = coords.T @ dN_dxi[i]
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)
        dN_dx = dN_dxi[i] @ invJ

        B = np.zeros((3, 6))
        for a in range(3):
            B[0, 2*a] = dN_dx[a, 0]
            B[1, 2*a+1] = dN_dx[a, 1]
            B[2, 2*a] = dN_dx[a, 1]
            B[2, 2*a+1] = dN_dx[a, 0]

        Ke += B.T @ C @ B * detJ * w
    return Ke

# Function to compute the element mass per element
def element_mass(coords, density, N, weights, dN_dxi):
    Me = np.zeros((6, 6))
    for i, w in enumerate(weights):
        Ni = N[i]
        Ni_matrix = np.zeros((2, 6))
        Ni_matrix[0, 0::2] = Ni
        Ni_matrix[1, 1::2] = Ni

        J = coords.T @ dN_dxi[i]
        detJ = np.linalg.det(J)

        Me += density * (Ni_matrix.T @ Ni_matrix) * detJ * w
    return Me

def element_force(coords, body_force, N, weights, dN_dxi):
    fe = np.zeros(6)
    for i, w in enumerate(weights):
        Ni = N[i]
        J = coords.T @ dN_dxi[i]
        detJ = np.linalg.det(J)
        for a in range(3):
            fe[2*a] += Ni[a] * body_force[0] * detJ * w
            fe[2*a+1] += Ni[a] * body_force[1] * detJ * w
    return fe

def assemble(mesh, elements, coords, C, density, body_force):
    # Calculate the total number of degrees of freedom (DOFs)
    num_dofs = coords.shape[0] * 2

    # Initialize global stiffness matrix, mass matrix, and force vector
    K = np.zeros((num_dofs, num_dofs))  # Stiffness matrix
    M = np.zeros((num_dofs, num_dofs))  # Mass matrix
    f = np.zeros(num_dofs)  # Force vector

    # Get quadrature points and weights for the triangle element
    qp, weights = GaussPoints()

    # Get shape functions and their derivatives in reference coordinates
    N, dN_dxi = shape_functions(qp)

    # Loop over all elements in the mesh
    for elem in elements:
        # Extract coordinates of the nodes for this element
        nodal_coords = coords[elem]
        
        # Define the degrees of freedom (DOFs) corresponding to the element's nodes
        dofs = np.array([[2*n, 2*n+1] for n in elem]).flatten()

        # Compute element stiffness matrix (Ke), mass matrix (Me), and force vector (fe)
        Ke = element_stiffness(nodal_coords, C, qp, weights, dN_dxi)
        Me = element_mass(nodal_coords, density, N, weights, dN_dxi)
        fe = element_force(nodal_coords, body_force, N, weights, dN_dxi)

        # Assemble element contributions into global matrices and vector
        for i in range(6):
            for j in range(6):
                # Add the contributions to the global stiffness and mass matrices
                K[dofs[i], dofs[j]] += Ke[i, j]
                M[dofs[i], dofs[j]] += Me[i, j]
            
            # Add the force vector contributions to the global force vector
            f[dofs[i]] += fe[i]
    
    # Return the assembled global stiffness matrix, mass matrix, and force vector
    return K, M, f

