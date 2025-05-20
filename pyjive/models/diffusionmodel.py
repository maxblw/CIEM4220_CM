from models.poissonmodel import *

RHO = 'rho'
INITIAL_VALUE = 'initialValue'
INITIAL_RATE = 'initialRate'

class DiffusionModel(PoissonModel):

    def configure(self, props, globdat):
        super().configure(props, globdat)
        self._rho = float(props[RHO])
        self._initialValue = float(props.get(INITIAL_VALUE, 0.))
        self._initialRate = float(props.get(INITIAL_RATE, 0.))

    def take_action(self, action, params, globdat):
        super().take_action(action, params, globdat)       
        if action == act.SETINITIAL:
            self._set_initial(params, globdat)
        if action == act.GETMATRIX1:
            self._get_m_matrix(params, globdat)
    
    def _set_initial(self, params, globdat):
        nodes = np.unique([node for elem in self._elems for node in elem.get_nodes()])
        for doftype in DOFTYPES:
            for node in nodes:
                idof = globdat[gn.DOFSPACE].get_dof(node, doftype)
                globdat[gn.STATE0][idof] = self._initialValue
                globdat[gn.STATE1][idof] = self._initialRate

    def _get_m_matrix(self, params, globdat):
        rho_c = np.array([[self._rho]])

        for elem in self._elems:
            # Get the nodal coordinates of each element
            inodes = elem.get_nodes()
            idofs = globdat[gn.DOFSPACE].get_dofs(inodes, DOFTYPES[0:self._rank])
            coords = np.stack([globdat[gn.NSET][i].get_coords() for i in inodes], axis=1)[0:self._rank, :]

            # Get the shape functions and weights of each integration point
            sfuncs = self._shape.get_shape_functions()
            weights = self._shape.get_integration_weights(coords)

            # Reset the element mass matrix
            elmat = np.zeros((self._dofcount, self._dofcount))

            for ip in range(self._ipcount):
                # Get the N matrix for each integration point
                N = np.zeros((1, self._dofcount))
                N[0, :] = sfuncs[:, ip].transpose()

                # Compute the element mass matrix
                elmat += weights[ip] * np.matmul(np.transpose(N), np.matmul(rho_c, N))

            # Add the element mass matrix to the global mass matrix
            params[pn.MATRIX1][np.ix_(idofs, idofs)] += elmat

def declare(factory):
    factory.declare_model('Diffusion', DiffusionModel)
