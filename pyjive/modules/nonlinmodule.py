import warnings
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

from numpy.linalg import norm as norm

from names import GlobNames as gn
from names import ParamNames as pn
from names import Actions as act

from modules.module import Module
from modules.controlmodule import ControlModule
from utils.constrainer import Constrainer

ITERMAX = 'itermax'
TOLERANCE = 'tolerance'

class NonlinModule(ControlModule):
    def init(self, props, globdat):
        super().init(props, globdat)
        myprops = props[self._name]
        self._itermax = int(myprops.get(ITERMAX, 100))
        self._tolerance = float(myprops.get(TOLERANCE, 1e-6))

    def run(self, globdat):
        dc = globdat[gn.DOFSPACE].dof_count()
        model = globdat[gn.MODEL]

        K = np.zeros((dc, dc))
        fext = np.zeros(dc)
        fint = np.zeros(dc)
        c = Constrainer(globdat[gn.STATE0])

        params = {pn.MATRIX0: K, pn.EXTFORCE: fext, pn.INTFORCE: fint, pn.CONSTRAINTS: c}

        # Initialize first iteration
        iteration = 0

        # Advance to next time step
        super().advance(globdat)
        model.take_action(act.ADVANCE, params, globdat)

        # Assemble K
        model.take_action(act.GETMATRIX0, params, globdat)

        # Assemble fext
        model.take_action(act.GETEXTFORCE, params, globdat)

        # Get constraints
        model.take_action(act.GETCONSTRAINTS, params, globdat)
        cdofs, cvals = c.get_constraints()
        fdofs = [i for i in range(dc) if i not in cdofs]

        # Calculate residual
        r = fext - fint

        # Constrain K and fext - fint
        Kc, rc = c.constrain(K, r)

        # Sparsify and solve
        smat = sparse.csr_matrix(Kc)
        u = linalg.spsolve(smat, rc)

        # Store solution in Globdat
        globdat[gn.STATE0] += u

        # Reference values to check convergence
        rel = 1
        ref = max(np.linalg.norm(r), np.linalg.norm(fext), 1)

        # Initialize iteration loop
        while rel > self._tolerance and iteration < self._itermax:
            iteration += 1
            params[pn.MATRIX0] = np.zeros((dc, dc))
            params[pn.INTFORCE] = np.zeros(dc)
            model.take_action(act.GETMATRIX0, params, globdat)
            r = fext - params[pn.INTFORCE]
            c.set_zero()
            Kc, rc = c.constrain(params[pn.MATRIX0], r)
            smat = sparse.csr_matrix(Kc)
            du = linalg.spsolve(smat, rc)
            globdat[gn.STATE0] += du
            rel = np.linalg.norm(r[np.ix_(fdofs)]) / ref
            print('Iteration %i, relative residual norm: %.4e' % (iteration, rel))

        # Alert if not convergence
        if rel > self._tolerance:
            if rel > 1:
                raise RuntimeError('Divergence in time step %i' % globdat[gn.TIMESTEP])
            else:
                warnings.warn('No convergence in time step %i' % globdat[gn.TIMESTEP])

        # Check commit
        model.take_action(act.CHECKCOMMIT, params, globdat)

        if globdat[gn.ACCEPTED]:
            print(f"Converged after {iteration} iterations\n")
            model.take_action(act.COMMIT, params, globdat)
        
        return super().run(globdat)

    def shutdown(self, globdat):
        pass


def declare(factory):
    factory.declare_module('Nonlin', NonlinModule)
