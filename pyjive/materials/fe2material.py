import copy
import sys
from contextlib import redirect_stdout
import numpy as np

import declare
from utils import proputils as pu
from names import GlobNames as gn
from materials.material import Material

FILENAME = 'filename'   # input file for micromechanical analysis
EPS = 1e-8              # small number for numerical differentiation

class FE2Material(Material):

    def __init__ (self,rank):
        pass

    def configure(self, props, globdat):
        self._filename = props.get(FILENAME)

    def create_material_points(self, npoints):
        props = pu.parse_file(self._filename)
        self._globdats = []
        factories = {}
        declare.declare_all(factories)
        modulefac = factories[gn.MODULEFACTORY]
        modelfac = factories[gn.MODELFACTORY]

        self._microchain = []

        for name in props:
            # Get the name of each item in the property file
            if 'type' in props[name]:
                typ = props[name]['type']
            else:
                typ = name.title()

            # If it refers to a module add it to the chain
            if modulefac.is_module(typ):
                self._microchain.append(modulefac.get_module(typ, name))
            elif not modelfac.is_model(typ):
                raise ValueError('%s is neither a module nor a model' % typ)

        for i in range(npoints):
            # setup a separate globdat for each material point
            microglobdat = factories.copy()
            fname = 'micro-'+str(i)+'.log'
            microglobdat['logger'] = open(fname, 'w')

            with redirect_stdout(microglobdat['logger']):
                for module in self._microchain:
                    module.init(props, microglobdat)

            self._globdats.append(microglobdat)

    def update(self,strain,ipoint):
        strcount = len(strain)
        microglobdat = self._globdats[ipoint]
        microglobdat['macrostrain'] = strain
        
        with redirect_stdout(microglobdat['logger']):
            for module in self._microchain:
                module.run(microglobdat)
                sys.stdout.flush()

            stress = microglobdat['homogenization']['stresses'][-1]
            stiff = np.zeros((strcount, strcount))

            for i in range(strcount):
                perturbed = strain.copy()
                perturbed[i] += EPS
                microglobdat['macrostrain'] = perturbed

                microglobdat[gn.TIMESTEP] -= 1
                for module in self._microchain:
                    module.run(microglobdat)

                dstress = microglobdat['homogenization']['stresses'][-1] - stress
                stiff[i] = (1/EPS) * dstress

        return stiff.T, stress

    def commit(self, ipoint=None):
        pass

