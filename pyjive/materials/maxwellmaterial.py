# import warnings
from materials.elasticmaterial import ElasticMaterial
from materials.elasticmaterial import ANMODEL_PROP, SOLID, PLANE_STRESS, PLANE_STRAIN, BAR
from utils import proputils as pu

from names import GlobNames as gn

import numpy as np
from math import exp
from math import isnan
import copy

STIFFS = 'prony_stiffs'
TIMES  = 'prony_times'

class MaxwellMaterial(ElasticMaterial):

    def __init__(self, rank):
        super().__init__(rank)

    def configure(self, props, globdat):
        self._globdat = globdat

        super().configure(props,globdat)

        self._stiffs = pu.parse_list(props[STIFFS],float)
        self._times  = pu.parse_list(props[TIMES],float)

        if len(self._stiffs) is not len(self._times):
            raise RuntimeError('MaxwellMaterial: stiffs and relaxation times must have the same size')

        self._oldtime = 0.0

    def update(self, strain, ipoint=None):
        # Compute delta time
        dt = self._globdat[gn.TIME] - self._oldtime

        # Compute strain increment
        self._neweps[:,ipoint] = strain
        deps = self._neweps[:,ipoint] - self._oldeps[:,ipoint]

        # Use parent ElasticMaterial to compute long-term stresses
        stiff_inf, sig_inf = super().update(strain,ipoint)

        # Update Prony elements and get current viscoelastic stress
        stiff_visco, sig_visco = self._prony[ipoint].update(deps,dt)

        # Combine long-term and time-dependent stresses and stiffness
        stiff  = stiff_inf + stiff_visco
        stress = sig_inf + sig_visco

        return stiff, stress

    def commit(self, ipoint=None):
        # Store time and previous strain for next step
        self._oldtime = self._globdat[gn.TIME]
        self._oldeps = np.copy(self._neweps)

        if ipoint:
            self._prony[ipoint].commit()
        else:
            # Commit history for all Prony elements
            for prony in self._prony:
                prony.commit()

    def create_material_points(self, npoints):
        self._oldeps = np.zeros((self._strcount,npoints))
        self._neweps = np.zeros((self._strcount,npoints))

        self._prony = []

        for i in range(npoints):
            self._prony.append(self._PronySeries(self._stiffs,self._times,self._nu,self._strcount,self._anmodel))
        print('Created ', npoints, ' integration point(s).\n')

    class _PronySeries:
        def __init__(self,stiffs,times,poisson,strcount,anmodel):
            # Store analysis model and size of stress/strain vectors
            self._anmodel = anmodel
            self._strcount = strcount

            # Store size of the Prony series
            self._size = len(times)

            # Store Prony series data
            self._stiffs = stiffs
            self._times  = times
            self._nu = poisson

            # Initialize history
            self._oldsig = np.zeros((self._strcount,self._size))
            self._newsig = np.zeros((self._strcount,self._size))

        def update(self, deps, dt):
            stress = np.zeros(self._strcount)
            stiff  = np.zeros((self._strcount,self._strcount))

            # Loop over Prony elements
            for i in range(self._size):
                # Populate stiff and stress here
                # <your code here>

                #

                # Store current Prony stresses
                #self._newsig[:,i] = ???

            return stiff, stress 

        def commit(self):
            # Store history for next step
            self._oldsig = np.copy(self._newsig)

        def _stiffness(self,e):
            stiff = np.zeros((self._strcount,self._strcount))

            if self._anmodel == BAR:
                stiff[0, 0] = e 

            elif self._anmodel == PLANE_STRESS:
                stiff[[0, 1], [0, 1]] = e / (1 - self._nu ** 2)
                stiff[[0, 1], [1, 0]] = (self._nu * e) / (1 - self._nu ** 2)
                stiff[2, 2] = 0.5 * e / (1 + self._nu)

            elif self._anmodel == PLANE_STRAIN:
                d = (1 + self._nu) * (1 - 2 * self._nu)
                stiff[[0, 1], [0, 1]] = e * (1 - self._nu) / d
                stiff[[0, 1], [1, 0]] = e * self._nu / d
                stiff[2, 2] = 0.5 * e / (1 + self._nu)

            elif self._anmodel == SOLID:
                d = (1 + self._nu) * (1 - 2 * self._nu)
                stiff[[0, 1, 2], [0, 1, 2]] = e * (1 - self._nu) / d
                stiff[[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]] = e * self._nu / d
                stiff[[3, 4, 5], [3, 4, 5]] = 0.5 * e / (1 + self._nu)

            return stiff


