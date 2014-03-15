#!/usr/bin/python
#----------------------------------------------------------------------------
# This python script uses the supergametools library to find the set of
# supergame equilibria in a simple 2-person prisoners dilemma.
#
# Author: 	Benjamin Tengelsen, btengels@cmu.edu
# Date: 	May 3, 2013
#----------------------------------------------------------------------------

import numpy as np
import supergametools as sgt
from mpi4py import MPI

# MPI preliminaries
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# determine payoff matrices p1 and p2
p1 = np.array([[4, 0], [6, 2]])
p2 = p1.T

# center and radious of initial guess
cen = np.array([3, 3], ndmin=2)
rad = 5

# find outer hull of set of supergame equilibria
outer_hull = sgt.outerbound_par(p1, p2, cen, rad)

# display results
if rank == 0:
    print outer_hull

