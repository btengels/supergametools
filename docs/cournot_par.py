#!/usr/bin/python
#----------------------------------------------------------------------------
# This python script uses the supergametools library to find the set of
# supergame equilibria in a simple 2-firm cournot game.
#
# Author: 	Benjamin Tengelsen, btengels@cmu.edu
# Date: 	May 3, 2013
#----------------------------------------------------------------------------

import numpy as np
from itertools import product
import supergametools as sgt
from mpi4py import MPI

# MPI preliminaries
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def payoff(cost, A):
    '''
    This function calculates the payoffs in a cournot duopoly over action space A and
    a unique marginal cost for each player. See supergametheory.pdf for example details.
    '''
    coordinates = product(xrange(len(A)), xrange(len(A)))
    c1, c2 = cost[0], cost[1]

    p1 = np.zeros((len(A), len(A)))
    p2 = np.zeros((len(A), len(A)))

    for index in coordinates:
        j, k = index
        q1 = A[j]
        q2 = A[k]

        p = max(6-q1-q2, 0)

        p1[j, k] = q1 * (p-c1)
        p2[j, k] = q2 * (p-c2)

    return p1, p2


# determine payoff matrices p1 and p2
cost = np.array([0.0, 0.0])
actions = np.linspace(0, 6, 15)
p1, p2 = payoff(cost, actions)

# center and radious of initial guess
cen = np.array([5, 5], ndmin=2)
rad = 10

# find outer hull of set of supergame equilibria
outer_hull = sgt.outerbound_par(p1, p2, cen, rad)

# display results
if rank == 0:
    print outer_hull
