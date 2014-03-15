#!/usr/bin/python
#----------------------------------------------------------------------------
# This python script uses the supergametools library to find the set of
# supergame equilibria in a simple 2-firm bertrand game.
#
# Author: 	Benjamin Tengelsen, btengels@cmu.edu
# Date: 	May 3, 2013
#----------------------------------------------------------------------------

import numpy as np
import supergametools as sgt
from itertools import product


def payoff(cost, A):
    '''
    TODO: fill this
    '''
    coordinates = product(xrange(len(A)), xrange(len(A)))
    c1, c2 = cost[0], cost[1]

    pay1 = np.zeros((len(A), len(A)))
    pay2 = np.zeros((len(A), len(A)))

    for index in coordinates:
        j, k = index
        p1 = A[j]
        p2 = A[k]

        p = min(p1, p2)
        q = 100 - 5*p
        q1 = q/2.0
        q2 = q/2.0

        pay1[j, k] = max(q1*(p-c1), 0)
        pay2[j, k] = max(q2*(p-c2), 0)

    return pay1, pay2


# determine payoff matrices p1 and p2
cost = np.array([0.0, 0.0])
actions = np.linspace(0, 6, 15)
p1, p2 = payoff(cost, actions)

# center and radious of initial guess
cen = np.array([10, 10], ndmin=2)
rad = 300

# find outer hull of set of supergame equilibria
outer_hull = sgt.outerbound(p1, p2, cen, rad, tol=1e-2)

# display results
print outer_hull
