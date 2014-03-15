#!/usr/bin/python
#----------------------------------------------------------------------------
# This python script uses the supergametools library to find the set of
# supergame equilibria in a simple 2-person battle of the sexes game.
#
# Author: 	Benjamin Tengelsen, btengels@cmu.edu
# Date: 	May 3, 2013
#----------------------------------------------------------------------------

import numpy as np
import supergametools as sgt

# determine payoff matrices p1 and p2
p1 = np.array([[3, 1], [0, 2]])
p2 = np.array([[2, 1], [0, 3]])

# center and radious of initial guess
cen = np.array([2.5, 2.5], ndmin=2)
rad = 3

# find outer hull of set of supergame equilibria
outer_approx = sgt.outerbound(p1, p2, cen, rad)

print "hello"
inner_approx = sgt.innerbound(p1, p2, cen, rad/10.0)

# display results

