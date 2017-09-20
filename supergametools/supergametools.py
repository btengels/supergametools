#! usr/bin/python
#-------------------------------------------------------------------------------
# Author: Benjamin Tengelsen, Carnegie Mellon University, btengels@cmu.edu
# This file computes an outerbound approximation for the set of supergame
# equilibria in a 2-player infinitely repeated game.
# It is based on the algorithm given in Judd, Yeltekin, Conklin (2003)
#-------------------------------------------------------------------------------

# TODO: errors for inputs for bad Hausdorff inputs
# TODO: errors for inputs for bad cylinder inputs
# TODO: rename b in outerbound

import time
import numpy as np
import cvxopt as cvx
import matplotlib.pyplot as plt
from cvxopt import solvers as solvers


__all__ = ['outerbound', 'outerbound_par', 'innerbound', 'innerbound_par', 'hausdorffnorm']     # this list gives what is imported with "from supergametools import *"


def _cylinder(r, n):
    '''
    Returns the unit cylinder that corresponds to the curve r.
    INPUTS:
    r : a vector of radii
    n : number of coordinates to return for each element in r
    OUTPUTS:
    x, y, z: coordinates of points around cylinder
    '''

    # ensure that r is a column vector
    r = np.atleast_2d(r)
    r_rows, r_cols = r.shape

    if r_cols > r_rows:
        r = r.T

    # find points along x and y axes
    points = np.linspace(0, 2*np.pi, n+1)
    x = np.cos(points)*r
    y = np.sin(points)*r

    # find points along z axis
    rpoints = np.atleast_2d(np.linspace(0, 1, len(r)))
    z = np.ones((1, n+1))*rpoints.T

    return x, y, z


def hausdorffnorm(A, B):
    '''
    Finds the hausdorff norm between two matrices A and B.
    INPUTS:
    A: numpy array
    B : numpy array
    OUTPUTS:
    Housdorff norm between matrices A and B
    '''
    # ensure matrices are 3 dimensional, and shaped conformably
    if len(A.shape) == 1:
        A = np.atleast_2d(A)

    if len(B.shape) == 1:
        B = np.atleast_2d(B)

    A = np.atleast_3d(A)
    B = np.atleast_3d(B)

    x, y, z = B.shape
    A = np.reshape(A, (z, x, y))
    B = np.reshape(B, (z, x, y))

    # find hausdorff norm: starting from A to B
    z, x, y = B.shape
    temp1 = np.tile(np.reshape(B.T, (y, z, x)), (max(A.shape), 1))
    temp2 = np.tile(np.reshape(A.T, (y, x, z)), (1, max(B.shape)))
    D1 = np.min(np.sqrt(np.sum((temp1-temp2)**2, 0)), axis=0)

    # starting from B to A
    temp1 = np.tile(np.reshape(A.T, (y, z, x)), (max(B.shape), 1))
    temp2 = np.tile(np.reshape(B.T, (y, x, z)), (1, max(A.shape)))
    D2 = np.min(np.sqrt(np.sum((temp1-temp2)**2, 0)), axis=0)

    return np.max([D1, D2])


def _loadbalance(n, p):
    '''
    This function assists with load balancing the parallel functions of supergametools.
    It determines how many search gradients are allocated to each process.
    INPUTS:
    n       number of search gradients, int.
    p       number of processes, int.
    OUTPUTS:
    load    list with the number of gradients assigned to each process, list.
    '''
    load = []
    inc = n/p
    R = n - inc*p

    load = [inc for i in range(p)]

    for i in range(int(R)):
        load[i] += 1

    load.append(0)
    load.sort()
    return load


def _perp(a):
    '''
    supports _seg_intersect
    '''
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def _seg_intersect(a1, a2, b1, b2):
    '''
    line segment a given by endpoints a1, a2
    line segment b given by endpoints b1, b2
    '''
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = _perp(da)
    denom = np.dot(dap, db)
    num = float(np.dot(dap, dp))
    return (num / denom)*db + b1


def innerbound(p1, p2, cen, rad, n_grad=8, delta=0.8, plot=True, tol=1e-4, MaxIter=200, display=True, Housdorff=False):
    '''
    This method computes the innerbound approximation of Judd, Yeltekin, Cronklin (2003)
    for 2 agents.
    INPUTS:
    p1:         payoff matrix for player 1. numpy array(n,n, ndim=2)
    p2:         payoff matrix for player 2. numpy array(n,n, ndim=2)
    cen:        center for initial guess, a circle. numpy.array (2,1, ndim=2)
    rad:        radius for initial guess, a circle. float.
    n_grad:     number of search gradients. int.
    delta:      discount factor. float.
    plot:       True will generate plots. boolean.
    tol:        Minimum tolerable convergence error. float.
    MaxIter:    Maximum number of iterations allowed. int.
    display:    Option to display output during iterations. boolean.
    Hausdorff:  Option to measure error using Hausdorff norm instead of standard relative error measure. boolean.
    OUTPUT:
    Z           Vertices for set of supergame equilibria
    figure      Plot of conhull and vertices from earlier iterations
    The function will generate plots by default. Set Plot==False to turn off plotting.
    '''

    start_time = time.time()
    #---------------------------------------------------------------------------
    # check inputs for correctness
    #---------------------------------------------------------------------------
    p1_x, p1_y = p1.shape
    if p1_x != p1_y or p1.shape != p2.shape:
        raise Exception("payoff matrices must be square and of the same size")
    if n_grad < 2:
        raise Exception("insufficient number of search gradients")

    #---------------------------------------------------------------------------
    # parameters
    #---------------------------------------------------------------------------
    del1 = 1-delta
    x, y = p1.shape
    p1max = np.reshape(np.tile(np.max(p1, 0), (max(x, y), 1)), (x*y, 1))
    p2max = np.reshape(np.tile(np.max(p2, 1), (max(x, y), 1)).T, (1, x*y))
    stagepay = np.hstack((np.reshape(p1, (1, -1)).T, np.reshape(p2, (1, -1)).T))
    BR = np.hstack((np.atleast_2d(p1max), np.atleast_2d(p2max).T))

    #---------------------------------------------------------------------------
    # gradients and tangency points
    #---------------------------------------------------------------------------
    incr = 360.0/n_grad
    cum = 0
    H = []
    Z = []
    while cum < 360:
        x = np.cos(cum * np.pi/180)
        y = np.sin(cum * np.pi/180)
        H.append((x, y))
        Z.append((cen[0, 0]+rad*x, cen[0, 1] + rad*y))
        cum += incr

    C = np.atleast_2d(np.sum(np.array(Z)*np.array(H), axis=1))
    Z = np.array(Z)
    L = len(H)
    A = len(stagepay)
    C = C.T

    # convert needed arrays to matrix objects (for linear programming routine)
    H = cvx.matrix(np.array(H))
    G = H

    #-------------------------------------------------------------------------------
    # Use subgradient same as search directions
    #-------------------------------------------------------------------------------
    [x, y, z] = _cylinder(rad, 200)

    if plot is True:
        # plot main circle and initial search points
        plt.figure()
        plt.plot(x[0, :] + cen[0, 0], y[0, :] + cen[0, 1])
        plt.plot(Z[:, 0], Z[:, 1], 'rx')

    #---------------------------------------------------------------------------
    # Begin optimization portion of program
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # iterative parameteres
    #---------------------------------------------------------------------------
    wmin = np.ones((2, 1))*-10
    iter = 0                            # Current number of iterations
    tolZ = 1                            # Initialized error between Z and Z'
    Zold = np.zeros((len(Z), 2))        # Initialized Z array

    solvers.options['show_progress'] = False

    #---------------------------------------------------------------------------
    # Begin algorithm for outer hyperplane approxmation
    #---------------------------------------------------------------------------
    if display is True:
        print('Inner Hyperplane Approximation')

    while tolZ > tol and iter < MaxIter:

        # Construct iteration
        Cla = np.zeros((L, A))          # The set of equilibrium payoffs
        Wla = np.zeros((A, 2, L))       # The set of equilibrium arguments

        # loop through L search gradiants and A possible actions
        for l in range(L):
            for a in range(A):
                #---------------------------------------------------------------
                # Step 1.a:
                # solve Cla = max_w h_l*[del1*payoff(a) + delta * w   sub. to
                #    del1*payoff(a) + delta*w \geq del1* BR(a) + delta * wmin
                #---------------------------------------------------------------
                pay = np.atleast_2d(stagepay[a, :])

                b = cvx.matrix(np.vstack((delta*C+del1 * np.dot(G, pay.T), -del1 * np.atleast_2d(BR[a, :]).T - delta * wmin)))

                T = solvers.lp(-H[l, :].T, cvx.matrix(np.vstack((G, -np.eye(2)))), b)

                if T['status'] == 'optimal':
                    Wla[a, :, l] = np.array(T['x'])[:, 0]
                    Cla[l, a] = -np.inner(-H[l, :], T['x'].T)
                else:
                    Cla[l, a] = -np.inf

        #----------------------------------------------------------------
        # Step 1.b:
        # find best action profile 'a_*' such that
        #       a_* = argmax{Cla}                 --- element of I
        #       z = del1*payoff(a_*) + del*Wla_*  --- element of C
        #----------------------------------------------------------------
        I = np.atleast_2d(np.argmax(Cla, axis=1)).T
        C = np.atleast_2d(np.max(Cla, axis=1)).T

        #----------------------------------------------------------------
        # Step 2:
        # Collect the set of vertices Z = Wla[I]
        # Define W+ = co(Z)
        #----------------------------------------------------------------
        for l in range(L):
            Z[l, :] = Wla[I[l], :, l]

        wmin = np.atleast_2d(np.min(Z, axis=0)).T

        #----------------------------------------------------------------
        # Add points of Z to plot
        #----------------------------------------------------------------
        if plot is True:
            plt.plot(Z[:, 0], Z[:, 1], 'o')

        #----------------------------------------------------------------
        # Measure convergence
        #----------------------------------------------------------------
        if Housdorff is True:
            tolZ = hausdorffnorm(Z, Zold)
        else:
            tolZ = np.max(np.max(np.abs(Z-Zold)/(1.+abs(Zold)), axis=0))

        if iter == MaxIter:
            print('No Convergence in allowed number of iterations \n')
            break

        #----------------------------------------------------------------
        # Print results
        #----------------------------------------------------------------
        if np.mod(iter, 5) == 0 and display is True:
            print('iteration: %d \t tolerance: %f' % (iter, tolZ))

        #----------------------------------------------------------------
        # update iterative parameters
        #----------------------------------------------------------------
        Zold = Z.copy()
        iter += 1

    #-------------------------------------------------------------------------------
    # Find convex hull of most recent Z array and plot
    #-------------------------------------------------------------------------------
    if plot is True:
        Zplot = np.vstack((Z, Z[0,:]))
        plt.plot(Zplot[:, 0], Zplot[:, 1], 'r-')
        plt.xlabel('Payoff: Player 1')
        plt.ylabel('Payoff: Player 2')

    #-------------------------------------------------------------------------------
    # Plot final results and display
    #-------------------------------------------------------------------------------
    if iter < MaxIter and display is True:
        print('Convergence after %d iterations' % (iter))

        # evaluate and display elapsed time
        elapsed_time = time.time() - start_time
        print('Elapsed time is %f seconds' % (elapsed_time))

    #display plot
    if plot is True:
        plt.show()

    return Z


def outerbound(p1, p2, cen, rad, n_grad=8, delta=0.8, plot=True, tol=1e-4, MaxIter=200, display=True, Housedorff=False):
    '''
    This method computes the outerbound approximation of Judd, Yeltekin, Cronklin (2003)
    for 2 agents.
    INPUTS:
    p1:         payoff matrix for player 1. numpy array(n,n, ndim=2)
    p2:         payoff matrix for player 2. numpy array(n,n, ndim=2)
    cen:        center for initial guess, a circle. numpy.array (2,1, ndim=2)
    rad:        radius for initial guess, a circle. float.
    n_grad:     number of search gradients. int.
    delta:      discount factor. float.
    plot:       True will generate plots. boolean.
    tol:        Minimum tolerable convergence error. float.
    MaxIter:    Maximum number of iterations allowed. int.
    display:    Option to display output during iterations. boolean.
    Hausdorff:  Option to measure error using Hausdorff norm instead of standard relative error measure. boolean.
    OUTPUT:
    Z           Vertices for set of supergame equilibria. array.
    figure      Plot of conhull and vertices from earlier iterations
    The function will generate plots by default. Set Plot==False to turn off plotting.
    '''

    start_time = time.time()
    #---------------------------------------------------------------------------
    # check inputs for correctness
    #---------------------------------------------------------------------------
    p1_x, p1_y = p1.shape
    if p1_x != p1_y or p1.shape != p2.shape:
        raise Exception("payoff matrices must be square and of the same size")
    if n_grad < 2:
        raise Exception("insufficient number of search gradients")

    #---------------------------------------------------------------------------
    # parameters
    #---------------------------------------------------------------------------
    del1 = 1-delta
    x, y = p1.shape
    p1max = np.reshape(np.tile(np.max(p1, 0), (max(x, y), 1)), (x*y, 1))
    p2max = np.reshape(np.tile(np.max(p2, 1), (max(x, y), 1)).T, (1, x*y))
    stagepay = np.hstack((np.reshape(p1, (1, -1)).T, np.reshape(p2, (1, -1)).T))
    BR = np.hstack((np.atleast_2d(p1max), np.atleast_2d(p2max).T))

    #---------------------------------------------------------------------------
    # gradients and tangency points
    #---------------------------------------------------------------------------
    incr = 360.0/n_grad
    cum = 0
    H = []
    Z = []
    while cum < 360:
        x = np.cos(cum*np.pi/180)
        y = np.sin(cum*np.pi/180)
        H.append((x, y))
        Z.append((cen[0, 0]+rad*x, cen[0, 1] + rad*y))
        cum += incr

    C = np.atleast_2d(np.sum(np.array(Z)*np.array(H), axis=1))
    Z = np.array(Z)
    L = len(H)
    A = len(stagepay)
    C = C.T

    # convert needed arrays to matrix objects (for linear programming routine)
    H = cvx.matrix(np.array(H))
    G = cvx.matrix(np.vstack((H, -np.eye(2))))

    #-------------------------------------------------------------------------------
    # Use subgradient same as search directions
    #-------------------------------------------------------------------------------
    [x, y, z] = _cylinder(rad, 200)

    if plot is True:
        # plot main circle and initial search points
        plt.figure()
        plt.plot(x[0, :] + cen[0, 0], y[0, :] + cen[0, 1])
        plt.plot(Z[:, 0], Z[:, 1], 'rx')

    #---------------------------------------------------------------------------
    # Begin optimization portion of program
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # iterative parameteres
    #---------------------------------------------------------------------------
    wmin = np.ones((2, 1))*-10
    iter = 0                            # Current number of iterations
    tolZ = 1                            # Initialized error between Z and Z'
    Zold = np.zeros((len(Z), 2))        # Initialized Z array

    solvers.options['show_progress'] = False

    #---------------------------------------------------------------------------
    # Begin algorithm for outer hyperplane approxmation
    #---------------------------------------------------------------------------
    if display is True:
        print('Outer Hyperplane Approximation')

    while tolZ > tol and iter < MaxIter:

        # Construct iteration
        Cla = np.zeros((L, A))          # The set of equilibrium payoffs
        Wla = np.zeros((A, 2, L))       # The set of equilibrium arguments

        # loop through L search gradiants and A possible actions
        for l in range(L):
            for a in range(A):
                #---------------------------------------------------------------
                # Step 1.a:
                # solve Cla = max_w h_l*[del1*payoff(a) + delta * w   sub. to
                #    del1*payoff(a) + delta*w \geq del1* BR(a) + delta * wmin
                #---------------------------------------------------------------
                pay = np.atleast_2d(stagepay[a, :])
                b = cvx.matrix(np.vstack((delta*C+del1*np.dot(H, pay.T), -del1*np.atleast_2d(BR[a, :]).T-delta*wmin)))
                T = solvers.lp(-H[l, :].T, G, b)

                if T['status'] == 'optimal':
                    Wla[a, :, l] = np.array(T['x'])[:, 0]
                    Cla[l, a] = -np.inner(-H[l, :], T['x'].T)
                else:
                    Cla[l, a] = -np.inf

        #----------------------------------------------------------------
        # Step 1.b:
        # find best action profile 'a_*' such that
        #       a_* = argmax{Cla}                 --- element of I
        #       z = del1*payoff(a_*) + del*Wla_*  --- element of C
        #----------------------------------------------------------------
        I = np.atleast_2d(np.argmax(Cla, axis=1)).T
        C = np.atleast_2d(np.max(Cla, axis=1)).T

        #----------------------------------------------------------------
        # Step 2:
        # Collect the set of vertices Z = Wla[I]
        # Search gradients are fixed so this gives supporting hyperplanes
        #----------------------------------------------------------------
        for l in range(L):
            Z[l, :] = Wla[I[l], :, l]

        wmin = np.atleast_2d(np.min(Z, axis=0)).T

        #----------------------------------------------------------------
        # Add points of Z to plot
        #----------------------------------------------------------------
        if plot is True:
            plt.plot(Z[:, 0], Z[:, 1], 'o')

        #----------------------------------------------------------------
        # Measure convergence
        #----------------------------------------------------------------
        if Housedorff is True:
            tolZ = hausdorffnorm(Z, Zold)
        else:
            tolZ = np.max(np.max(np.abs(Z-Zold)/(1.+abs(Zold)), axis=0))


        if iter == MaxIter:
            print('No Convergence in allowed number of iterations \n')
            break

        #----------------------------------------------------------------
        # Print results
        #----------------------------------------------------------------
        if np.mod(iter, 5) == 0 and display is True:
            print('iteration: %d \t tolerance: %f' % (iter, tolZ))

        #----------------------------------------------------------------
        # update iterative parameters
        #----------------------------------------------------------------
        Zold = Z.copy()
        iter += 1

    #-------------------------------------------------------------------------------
    # Find shape defined by supporting hyperplanes
    #-------------------------------------------------------------------------------
    H_perp = np.hstack((-H[:, 1], H[:, 0]))*2

    lines = []
    for l in range(L):
        lines.append((Z[l, :] + H_perp[l, :], Z[l, :] - H_perp[l, :]))

    lines.append(lines[0])
    vertices = []
    for l in range(1, L+1):
        a1, a2 = lines[l-1]
        b1, b2 = lines[l]
        vertices.append(_seg_intersect(a1, a2, b1, b2))

    vertices = np.array(vertices)

    if plot is True:
        Vplot = np.vstack((vertices, vertices[0, :]))
        plt.plot(Vplot[:, 0], Vplot[:, 1], 'r-')
        plt.xlabel('Payoff: Player 1')
        plt.ylabel('Payoff: Player 2')


    #-------------------------------------------------------------------------------
    # Plot final results and display
    #-------------------------------------------------------------------------------
    if iter < MaxIter and display is True:
        print('Convergence after %d iterations' % (iter))

        # evaluate and display elapsed time
        elapsed_time = time.time() - start_time
        print('Elapsed time is %f seconds' % (elapsed_time))

    #display plot
    if plot is True:
        plt.show()

    return vertices


def innerbound_par(p1, p2, cen, rad, n_grad=8, delta=0.8, tol=1e-4, MaxIter=200, plot=True, display=True, Housedorff=False):
    '''
    This method computes the innerbound approximation of Judd, Yeltekin, Cronklin (2003)
    for 2 agents.
    INPUTS:
    p1:         payoff matrix for player 1. numpy array(n,n, ndim=2)
    p2:         payoff matrix for player 2. numpy array(n,n, ndim=2)
    cen:        center for initial guess, a circle. numpy.array (2,1, ndim=2)
    rad:        radius for initial guess, a circle. float.
    n_grad:     number of search gradients. int.
    delta:      discount factor. float.
    plot:       True will generate plots. boolean.
    tol:        Minimum tolerable convergence error. float.
    MaxIter:    Maximum number of iterations allowed. int.
    display:    Option to display output during iterations. boolean.
    Hausdorff:  Option to measure error using Hausdorff norm instead of standard relative error measure. boolean.
    OUTPUT:
    Z           Vertices for set of supergame equilibria. array.
    figure      Plot of conhull and vertices from earlier iterations
                (True by default. Set Plot==False to turn off)
    '''
    # MPI preliminaries
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #-------------------------------------------------------------------------------
    # start timer
    #-------------------------------------------------------------------------------
    if rank == 0:
        start_time = time.time()

    #-------------------------------------------------------------------------------
    # check inputs for correctness
    #-------------------------------------------------------------------------------
    p1_x, p1_y = p1.shape
    if p1_x != p1_y or p1.shape != p2.shape:
        raise Exception("payoff matrices must be square and of the same size")

    if n_grad < 2:
        raise Exception("insufficient number of search gradients")

    #-------------------------------------------------------------------------------
    # parameters
    #-------------------------------------------------------------------------------
    del1 = 1 - delta
    x, y = p1.shape
    p1max = np.reshape(np.tile(np.max(p1, 0), (max(x, y), 1)), (x*y, 1))
    p2max = np.reshape(np.tile(np.max(p2, 1), (max(x, y), 1)).T, (1, x*y))
    stagepay = np.hstack((np.reshape(p1, (1, -1)).T, np.reshape(p2, (1, -1)).T))
    BR = np.hstack((np.atleast_2d(p1max), np.atleast_2d(p2max).T))

    #-------------------------------------------------------------------------------
    # gradients and tangency points
    #-------------------------------------------------------------------------------
    incr = 360.0/n_grad
    cum = 0
    H = []
    Z = []
    while cum < 360:
        x = np.cos(cum * np.pi/180)
        y = np.sin(cum * np.pi/180)
        H.append((x, y))
        Z.append((cen[0, 0]+rad*x, cen[0, 1]+rad*y))
        cum = cum + incr

    C = np.atleast_2d(np.sum(np.array(Z)*np.array(H), axis=1))
    Z = np.array(Z)
    L = len(H)
    A = len(stagepay)
    C = C.T

    #-------------------------------------------------------------------------------
    # convert necessary arrays to matrix objects (for CVXOPT lp routine)
    #-------------------------------------------------------------------------------
    H = cvx.matrix(np.array(H))
    G = H

    #-------------------------------------------------------------------------------
    # Use subgradient same as search directions
    #-------------------------------------------------------------------------------
    [x, y, z] = _cylinder(rad, 200)

    # plot main circle and initial search points
    if rank == 0 and plot is True:
        plt.figure()
        plt.plot(x[0, :] + cen[0, 0], y[0, :] + cen[0, 1])
        plt.plot(Z[:, 0], Z[:, 1], 'rx')

    #-------------------------------------------------------------------------------
    # Begin optimization portion of program
    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------
    # iterative parameteres
    #-------------------------------------------------------------------------------
    wmin = np.ones((2, 1))*-10
    iter = 0
    tolZ = 1
    Zold = np.zeros((len(Z), 2))
    slices = _loadbalance(n_grad, size)
    cumslices = np.cumsum(np.array(slices))
    slice = slices[rank+1]

    solvers.options['show_progress'] = False

    if rank == 0 and display is True:
        print('Inner Hyperplane Approximation')

    #-------------------------------------------------------------------------------
    # Begin algorithm for inner hyperplane approximation
    #-------------------------------------------------------------------------------
    # if rank <= n:
    while tolZ > tol and iter < MaxIter:

        # Construct iteration
        WlaCla_Buffer = np.zeros((A, 3, L))
        WlaCla_entry = np.zeros((A, 3, L))

        # loop through L search gradients
        for k in range(slice):
            l = int(cumslices[rank]) + k

            # loop A possible actions
            for a in range(A):
                #----------------------------------------------------------------
                # Step 1.a:
                # solve Cla = max_w h_l*[del1*payoff(a) + delta * w   sub. to
                #       del1*payoff(a) + delta*w \geq del1* BR(a) + delta * wmin
                #----------------------------------------------------------------
                pay = np.atleast_2d(stagepay[a, :])
                b = cvx.matrix(np.vstack((delta*C+del1 * np.dot(G, pay.T), -del1 * np.atleast_2d(BR[a, :]).T - delta * wmin)))
                T = solvers.lp(-H[l, :].T, cvx.matrix(np.vstack((G, -np.eye(2)))), b)

                if T['status'] == 'optimal':
                    WlaCla_entry[a, 0:2, l] = np.array(T['x'])[:, 0]            # Wla
                    WlaCla_entry[a, 2, l] = -np.inner(-H[l, :], T['x'].T)       # Cla
                else:
                    WlaCla_entry[a, 2, l] = -np.inf

        #----------------------------------------------------------------
        # gather all the pieces
        #----------------------------------------------------------------
        comm.Allreduce(WlaCla_entry, WlaCla_Buffer, op=MPI.SUM)
        Wla = WlaCla_Buffer[:, 0:2, :]
        Cla = WlaCla_Buffer[:, 2, :].T

        #----------------------------------------------------------------
        # Step 1.b:
        # find best action profile 'a_*' such that
        #       a_* = argmax{Cla}                 --- element of I
        #       z = del1*payoff(a_*) + del*Wla_*  --- element of C
        #----------------------------------------------------------------
        I = np.atleast_2d(np.argmax(Cla, axis=1)).T
        C = np.atleast_2d(np.max(Cla, axis=1)).T

        #----------------------------------------------------------------
        # Set 2:
        # Collect the set of vertices Z = Wla[I]
        # Define W+ = co(Z)
        #----------------------------------------------------------------
        for l in range(L):
            Z[l, :] = Wla[I[l], :, l]

        wmin = np.atleast_2d(np.min(Z, axis=0)).T

        #----------------------------------------------------------------
        # Measure convergence
        #----------------------------------------------------------------
        if Housedorff is True:
            tolZ = hausdorffnorm(Z, Zold)
        else:
            tolZ = np.max(np.max(np.abs(Z-Zold)/(1.+abs(Zold)), axis=0))

        #----------------------------------------------------------------
        # Add points of Z to plot
        #----------------------------------------------------------------
        if rank == 0:
            if iter == MaxIter:
                print('No Convergence in allowed number of iterations \n')
                break

            #----------------------------------------------------------------
            # Print results
            #----------------------------------------------------------------
            if np.mod(iter, 5) == 0 and display is True:
                print('iteration: %d \t tolerance: %f' % (iter, tolZ))

            if plot is True:
                plt.plot(Z[:, 0], Z[:, 1], 'o')

        #----------------------------------------------------------------
        # update iterative parameters
        #----------------------------------------------------------------
        Zold = Z.copy()
        iter += 1

    if rank == 0:
        #-------------------------------------------------------------------------------
        # Find convex hull of most recent Z array and plot
        #-------------------------------------------------------------------------------
        if plot is True:
            Zplot = np.vstack((Z, Z[0,:]))
            plt.plot(Zplot[:, 0], Zplot[:, 1], 'r-')
            plt.xlabel('Payoff: Player 1')
            plt.ylabel('Payoff: Player 2')

        #-------------------------------------------------------------------------------
        # Plot final results and display
        #-------------------------------------------------------------------------------
        if iter < MaxIter and display is True:
            print('Convergence after %d iterations' % (iter))

            # evaluate and display elapsed time
            elapsed_time = time.time() - start_time
            print('Elapsed time is %f seconds' % (elapsed_time))

        #display plot
        if plot is True:
            plt.show()

        return Z


def outerbound_par(p1, p2, cen, rad, n_grad=8, delta=0.8, tol=1e-4, MaxIter=200, plot=True, display=True, Housedorff=False):
    '''
    This method computes the outerbound approximation of Judd, Yeltekin, Cronklin (2003)
    for 2 agents.
    INPUTS:
    p1:         payoff matrix for player 1. numpy array(n,n, ndim=2)
    p2:         payoff matrix for player 2. numpy array(n,n, ndim=2)
    cen:        center for initial guess, a circle. numpy.array (2,1, ndim=2)
    rad:        radius for initial guess, a circle. float.
    n_grad:     number of search gradients. int.
    delta:      discount factor. float.
    plot:       True will generate plots. boolean.
    tol:        Minimum tolerable convergence error. float.
    MaxIter:    Maximum number of iterations allowed. int.
    display:    Option to display output during iterations. boolean.
    Hausdorff:  Option to measure error using Hausdorff norm instead of standard relative error measure. boolean.
    OUTPUT:
    Z           Vertices for set of supergame equilibria. array.
    figure      Plot of conhull and vertices from earlier iterations
                (True by default. Set Plot==False to turn off)
    '''
    # MPI preliminaries
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #-------------------------------------------------------------------------------
    # start timer
    #-------------------------------------------------------------------------------
    if rank == 0:
        start_time = time.time()

    #-------------------------------------------------------------------------------
    # check inputs for correctness
    #-------------------------------------------------------------------------------
    p1_x, p1_y = p1.shape
    if p1_x != p1_y or p1.shape != p2.shape:
        raise Exception("payoff matrices must be square and of the same size")

    if n_grad < 2:
        raise Exception("insufficient number of search gradients")

    #-------------------------------------------------------------------------------
    # parameters
    #-------------------------------------------------------------------------------
    del1 = 1-delta
    x, y = p1.shape
    p1max = np.reshape(np.tile(np.max(p1, 0), (max(x, y), 1)), (x*y, 1))
    p2max = np.reshape(np.tile(np.max(p2, 1), (max(x, y), 1)).T, (1, x*y))
    stagepay = np.hstack((np.reshape(p1, (1, -1)).T, np.reshape(p2, (1, -1)).T))
    BR = np.hstack((np.atleast_2d(p1max), np.atleast_2d(p2max).T))

    #-------------------------------------------------------------------------------
    # gradients and tangency points
    #-------------------------------------------------------------------------------
    incr = 360.0/n_grad
    cum = 0
    H = []
    Z = []
    while cum < 360:
        x = np.cos(cum * np.pi/180)
        y = np.sin(cum * np.pi/180)
        H.append((x, y))
        Z.append((cen[0, 0]+rad*x, cen[0, 1]+rad*y))
        cum = cum + incr

    C = np.atleast_2d(np.sum(np.array(Z)*np.array(H), axis=1))
    Z = np.array(Z)
    L = len(H)
    A = len(stagepay)
    C = C.T

    #-------------------------------------------------------------------------------
    # convert necessary arrays to matrix objects (for CVXOPT lp routine)
    #-------------------------------------------------------------------------------
    H = cvx.matrix(np.array(H))
    G = cvx.matrix(np.vstack((H, -np.eye(2))))

    #-------------------------------------------------------------------------------
    # Use subgradient same as search directions
    #-------------------------------------------------------------------------------
    [x, y, z] = _cylinder(rad, 200)

    # plot main circle and initial search points
    if rank == 0 and plot is True:
        plt.figure()
        plt.plot(x[0, :] + cen[0, 0], y[0, :] + cen[0, 1])
        plt.plot(Z[:, 0], Z[:, 1], 'rx')

    #-------------------------------------------------------------------------------
    # Begin optimization portion of program
    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------
    # iterative parameteres
    #-------------------------------------------------------------------------------
    wmin = np.ones((2, 1))*-10
    iter = 0
    tolZ = 1
    Zold = np.zeros((len(Z), 2))
    slices = _loadbalance(n_grad, size)
    cumslices = np.cumsum(np.array(slices))
    slice = slices[rank+1]

    solvers.options['show_progress'] = False

    if rank == 0 and display is True:
        print('Outer Hyperplane Approximation')

    #-------------------------------------------------------------------------------
    # Begin algorithm for outer hyperplane approximation
    #-------------------------------------------------------------------------------
    # if rank <= n:
    while tolZ > tol and iter < MaxIter:

        # Construct iteration
        WlaCla_Buffer = np.zeros((A, 3, L))
        WlaCla_entry = np.zeros((A, 3, L))

        # loop through L search gradients
        for k in range(slice):
            l = int(cumslices[rank]) + k

            # loop A possible actions
            for a in range(A):
                #----------------------------------------------------------------
                # Step 1.a:
                # solve Cla = max_w h_l*[del1*payoff(a) + delta * w   sub. to
                #       del1*payoff(a) + delta*w \geq del1* BR(a) + delta * wmin
                #----------------------------------------------------------------
                pay = np.atleast_2d(stagepay[a, :])
                b = cvx.matrix(np.vstack((delta*C+del1*np.dot(H, pay.T), -del1*np.atleast_2d(BR[a, :]).T-delta*wmin)))
                T = solvers.lp(-H[l, :].T, G, b)

                if T['status'] == 'optimal':
                    WlaCla_entry[a, 0:2, l] = np.array(T['x'])[:, 0]            # Wla
                    WlaCla_entry[a, 2, l] = -np.inner(-H[l, :], T['x'].T)       # Cla
                else:
                    WlaCla_entry[a, 2, l] = -np.inf

        #----------------------------------------------------------------
        # gather all the pieces
        #----------------------------------------------------------------
        comm.Allreduce(WlaCla_entry, WlaCla_Buffer, op=MPI.SUM)
        Wla = WlaCla_Buffer[:, 0:2, :]
        Cla = WlaCla_Buffer[:, 2, :].T

        #----------------------------------------------------------------
        # Step 1.b:
        # find best action profile 'a_*' such that
        #       a_* = argmax{Cla}                 --- element of I
        #       z = del1*payoff(a_*) + del*Wla_*  --- element of C
        #----------------------------------------------------------------
        I = np.atleast_2d(np.argmax(Cla, axis=1)).T
        C = np.atleast_2d(np.max(Cla, axis=1)).T

        #----------------------------------------------------------------
        # Set 2:
        # Collect the set of vertices Z = Wla[I]
        # Define W+ = co(Z)
        #----------------------------------------------------------------
        for l in range(L):
            Z[l, :] = Wla[I[l], :, l]

        wmin = np.atleast_2d(np.min(Z, axis=0)).T

        #----------------------------------------------------------------
        # Measure convergence
        #----------------------------------------------------------------
        if Housedorff is True:
            tolZ = hausdorffnorm(Z, Zold)
        else:
            tolZ = np.max(np.max(np.abs(Z-Zold)/(1.+abs(Zold)), axis=0))

        #----------------------------------------------------------------
        # Add points of Z to plot
        #----------------------------------------------------------------
        if rank == 0:
            if iter == MaxIter:
                print('No Convergence in allowed number of iterations \n')
                break

            #----------------------------------------------------------------
            # Print results
            #----------------------------------------------------------------
            if np.mod(iter, 5) == 0 and display is True:
                print('iteration: %d \t tolerance: %f' % (iter, tolZ))

            if plot is True:
                plt.plot(Z[:, 0], Z[:, 1], 'o')

        #----------------------------------------------------------------
        # update iterative parameters
        #----------------------------------------------------------------
        Zold = Z.copy()
        iter += 1

    if rank == 0:
        #-------------------------------------------------------------------------------
        # Find covex hull of most recent Z array and plot
        #-------------------------------------------------------------------------------
        H_perp = np.hstack((-H[:, 1], H[:, 0]))*2

        lines = []
        for l in range(L):
            lines.append((Z[l, :] + H_perp[l, :], Z[l, :] - H_perp[l, :]))

        lines.append(lines[0])
        vertices = []
        for l in range(1, L+1):
            a1, a2 = lines[l-1]
            b1, b2 = lines[l]
            vertices.append(_seg_intersect(a1, a2, b1, b2))

        vertices = np.array(vertices)

        if plot is True:
            Vplot = np.vstack((vertices, vertices[0, :]))
            plt.plot(Vplot[:, 0], Vplot[:, 1], 'r-')
            plt.xlabel('Payoff: Player 1')
            plt.ylabel('Payoff: Player 2')


        #-------------------------------------------------------------------------------
        # Plot final results and display
        #-------------------------------------------------------------------------------
        if iter < MaxIter and display is True:
            print('Convergence after %d iterations' % (iter))

            # evaluate and display elapsed time
            elapsed_time = time.time() - start_time
            print('Elapsed time is %f seconds' % (elapsed_time))

        #display plot
        if plot is True:
            plt.show()

        return vertices


if __name__ == '__main__':
    p1 = np.array([[4, 0], [6, 2]])
    p2 = p1.T
    cen = np.array([3, 3], ndmin=2)
    rad = .5
    Z_inner = innerbound(p1, p2, cen, rad)
    Z_outer = outerbound(p1, p2, cen, rad*10)