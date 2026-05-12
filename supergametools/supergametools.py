#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Author: Benjamin Tengelsen, Carnegie Mellon University, btengels@cmu.edu
# This file computes an outerbound approximation for the set of supergame
# equilibria in a 2-player infinitely repeated game.
# It is based on the algorithm given in Judd, Yeltekin, Conklin (2003)
# ------------------------------------------------------------------------------

import time
import numpy as np
import cvxopt as cvx
import matplotlib.pyplot as plt
from cvxopt import solvers


__all__ = ['outerbound', 'outerbound_par', 'innerbound', 'innerbound_par', 'hausdorffnorm']


def _cylinder(r, n):
    '''
    Returns the unit cylinder that corresponds to the curve r.
    INPUTS:
    r : a vector of radii
    n : number of coordinates to return for each element in r
    OUTPUTS:
    x, y, z: coordinates of points around cylinder
    '''
    r = np.atleast_2d(r)
    r_rows, r_cols = r.shape

    if r_cols > r_rows:
        r = r.T

    points = np.linspace(0, 2*np.pi, n+1)
    x = np.cos(points)*r
    y = np.sin(points)*r

    rpoints = np.atleast_2d(np.linspace(0, 1, len(r)))
    z = np.ones((1, n+1))*rpoints.T

    return x, y, z


def hausdorffnorm(A, B):
    '''
    Finds the Hausdorff norm between two matrices A and B.
    INPUTS:
    A : numpy array
    B : numpy array
    OUTPUTS:
    Hausdorff norm between matrices A and B
    '''
    if len(A.shape) == 1:
        A = np.atleast_2d(A)

    if len(B.shape) == 1:
        B = np.atleast_2d(B)

    A = np.atleast_3d(A)
    B = np.atleast_3d(B)

    x, y, z = B.shape
    A = np.reshape(A, (z, x, y))
    B = np.reshape(B, (z, x, y))

    z, x, y = B.shape
    temp1 = np.tile(np.reshape(B.T, (y, z, x)), (max(A.shape), 1))
    temp2 = np.tile(np.reshape(A.T, (y, x, z)), (1, max(B.shape)))
    D1 = np.min(np.sqrt(np.sum((temp1-temp2)**2, 0)), axis=0)

    temp1 = np.tile(np.reshape(A.T, (y, z, x)), (max(B.shape), 1))
    temp2 = np.tile(np.reshape(B.T, (y, x, z)), (1, max(A.shape)))
    D2 = np.min(np.sqrt(np.sum((temp1-temp2)**2, 0)), axis=0)

    return np.max([D1, D2])


def _loadbalance(n, p):
    '''
    Determines how many search gradients are allocated to each process.
    INPUTS:
    n       number of search gradients, int.
    p       number of processes, int.
    OUTPUTS:
    load    list with the number of gradients assigned to each process, list.
    '''
    inc = n // p
    R = n - inc * p

    load = [inc for _ in range(p)]

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
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = _perp(da)
    denom = np.dot(dap, db)
    num = float(np.dot(dap, dp))
    return (num / denom)*db + b1


def innerbound(p1, p2, cen, rad, n_grad=8, delta=0.8, plot=True, tol=1e-4, max_iter=200, display=True, hausdorff=False):
    '''
    Computes the innerbound approximation of Judd, Yeltekin, Conklin (2003)
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
    max_iter:   Maximum number of iterations allowed. int.
    display:    Option to display output during iterations. boolean.
    hausdorff:  Option to measure error using Hausdorff norm instead of standard relative error measure. boolean.
    OUTPUT:
    Z           Vertices for set of supergame equilibria
    figure      Plot of convex hull and vertices from earlier iterations
    The function will generate plots by default. Set plot=False to turn off plotting.
    '''
    start_time = time.time()

    # check inputs
    p1_x, p1_y = p1.shape
    if p1_x != p1_y or p1.shape != p2.shape:
        raise Exception("payoff matrices must be square and of the same size")
    if n_grad < 2:
        raise Exception("insufficient number of search gradients")

    # parameters
    del1 = 1 - delta
    n = p1.shape[0]
    p1max = np.reshape(np.tile(np.max(p1, 0), (n, 1)), (n*n, 1))
    p2max = np.reshape(np.tile(np.max(p2, 1), (n, 1)).T, (1, n*n))
    stagepay = np.hstack((np.reshape(p1, (1, -1)).T, np.reshape(p2, (1, -1)).T))
    BR = np.hstack((np.atleast_2d(p1max), np.atleast_2d(p2max).T))

    # gradients and tangency points
    H = []
    Z = []
    for i in range(n_grad):
        angle = i * 360.0 / n_grad
        hx = np.cos(angle * np.pi / 180)
        hy = np.sin(angle * np.pi / 180)
        H.append((hx, hy))
        Z.append((cen[0, 0] + rad*hx, cen[0, 1] + rad*hy))

    C = np.atleast_2d(np.sum(np.array(Z)*np.array(H), axis=1)).T
    Z = np.array(Z)
    L = len(H)
    A = len(stagepay)

    H = cvx.matrix(np.array(H))
    G = H

    # pre-compute per-action arrays used inside the iteration loop
    pay_list = [np.atleast_2d(stagepay[a, :]) for a in range(A)]
    BR_list = [np.atleast_2d(BR[a, :]).T for a in range(A)]

    cx, cy, _ = _cylinder(rad, 200)

    if plot:
        plt.figure()
        plt.plot(cx[0, :] + cen[0, 0], cy[0, :] + cen[0, 1])
        plt.plot(Z[:, 0], Z[:, 1], 'rx')

    # iterative parameters
    wmin = np.ones((2, 1)) * -10
    n_iter = 0
    tolZ = 1
    Zold = np.zeros((L, 2))
    Cla = np.zeros((L, A))
    Wla = np.zeros((A, 2, L))

    solvers.options['show_progress'] = False

    if display:
        print('Inner Hyperplane Approximation')

    while tolZ > tol and n_iter < max_iter:

        Cla.fill(0)
        Wla.fill(0)

        for l in range(L):
            for a in range(A):
                b = cvx.matrix(np.vstack((delta*C + del1*np.dot(G, pay_list[a].T), -del1*BR_list[a] - delta*wmin)))
                T = solvers.lp(-H[l, :].T, cvx.matrix(np.vstack((G, -np.eye(2)))), b)

                if T['status'] == 'optimal':
                    Wla[a, :, l] = np.array(T['x'])[:, 0]
                    Cla[l, a] = -np.inner(-H[l, :], T['x'].T)
                else:
                    Cla[l, a] = -np.inf

        I = np.atleast_2d(np.argmax(Cla, axis=1)).T
        C = np.atleast_2d(np.max(Cla, axis=1)).T

        Z = Wla[I.ravel(), :, np.arange(L)]
        wmin = np.atleast_2d(np.min(Z, axis=0)).T

        if plot:
            plt.plot(Z[:, 0], Z[:, 1], 'o')

        if hausdorff:
            tolZ = hausdorffnorm(Z, Zold)
        else:
            tolZ = np.max(np.abs(Z - Zold) / (1. + np.abs(Zold)))

        Zold = Z.copy()
        n_iter += 1

        if np.mod(n_iter, 5) == 0 and display:
            print(f'iteration: {n_iter} \t tolerance: {tolZ:.6f}')

    if n_iter >= max_iter and display:
        print('No convergence in allowed number of iterations')

    if plot:
        Zplot = np.vstack((Z, Z[0, :]))
        plt.plot(Zplot[:, 0], Zplot[:, 1], 'r-')
        plt.xlabel('Payoff: Player 1')
        plt.ylabel('Payoff: Player 2')

    if n_iter < max_iter and display:
        print(f'Convergence after {n_iter} iterations')
        elapsed_time = time.time() - start_time
        print(f'Elapsed time is {elapsed_time:.4f} seconds')

    if plot:
        plt.show()

    return Z


def outerbound(p1, p2, cen, rad, n_grad=8, delta=0.8, plot=True, tol=1e-4, max_iter=200, display=True, hausdorff=False):
    '''
    Computes the outerbound approximation of Judd, Yeltekin, Conklin (2003)
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
    max_iter:   Maximum number of iterations allowed. int.
    display:    Option to display output during iterations. boolean.
    hausdorff:  Option to measure error using Hausdorff norm instead of standard relative error measure. boolean.
    OUTPUT:
    vertices    Vertices for set of supergame equilibria. array.
    figure      Plot of convex hull and vertices from earlier iterations
    The function will generate plots by default. Set plot=False to turn off plotting.
    '''
    start_time = time.time()

    # check inputs
    p1_x, p1_y = p1.shape
    if p1_x != p1_y or p1.shape != p2.shape:
        raise Exception("payoff matrices must be square and of the same size")
    if n_grad < 2:
        raise Exception("insufficient number of search gradients")

    # parameters
    del1 = 1 - delta
    n = p1.shape[0]
    p1max = np.reshape(np.tile(np.max(p1, 0), (n, 1)), (n*n, 1))
    p2max = np.reshape(np.tile(np.max(p2, 1), (n, 1)).T, (1, n*n))
    stagepay = np.hstack((np.reshape(p1, (1, -1)).T, np.reshape(p2, (1, -1)).T))
    BR = np.hstack((np.atleast_2d(p1max), np.atleast_2d(p2max).T))

    # gradients and tangency points
    H = []
    Z = []
    for i in range(n_grad):
        angle = i * 360.0 / n_grad
        hx = np.cos(angle * np.pi / 180)
        hy = np.sin(angle * np.pi / 180)
        H.append((hx, hy))
        Z.append((cen[0, 0] + rad*hx, cen[0, 1] + rad*hy))

    C = np.atleast_2d(np.sum(np.array(Z)*np.array(H), axis=1)).T
    Z = np.array(Z)
    L = len(H)
    A = len(stagepay)

    H = cvx.matrix(np.array(H))
    G = cvx.matrix(np.vstack((H, -np.eye(2))))

    # pre-compute per-action arrays used inside the iteration loop
    pay_list = [np.atleast_2d(stagepay[a, :]) for a in range(A)]
    BR_list = [np.atleast_2d(BR[a, :]).T for a in range(A)]

    cx, cy, _ = _cylinder(rad, 200)

    if plot:
        plt.figure()
        plt.plot(cx[0, :] + cen[0, 0], cy[0, :] + cen[0, 1])
        plt.plot(Z[:, 0], Z[:, 1], 'rx')

    # iterative parameters
    wmin = np.ones((2, 1)) * -10
    n_iter = 0
    tolZ = 1
    Zold = np.zeros((L, 2))
    Cla = np.zeros((L, A))
    Wla = np.zeros((A, 2, L))

    solvers.options['show_progress'] = False

    if display:
        print('Outer Hyperplane Approximation')

    while tolZ > tol and n_iter < max_iter:

        Cla.fill(0)
        Wla.fill(0)

        for l in range(L):
            for a in range(A):
                b = cvx.matrix(np.vstack((delta*C + del1*np.dot(H, pay_list[a].T), -del1*BR_list[a] - delta*wmin)))
                T = solvers.lp(-H[l, :].T, G, b)

                if T['status'] == 'optimal':
                    Wla[a, :, l] = np.array(T['x'])[:, 0]
                    Cla[l, a] = -np.inner(-H[l, :], T['x'].T)
                else:
                    Cla[l, a] = -np.inf

        I = np.atleast_2d(np.argmax(Cla, axis=1)).T
        C = np.atleast_2d(np.max(Cla, axis=1)).T

        Z = Wla[I.ravel(), :, np.arange(L)]
        wmin = np.atleast_2d(np.min(Z, axis=0)).T

        if plot:
            plt.plot(Z[:, 0], Z[:, 1], 'o')

        if hausdorff:
            tolZ = hausdorffnorm(Z, Zold)
        else:
            tolZ = np.max(np.abs(Z - Zold) / (1. + np.abs(Zold)))

        Zold = Z.copy()
        n_iter += 1

        if np.mod(n_iter, 5) == 0 and display:
            print(f'iteration: {n_iter} \t tolerance: {tolZ:.6f}')

    if n_iter >= max_iter and display:
        print('No convergence in allowed number of iterations')

    # find shape defined by supporting hyperplanes
    H_perp = np.hstack((-H[:, 1], H[:, 0])) * 2

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

    if plot:
        Vplot = np.vstack((vertices, vertices[0, :]))
        plt.plot(Vplot[:, 0], Vplot[:, 1], 'r-')
        plt.xlabel('Payoff: Player 1')
        plt.ylabel('Payoff: Player 2')

    if n_iter < max_iter and display:
        print(f'Convergence after {n_iter} iterations')
        elapsed_time = time.time() - start_time
        print(f'Elapsed time is {elapsed_time:.4f} seconds')

    if plot:
        plt.show()

    return vertices


def innerbound_par(p1, p2, cen, rad, n_grad=8, delta=0.8, tol=1e-4, max_iter=200, plot=True, display=True, hausdorff=False):
    '''
    Computes the innerbound approximation of Judd, Yeltekin, Conklin (2003)
    for 2 agents. Parallel version using MPI.
    INPUTS:
    p1:         payoff matrix for player 1. numpy array(n,n, ndim=2)
    p2:         payoff matrix for player 2. numpy array(n,n, ndim=2)
    cen:        center for initial guess, a circle. numpy.array (2,1, ndim=2)
    rad:        radius for initial guess, a circle. float.
    n_grad:     number of search gradients. int.
    delta:      discount factor. float.
    plot:       True will generate plots. boolean.
    tol:        Minimum tolerable convergence error. float.
    max_iter:   Maximum number of iterations allowed. int.
    display:    Option to display output during iterations. boolean.
    hausdorff:  Option to measure error using Hausdorff norm instead of standard relative error measure. boolean.
    OUTPUT:
    Z           Vertices for set of supergame equilibria. array.
    figure      Plot of convex hull and vertices from earlier iterations
                (True by default. Set plot=False to turn off)
    '''
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        start_time = time.time()

    # check inputs
    p1_x, p1_y = p1.shape
    if p1_x != p1_y or p1.shape != p2.shape:
        raise Exception("payoff matrices must be square and of the same size")
    if n_grad < 2:
        raise Exception("insufficient number of search gradients")

    # parameters
    del1 = 1 - delta
    n = p1.shape[0]
    p1max = np.reshape(np.tile(np.max(p1, 0), (n, 1)), (n*n, 1))
    p2max = np.reshape(np.tile(np.max(p2, 1), (n, 1)).T, (1, n*n))
    stagepay = np.hstack((np.reshape(p1, (1, -1)).T, np.reshape(p2, (1, -1)).T))
    BR = np.hstack((np.atleast_2d(p1max), np.atleast_2d(p2max).T))

    # gradients and tangency points
    H = []
    Z = []
    for i in range(n_grad):
        angle = i * 360.0 / n_grad
        hx = np.cos(angle * np.pi / 180)
        hy = np.sin(angle * np.pi / 180)
        H.append((hx, hy))
        Z.append((cen[0, 0] + rad*hx, cen[0, 1] + rad*hy))

    C = np.atleast_2d(np.sum(np.array(Z)*np.array(H), axis=1)).T
    Z = np.array(Z)
    L = len(H)
    A = len(stagepay)

    H = cvx.matrix(np.array(H))
    G = H

    # pre-compute per-action arrays used inside the iteration loop
    pay_list = [np.atleast_2d(stagepay[a, :]) for a in range(A)]
    BR_list = [np.atleast_2d(BR[a, :]).T for a in range(A)]

    cx, cy, _ = _cylinder(rad, 200)

    if rank == 0 and plot:
        plt.figure()
        plt.plot(cx[0, :] + cen[0, 0], cy[0, :] + cen[0, 1])
        plt.plot(Z[:, 0], Z[:, 1], 'rx')

    # iterative parameters
    wmin = np.ones((2, 1)) * -10
    n_iter = 0
    tolZ = 1
    Zold = np.zeros((L, 2))
    slices = _loadbalance(n_grad, size)
    cumslices = np.cumsum(np.array(slices))
    grad_slice = slices[rank+1]
    WlaCla_Buffer = np.zeros((A, 3, L))
    WlaCla_entry = np.zeros((A, 3, L))

    solvers.options['show_progress'] = False

    if rank == 0 and display:
        print('Inner Hyperplane Approximation')

    while tolZ > tol and n_iter < max_iter:

        WlaCla_entry.fill(0)

        for k in range(grad_slice):
            l = int(cumslices[rank]) + k

            for a in range(A):
                b = cvx.matrix(np.vstack((delta*C + del1*np.dot(G, pay_list[a].T), -del1*BR_list[a] - delta*wmin)))
                T = solvers.lp(-H[l, :].T, cvx.matrix(np.vstack((G, -np.eye(2)))), b)

                if T['status'] == 'optimal':
                    WlaCla_entry[a, 0:2, l] = np.array(T['x'])[:, 0]
                    WlaCla_entry[a, 2, l] = -np.inner(-H[l, :], T['x'].T)
                else:
                    WlaCla_entry[a, 2, l] = -np.inf

        comm.Allreduce(WlaCla_entry, WlaCla_Buffer, op=MPI.SUM)
        Wla = WlaCla_Buffer[:, 0:2, :]
        Cla = WlaCla_Buffer[:, 2, :].T

        I = np.atleast_2d(np.argmax(Cla, axis=1)).T
        C = np.atleast_2d(np.max(Cla, axis=1)).T

        Z = Wla[I.ravel(), :, np.arange(L)]
        wmin = np.atleast_2d(np.min(Z, axis=0)).T

        if hausdorff:
            tolZ = hausdorffnorm(Z, Zold)
        else:
            tolZ = np.max(np.abs(Z - Zold) / (1. + np.abs(Zold)))

        if rank == 0:
            if np.mod(n_iter, 5) == 0 and display:
                print(f'iteration: {n_iter} \t tolerance: {tolZ:.6f}')

            if plot:
                plt.plot(Z[:, 0], Z[:, 1], 'o')

        Zold = Z.copy()
        n_iter += 1

    if rank == 0:
        if n_iter >= max_iter and display:
            print('No convergence in allowed number of iterations')

        if plot:
            Zplot = np.vstack((Z, Z[0, :]))
            plt.plot(Zplot[:, 0], Zplot[:, 1], 'r-')
            plt.xlabel('Payoff: Player 1')
            plt.ylabel('Payoff: Player 2')

        if n_iter < max_iter and display:
            print(f'Convergence after {n_iter} iterations')
            elapsed_time = time.time() - start_time
            print(f'Elapsed time is {elapsed_time:.4f} seconds')

        if plot:
            plt.show()

        return Z


def outerbound_par(p1, p2, cen, rad, n_grad=8, delta=0.8, tol=1e-4, max_iter=200, plot=True, display=True, hausdorff=False):
    '''
    Computes the outerbound approximation of Judd, Yeltekin, Conklin (2003)
    for 2 agents. Parallel version using MPI.
    INPUTS:
    p1:         payoff matrix for player 1. numpy array(n,n, ndim=2)
    p2:         payoff matrix for player 2. numpy array(n,n, ndim=2)
    cen:        center for initial guess, a circle. numpy.array (2,1, ndim=2)
    rad:        radius for initial guess, a circle. float.
    n_grad:     number of search gradients. int.
    delta:      discount factor. float.
    plot:       True will generate plots. boolean.
    tol:        Minimum tolerable convergence error. float.
    max_iter:   Maximum number of iterations allowed. int.
    display:    Option to display output during iterations. boolean.
    hausdorff:  Option to measure error using Hausdorff norm instead of standard relative error measure. boolean.
    OUTPUT:
    vertices    Vertices for set of supergame equilibria. array.
    figure      Plot of convex hull and vertices from earlier iterations
                (True by default. Set plot=False to turn off)
    '''
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        start_time = time.time()

    # check inputs
    p1_x, p1_y = p1.shape
    if p1_x != p1_y or p1.shape != p2.shape:
        raise Exception("payoff matrices must be square and of the same size")
    if n_grad < 2:
        raise Exception("insufficient number of search gradients")

    # parameters
    del1 = 1 - delta
    n = p1.shape[0]
    p1max = np.reshape(np.tile(np.max(p1, 0), (n, 1)), (n*n, 1))
    p2max = np.reshape(np.tile(np.max(p2, 1), (n, 1)).T, (1, n*n))
    stagepay = np.hstack((np.reshape(p1, (1, -1)).T, np.reshape(p2, (1, -1)).T))
    BR = np.hstack((np.atleast_2d(p1max), np.atleast_2d(p2max).T))

    # gradients and tangency points
    H = []
    Z = []
    for i in range(n_grad):
        angle = i * 360.0 / n_grad
        hx = np.cos(angle * np.pi / 180)
        hy = np.sin(angle * np.pi / 180)
        H.append((hx, hy))
        Z.append((cen[0, 0] + rad*hx, cen[0, 1] + rad*hy))

    C = np.atleast_2d(np.sum(np.array(Z)*np.array(H), axis=1)).T
    Z = np.array(Z)
    L = len(H)
    A = len(stagepay)

    H = cvx.matrix(np.array(H))
    G = cvx.matrix(np.vstack((H, -np.eye(2))))

    # pre-compute per-action arrays used inside the iteration loop
    pay_list = [np.atleast_2d(stagepay[a, :]) for a in range(A)]
    BR_list = [np.atleast_2d(BR[a, :]).T for a in range(A)]

    cx, cy, _ = _cylinder(rad, 200)

    if rank == 0 and plot:
        plt.figure()
        plt.plot(cx[0, :] + cen[0, 0], cy[0, :] + cen[0, 1])
        plt.plot(Z[:, 0], Z[:, 1], 'rx')

    # iterative parameters
    wmin = np.ones((2, 1)) * -10
    n_iter = 0
    tolZ = 1
    Zold = np.zeros((L, 2))
    slices = _loadbalance(n_grad, size)
    cumslices = np.cumsum(np.array(slices))
    grad_slice = slices[rank+1]
    WlaCla_Buffer = np.zeros((A, 3, L))
    WlaCla_entry = np.zeros((A, 3, L))

    solvers.options['show_progress'] = False

    if rank == 0 and display:
        print('Outer Hyperplane Approximation')

    while tolZ > tol and n_iter < max_iter:

        WlaCla_entry.fill(0)

        for k in range(grad_slice):
            l = int(cumslices[rank]) + k

            for a in range(A):
                b = cvx.matrix(np.vstack((delta*C + del1*np.dot(H, pay_list[a].T), -del1*BR_list[a] - delta*wmin)))
                T = solvers.lp(-H[l, :].T, G, b)

                if T['status'] == 'optimal':
                    WlaCla_entry[a, 0:2, l] = np.array(T['x'])[:, 0]
                    WlaCla_entry[a, 2, l] = -np.inner(-H[l, :], T['x'].T)
                else:
                    WlaCla_entry[a, 2, l] = -np.inf

        comm.Allreduce(WlaCla_entry, WlaCla_Buffer, op=MPI.SUM)
        Wla = WlaCla_Buffer[:, 0:2, :]
        Cla = WlaCla_Buffer[:, 2, :].T

        I = np.atleast_2d(np.argmax(Cla, axis=1)).T
        C = np.atleast_2d(np.max(Cla, axis=1)).T

        Z = Wla[I.ravel(), :, np.arange(L)]
        wmin = np.atleast_2d(np.min(Z, axis=0)).T

        if hausdorff:
            tolZ = hausdorffnorm(Z, Zold)
        else:
            tolZ = np.max(np.abs(Z - Zold) / (1. + np.abs(Zold)))

        if rank == 0:
            if np.mod(n_iter, 5) == 0 and display:
                print(f'iteration: {n_iter} \t tolerance: {tolZ:.6f}')

            if plot:
                plt.plot(Z[:, 0], Z[:, 1], 'o')

        Zold = Z.copy()
        n_iter += 1

    if rank == 0:
        if n_iter >= max_iter and display:
            print('No convergence in allowed number of iterations')

        H_perp = np.hstack((-H[:, 1], H[:, 0])) * 2

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

        if plot:
            Vplot = np.vstack((vertices, vertices[0, :]))
            plt.plot(Vplot[:, 0], Vplot[:, 1], 'r-')
            plt.xlabel('Payoff: Player 1')
            plt.ylabel('Payoff: Player 2')

        if n_iter < max_iter and display:
            print(f'Convergence after {n_iter} iterations')
            elapsed_time = time.time() - start_time
            print(f'Elapsed time is {elapsed_time:.4f} seconds')

        if plot:
            plt.show()

        return vertices


if __name__ == '__main__':
    p1 = np.array([[4, 0], [6, 2]])
    p2 = p1.T
    cen = np.array([3, 3], ndmin=2)
    rad = .5
    Z_inner = innerbound(p1, p2, cen, rad)
    Z_outer = outerbound(p1, p2, cen, rad*10)
