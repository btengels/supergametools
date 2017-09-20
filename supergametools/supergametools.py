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
from itertools import product
from numpy.linalg import norm


import mpl_toolkits.mplot3d.axes3d as plot3
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d



__all__ = ['outerbound', 'outerbound_par', 'innerbound', 'innerbound_par', 'hausdorffnorm', 'make_2d_plots', 'make_3d_plots']     # this list gives what is imported with "from supergametools import *"


def make_2d_plots(points_list, line_type, filename=None, save=True):
    '''
    '''
    fig = plt.figure()
    for i, point_array in enumerate(points_list):
        line = line_type[i]
        vertex_plot = np.vstack((point_array, point_array[0, :]))

        plt.plot(vertex_plot[:, 0], vertex_plot[:, 1], line)
        plt.xlabel('Payoff: Player 1')
        plt.ylabel('Payoff: Player 2')

    if save==True:
        plt.savefig(filename, pad_inches = .5, bbox_inches = 'tight')
    else:
        plt.show()


def make_3d_plots( points_array , filename=None, save=True):
    '''
    '''
    from scipy import spatial
    fig = plt.figure()
    plt.margins(1,1)

    ax = plot3.Axes3D(fig)
    # ax = fig.add_subplot(len(states_observed), 1, state_observed_index, projection='3d')
    # ax.set_title(str(states[state_index]))
    ax.set_xlabel(r'$V_1$', family='Times New Roman', style='normal', fontsize=16)
    ax.set_ylabel(r'$V_2$', family='Times New Roman', style='normal', fontsize=16)
    ax.set_zlabel(r'$V_3$', family='Times New Roman', style='normal', fontsize=16)

    # set background color
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # change "camera angle" for 3d plot
    ax.view_init(50, -50)

    # approximate hull with tight points_array of points
    hull = spatial.Delaunay(points_array)

    xi = np.linspace(min(points_array[:,0]), max(points_array[:,0]), num=25)
    yi = np.linspace(min(points_array[:,1]), max(points_array[:,1]), num=25)
    zi = np.linspace(min(points_array[:,2]), max(points_array[:,2]), num=25)

    grid_points = np.array([(x, y, z) for x in xi for y in yi for z in zi])
    grid_inhull = grid_points[in_hull(grid_points, hull)]


    # slice grid and plot contour hulls of each slice
    for x_fixed in xi:
        tempindex = grid_inhull[:,0] == x_fixed
        temppoints = grid_inhull[tempindex]


        if len(temppoints)>0:
            try:
                hull = spatial.ConvexHull(temppoints[:,1:3])
                for simplex in hull.simplices:
                    x = x_fixed * np.ones(hull.points[simplex, 0].shape)
                    z_color = hull.points[simplex, 1][1]/max(zi)
                    ax.plot(x, hull.points[simplex, 0], hull.points[simplex, 1], color = (0.0, z_color, z_color, z_color))

            except:
                ax.plot(temppoints[:,0],temppoints[:,1],temppoints[:,2], 'b.')


    for y_fixed in yi:
        tempindex = grid_inhull[:,1] == y_fixed
        temppoints = grid_inhull[tempindex]

        if len(temppoints)>0:
            try:
                hull = spatial.ConvexHull(temppoints[:,(0, 2)])
                for simplex in hull.simplices:
                    y = y_fixed * np.ones(hull.points[simplex, 0].shape)
                    z_color = hull.points[simplex, 1][1]/max(zi)
                    ax.plot(hull.points[simplex, 0], y,  hull.points[simplex, 1], color = (0.0, z_color, z_color, z_color))

            except:
                ax.plot(temppoints[:,0],temppoints[:,1],temppoints[:,2], 'b.')

    if save==True:
        plt.savefig(filename, pad_inches = .5, bbox_inches = 'tight')
    else:
        plt.show()




def _make_circle(n_grad, cen, rad):
    '''
    '''
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

    return np.array(H), np.array(Z)


def _make_sphere(cen, rad, n_points, n_dim ):
    '''
    This function coveres a sphere of dimension n_dim
    with a given number of points.
    INPUT:  n_points - number of covering points
            n_dim - dimension of sphere
            rad - radius of sphere
    OUTPUT: x - covering points
    '''

    # randomly sample points and normalize
    H = np.random.randn(int(n_points/2), n_dim)
    H /=  np.tile(norm(H, axis=1), (n_dim,1)).T

    # put all points on half of sphere
    H[:,0]  = np.abs(H[:,0])

    # copy points to cover whole sphere and scale by given radius
    H = _spread_points(H, n_dim)
    H = np.vstack((H,-H))
    H = _spread_points(H, n_dim)

    Z = H*rad + cen

    return H, Z


def _spread_points(x, n_dim):
    '''
    This point takes a set of covering points for a sphere
    and determines which points are too close. If two points
    are too close, one of them is moved at random.
    INPUT:  x - covering points
            n_dim - dimension of sphere

    OUTPUT: x - covering points
    '''
    for i in range( len(x) ):
        for j in range( i+1,len(x) ):

            # if points are "too close", resample
            if n_dim ==2:
                dist = .75*np.pi/len(x)
            else:
                dist = 2.0*np.pi/len(x) # TODO: fix this

            if norm( x[i,:] - x[j,:] ) <  dist: #todo
                x_temp = np.random.randn(1, n_dim)
                x_temp[0,0] = abs(x_temp[0,0])
                x[j,:] = x_temp/norm(x_temp)

                _spread_points(x, n_dim)

    return x


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


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimension
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimension for which a Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0


def innerbound(P, cen, rad, n_grad=24, delta=0.8, tol=1e-4, MaxIter=200, display=True, Housdorff=False):
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
    Payoff_shape = P[0].shape
    if sum([Payoff_shape==p.shape for p in P])!=len(P):
        raise Exception("payoff matrices must all be of the same size")

    elif sum(Payoff_shape[0] == p for p in Payoff_shape) !=len(Payoff_shape):
        raise Exception("payoff matrices must all be square")

    if n_grad < 2:
        raise Exception("insufficient number of search gradients")

    #---------------------------------------------------------------------------
    # parameters
    #---------------------------------------------------------------------------
    del1 = 1-delta
    n_dim = len(P)

    Pmax = [tuple(np.max(p,i).flatten()) for i, p in enumerate(P)]
    stagepay = np.vstack( (p.flatten() for p in P ) ).T
    BR = np.vstack( (p for p in product(*Pmax)) )[:,range(len(P)-1,-1,-1)]

    #---------------------------------------------------------------------------
    # gradients and tangency points
    #---------------------------------------------------------------------------
    if n_dim == 2:
        H, Z = _make_circle(n_grad, cen, rad)
        # H, Z = _make_sphere(cen, rad, n_grad, n_dim)
    else:
        H, Z = _make_sphere(cen, rad, n_grad, n_dim)

    C = np.atleast_2d(np.sum(Z * H , axis=1))
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


    #---------------------------------------------------------------------------
    # Begin optimization portion of program
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # iterative parameteres
    #---------------------------------------------------------------------------
    wmin = np.ones((n_dim, 1))*-10
    iter = 0                            # Current number of iterations
    tolZ = 1                            # Initialized error between Z and Z'
    Zold = np.zeros((len(Z), n_dim))        # Initialized Z array

    solvers.options['show_progress'] = False

    #---------------------------------------------------------------------------
    # Begin algorithm for outer hyperplane approxmation
    #---------------------------------------------------------------------------
    if display is True:
        print('Inner Hyperplane Approximation')

    while tolZ > tol and iter < MaxIter:

        # Construct iteration
        Cla = np.zeros((L, A))          # The set of equilibrium payoffs
        Wla = np.zeros((A, n_dim, L))       # The set of equilibrium arguments

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

                T = solvers.lp(-H[l, :].T, cvx.matrix(np.vstack((G, -np.eye(n_dim)))), b)

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
    # Plot final results and display
    #-------------------------------------------------------------------------------
    if iter < MaxIter and display is True:
        print('Convergence after %d iterations' % (iter))

        # evaluate and display elapsed time
        elapsed_time = time.time() - start_time
        print('Elapsed time is %f seconds' % (elapsed_time))

    return Zold, H


def outerbound(P, cen, rad, n_grad=24, delta=0.8, tol=1e-4, MaxIter=100, display=True, Housedorff=False):
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
    Payoff_shape = P[0].shape
    if sum([Payoff_shape==p.shape for p in P])!=len(P):
        raise Exception("payoff matrices must all be of the same size")

    elif sum(Payoff_shape[0] == p for p in Payoff_shape) !=len(Payoff_shape):
        raise Exception("payoff matrices must all be square")

    if n_grad < 2:
        raise Exception("insufficient number of search gradients")

    #---------------------------------------------------------------------------
    # parameters
    #---------------------------------------------------------------------------
    del1 = 1-delta
    n_dim = len(P)

    Pmax = [tuple(np.max(p,i).flatten()) for i, p in enumerate(P)]
    stagepay = np.vstack( (p.flatten() for p in P ) ).T
    BR = np.vstack( (p for p in product(*Pmax)) )[:,range(len(P)-1,-1,-1)]

    #---------------------------------------------------------------------------
    # gradients and tangency points
    #---------------------------------------------------------------------------
    if n_dim == 2:
        H, Z = _make_circle(n_grad, cen, rad)
        print(H)
        # H, Z = _make_sphere(cen, rad, n_grad, n_dim)
    else:
        H, Z = _make_sphere(cen, rad, n_grad, n_dim)

    C = np.atleast_2d(np.sum(Z * H , axis=1))
    Z = np.array(Z)
    L = len(H)
    A = len(stagepay)
    C = C.T

    # convert needed arrays to matrix objects (for linear programming routine)
    H = cvx.matrix(np.array(H))
    G = cvx.matrix(np.vstack((H, -np.eye(n_dim))))


    #---------------------------------------------------------------------------
    # Begin optimization portion of function
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # iterative parameteres
    #---------------------------------------------------------------------------
    wmin = np.ones((n_dim, 1))*-10
    iter = 0                            # Current number of iterations
    tolZ = 1                            # Initialized error between Z and Z'
    Zold = np.zeros((len(Z), n_dim))        # Initialized Z array

    solvers.options['show_progress'] = False

    #---------------------------------------------------------------------------
    # Begin algorithm for outer hyperplane approxmation
    #---------------------------------------------------------------------------
    if display is True:
        print('Outer Hyperplane Approximation')

    while tolZ > tol and iter < MaxIter*n_dim:

        # Construct iteration
        Cla = np.zeros((L, A))          # The set of equilibrium payoffs
        Wla = np.zeros((A, n_dim, L))       # The set of equilibrium arguments

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
        # Measure convergence
        #----------------------------------------------------------------
        if Housedorff is True:
            tolZ = hausdorffnorm(Z, Zold)
        else:
            tolZ = np.max(np.max(np.abs(Z-Zold)/(1.+abs(Zold)), axis=0))


        if iter == MaxIter*n_dim:
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
    if n_dim ==2:
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

        Zold = vertices

    #-------------------------------------------------------------------------------
    # Plot final results and display
    #-------------------------------------------------------------------------------
    if iter < MaxIter and display is True:
        print('Convergence after %d iterations' % (iter))

        # evaluate and display elapsed time
        elapsed_time = time.time() - start_time
        print('Elapsed time is %f seconds' % (elapsed_time))


    return Zold, H


def innerbound_par(P, cen, rad, n_grad=24, delta=0.8, tol=1e-4, MaxIter=200, plot=True, display=True, Housedorff=False):
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
    #---------------------------------------------------------------------------
    # check inputs for correctness
    #---------------------------------------------------------------------------
    Payoff_shape = P[0].shape
    if sum([Payoff_shape==p.shape for p in P])!=len(P):
        raise Exception("payoff matrices must all be of the same size")

    elif sum(Payoff_shape[0] == p for p in Payoff_shape) !=len(Payoff_shape):
        raise Exception("payoff matrices must all be square")

    if n_grad < 4:
        raise Exception("insufficient number of search gradients")

    #---------------------------------------------------------------------------
    # parameters
    #---------------------------------------------------------------------------
    del1 = 1-delta
    n_dim = len(P)

    Pmax = [tuple(np.max(p,i).flatten()) for i, p in enumerate(P)]
    stagepay = np.vstack( (p.flatten() for p in P ) ).T
    BR = np.vstack( (p for p in product(*Pmax)) )[:,range(len(P)-1,-1,-1)]


    #---------------------------------------------------------------------------
    # gradients and tangency points
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # gradients and tangency points
    #---------------------------------------------------------------------------
    if n_dim == 2:
        H, Z = _make_circle(n_grad, cen, rad)
        print(H)
        # H, Z = _make_sphere(cen, rad, n_grad, n_dim)
    else:
        H, Z = _make_sphere(cen, rad, n_grad, n_dim)

    C = np.atleast_2d(np.sum(Z * H , axis=1))
    Z = np.array(Z)
    L = len(H)
    A = len(stagepay)
    C = C.T



    # convert needed arrays to matrix objects (for linear programming routine)
    H = cvx.matrix(np.array(H))
    G = cvx.matrix(np.vstack((H, -np.eye(n_dim))))


    #---------------------------------------------------------------------------
    # Begin optimization portion of function
    #---------------------------------------------------------------------------
    #-------------------------------------------------------------------------------
    # iterative parameteres
    #-------------------------------------------------------------------------------
    wmin = np.ones((n_dim, 1))*-10
    iter = 0                            # Current number of iterations
    tolZ = 1                            # Initialized error between Z and Z'
    Zold = np.zeros((len(Z), n_dim))        # Initialized Z array

    solvers.options['show_progress'] = False

    #-------------------------------------------------------------------------------
    # Begin algorithm for inner hyperplane approximation
    #-------------------------------------------------------------------------------
    if rank == 0 and display is True:
        print('Inner Hyperplane Approximation')

    while tolZ > tol and iter < MaxIter:

        # Construct iteration
        WlaCla_Buffer = np.zeros((A, n_dim+1, L))
        WlaCla_entry = np.zeros((A, n_dim+1, L))

        # loop through L search gradients
        for l in range(rank, L, size):
            # loop A possible actions
            for a in range(A):
                #----------------------------------------------------------------
                # Step 1.a:
                # solve Cla = max_w h_l*[del1*payoff(a) + delta * w   sub. to
                #       del1*payoff(a) + delta*w \geq del1* BR(a) + delta * wmin
                #----------------------------------------------------------------

                pay = np.atleast_2d(stagepay[a, :])
                b = cvx.matrix(np.vstack((delta*C+del1 * np.dot(H, pay.T), -del1 * np.atleast_2d(BR[a, :]).T - delta * wmin)))
                T = solvers.lp(-H[l, :].T, G, b)

                if T['status'] == 'optimal':
                    WlaCla_entry[a, 0:n_dim, l] = np.array(T['x'])[:, 0]
                    WlaCla_entry[a, n_dim, l] = -np.inner(-H[l, :], T['x'].T)
                else:
                    WlaCla_entry[a, n_dim,l] = -np.inf

        #----------------------------------------------------------------
        # gather all the pieces
        #----------------------------------------------------------------
        comm.Allreduce(WlaCla_entry, WlaCla_Buffer, op=MPI.SUM)
        Wla = WlaCla_Buffer[:, 0:n_dim, :]
        Cla = WlaCla_Buffer[:, n_dim, :].T

        #----------------------------------------------------------------
        # Step 1.b:
        # find best action profile 'a_*' such that
        #       a_* = argmax{Cla}                 --- element of I
        #       z = del1*payoff(a_*) + del*Wla_*  --- element of C
        #----------------------------------------------------------------
        I = np.atleast_2d(np.argmax(Cla, axis=1)).T
        C = np.atleast_2d(np.max(Cla, axis=0)).T; print(C)

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


def outerbound_par(P, cen, rad, n_grad=24, delta=0.8, tol=1e-4, MaxIter=200, plot=True, display=True, Housedorff=False):
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

    #---------------------------------------------------------------------------
    # check inputs for correctness
    #---------------------------------------------------------------------------
    Payoff_shape = P[0].shape
    if sum([Payoff_shape==p.shape for p in P])!=len(P):
        raise Exception("payoff matrices must all be of the same size")

    elif sum(Payoff_shape[0] == p for p in Payoff_shape) !=len(Payoff_shape):
        raise Exception("payoff matrices must all be square")

    if n_grad < 2:
        raise Exception("insufficient number of search gradients")

   #---------------------------------------------------------------------------
    # parameters
    #---------------------------------------------------------------------------
    del1 = 1-delta
    n_dim = len(P)

    Pmax = [tuple(np.max(p,i).flatten()) for i, p in enumerate(P)]
    stagepay = np.vstack( (p.flatten() for p in P ) ).T
    BR = np.vstack( (p for p in product(*Pmax)) )[:,range(len(P)-1,-1,-1)]


    #---------------------------------------------------------------------------
    # gradients and tangency points
    #---------------------------------------------------------------------------
    if n_dim == 2:
        H, Z = _make_circle(n_grad, cen, rad)
        print(H)
        # H, Z = _make_sphere(cen, rad, n_grad, n_dim)
    else:
        H, Z = _make_sphere(cen, rad, n_grad, n_dim)

    C = np.atleast_2d(np.sum(Z * H , axis=1))
    Z = np.array(Z)
    L = len(H)
    A = len(stagepay)
    C = C.T

    C = np.atleast_2d(np.sum(Z * H , axis=1))
    L = len(H)
    A = len(stagepay)
    C = C.T

    # convert needed arrays to matrix objects (for linear programming routine)
    H = cvx.matrix(np.array(H))
    G = cvx.matrix(np.vstack((H, -np.eye(n_dim))))

    #---------------------------------------------------------------------------
    # Begin optimization portion of function
    #---------------------------------------------------------------------------
    #-------------------------------------------------------------------------------
    # iterative parameteres
    #-------------------------------------------------------------------------------
    wmin = np.ones((2, 1))*-10
    iter = 0
    tolZ = 1
    Zold = np.zeros((len(Z), 2))
 
    solvers.options['show_progress'] = False

    #-------------------------------------------------------------------------------
    # Begin algorithm for outer hyperplane approximation
    #-------------------------------------------------------------------------------
    if rank == 0 and display is True:
        print('Outer Hyperplane Approximation')


    while tolZ > tol and iter < MaxIter:

        # Construct iteration
        WlaCla_Buffer = np.zeros((A, 3, L))
        WlaCla_entry = np.zeros((A, 3, L))

        # loop through L search gradients
        for l in range(rank, L, size):
         
            # loop A possible actions
            for a in range(A):
                #----------------------------------------------------------------
                # Step 1.a:
                # solve Cla = max_w h_l*[del1*payoff(a) + delta * w   sub. to
                #       del1*payoff(a) + delta*w \geq del1* BR(a) + delta * wmin
                #----------------------------------------------------------------
                pay = np.atleast_2d(stagepay[a, :])
                b = cvx.matrix(np.vstack((delta*C + del1*np.dot(H, pay.T), -del1*np.atleast_2d(BR[a, :]).T-delta*wmin)))
                T = solvers.lp(-H[l, :].T, G, b)

                if T['status'] == 'optimal':
                    WlaCla_entry[a, :, l] = np.array(T['x'])[:, 0]
                    WlaCla_entry[l, a] = -np.inner(-H[l, :], T['x'].T)
                else:
                    WlaCla_entry[l, a] = -np.inf

        #----------------------------------------------------------------
        # gather all the pieces
        #----------------------------------------------------------------
        comm.Allreduce(WlaCla_entry, WlaCla_Buffer, op=MPI.SUM)
        Wla = WlaCla_Buffer[:, 0:n_dim-2, :]
        Cla = WlaCla_Buffer[:, n_dim-1, :].T

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


    #-------------------------------------------------------------------------------
    # Find shape defined by supporting hyperplanes
    #-------------------------------------------------------------------------------
    if n_dim ==2:
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

        Zold = vertices

    if rank == 0:
    #-------------------------------------------------------------------------------
    # Plot final results and display
    #-------------------------------------------------------------------------------
        if iter < MaxIter and display is True:
            print('Convergence after %d iterations' % (iter))

            # evaluate and display elapsed time
            elapsed_time = time.time() - start_time
            print('Elapsed time is %f seconds' % (elapsed_time))


        return Zold, H


if __name__ == '__main__':
    p1 = np.array([[[1.46, 2.2 ], [1.5, 1.85]],  [[1.39, 2.1 ], [0.90, 1.54]]])
    p2 = np.array([[[1.46, 1.5 ], [2.2, 1.85]],  [[1.39, .90 ], [2.1, 1.54]]])
    p3 = np.array([[[1, 1],  [1, 2]],  [[2,  2], [2, 3]]])

    cen = np.array([3, 3, 3], ndmin=2)
    rad = 15
    Z_inner, H = innerbound([p1, p2, p3], cen, rad)
    Z_outer, H = outerbound([p1, p2, p3], cen, rad*10)
    make_3d_plots(Z_inner, save=False)    
ots(Z_inner, save=False)    
