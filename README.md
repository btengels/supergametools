
======================================================================
Supergametools - a python toolkit for repeated economic games
======================================================================
- Author: Benjamin Tengelsen <btengels@cmu.edu>
- Website: https://sites.google.com/site/btengelsenresearch
- Last revision: May 6, 2013


======================================================================
Introduction
======================================================================
Supergametools is a python library that contains functions for
approximating the set of equilibria within an infinitely repeated
game. The functions are based on algorithms presented in the paper
"Computing Supergame Equilibria" by Judd, Conklin, & Yeltekin (2003).


======================================================================
In this release
======================================================================
- Functions for inner and outer hyperplane approximation
- Parallelized optimization routines
- 2 player games only


======================================================================
Dependencies
======================================================================
Supergametools calls the following python libraries:

	- numpy -- <http://www.numpy.org>
	- scipy -- <http://www.scipy.org>
	- matplotlib -- <http://matplotlib.org>
	- cvxopt -- <http://cvxopt.org>
	- mpi4py -- <http://mpi4py.scipy.org>

The software is only tested for Python 2.7.


======================================================================
Installation
======================================================================
After downloading the source files (zipped folder from website), unzip
the files and change your directory to the source files. Then run (you
may need to become a root):

	python setup.py install

To see if the library installed correctly, open python (or ipython)
and type

	import supergametools

If the library imports without any errors, the software is properly
installed. Assuming all dependencies are installed, you should be able
to run any of the test scripts accompanying the source code.


======================================================================
To Use
======================================================================
Each function in the supergametools library has both a serial and
a parallel version. The parallel functions have the same names as
their parallel counterparts, but with the term "_par" on the end.

The serial functions can be used interactively in python (or ipython)
with no special instruction. This is a good way to initially code a
program and/or check for correctness. Specific information for each
function is given in its docstring.

The parallel functions have not been tested for interactive use. If
your script calls the parallelized functions, you can run your script
using the following command-line syntax (from the files directory):

	mpiexec -n [# of processes] python yourfile.py

Note that each process will run the entire script, but will work
together to evaluate the supergametools functions. The following site
may be helpful for those who wish to use the parallelized
supergametools functions in an interactive setting:
http://ipython.org/ipython-doc/dev/parallel/


======================================================================
Items for future releases
======================================================================
- 3+ player games
- improved timing of parallelized functions


======================================================================
Acknowledgements
======================================================================
Special thanks to Sevin Yeltekin, Chase Coleman, and Spencer Lyon for
user feedback and helpful suggestions.

