
# Supergametools - a python toolkit for repeated economic games
- Author: Benjamin Tengelsen

## Introduction

Supergametools is a python library that contains functions for
approximating the set of equilibria within an infinitely repeated
game. The functions are based on algorithms presented in the paper
"Computing Supergame Equilibria" by Judd, Conklin, & Yeltekin (2003).


## In this release

- Functions for inner and outer hyperplane approximation
- Parallelized optimization routines
- 2 player games only


## Dependencies

Supergametools calls the following python libraries:

  - numpy -- <http://www.numpy.org>
  - scipy -- <http://www.scipy.org>
  - matplotlib -- <http://matplotlib.org>
  - cvxopt -- <http://cvxopt.org>
  - mpi4py -- <http://mpi4py.scipy.org> (required only for parallel functions)

Requires Python 3.8 or later.


## Installation

Install directly from the source directory:

	pip install .

To install with parallel support (mpi4py):

	pip install ".[parallel]"

To verify the installation, open python and type:

	import supergametools

If the library imports without any errors, the software is properly
installed. Assuming all dependencies are installed, you should be able
to run any of the example scripts in the `docs/` directory.


## To Use
Each function in the supergametools library has both a serial and
a parallel version. The parallel functions have the same names as
their serial counterparts, but with the term `_par` appended.

The serial functions can be used interactively in python (or ipython)
with no special instruction. This is a good way to initially code a
program and/or check for correctness. Specific information for each
function is given in its docstring.

The parallel functions require mpi4py and are run via:

	mpiexec -n [# of processes] python yourfile.py


## Items for future releases
- 3+ player games
- Improved timing of parallelized functions


## Acknowledgements
Special thanks to Sevin Yeltekin, Chase Coleman, and Spencer Lyon for
user feedback and helpful suggestions.
