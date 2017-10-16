DESPOT
------

DESPOT is a C++ implementation of the DESPOT algorithm [1]. It takes as input
a POMDP model (coded in C++) and simulates an agent acting in the environment.

For bug reports and suggestions, please email <adhirajsomani@gmail.com>

[1] Online POMDP Planning with Regularization.


============
REQUIREMENTS
============

Operating systems:        Linux
                          Mac OS X

Tested compilers:         gcc/g++ 4.8.1 under Linux
                          gcc/g++ 4.6.3 under Linux
                          clang++ 4.2   under Mac OS X Lion

* General
  - GNU make is required for building.
  - The code uses features of the C++11 standard.


==========
QUICKSTART
==========

After unzipping the package, cd into the extracted directory and run `make`
(see Makefile for compiler options). A single executable will be generated at
`bin/despot`.

Parameters to the program are specified as command line arguments (see section
below). 6 sample problems are included in the package: Tag, LaserTag, 
RockSample, Tiger, Bridge and Pocman.

At each timestep the algorithm outputs the action, observation, reward and 
current state of the world. Additional information like the lower/upper bounds
before/after the search go to standard error and can be suppressed. The total
discounted reward is output at the end of the run.

Examples:

./bin/despot -q tag -n 250 -d 80 -m src/problems/tag/model-params
  Run tag with 250 particles and search depth 80

./bin/despot -q rocksample -p 0.1 -b particle -m src/problems/rocksample/model-params
  Run rocksample with pruning constant 0.1 and particle filter belief-update

./bin/despot -q tiger -r 33 -l 100 -m src/problems/tiger/model-params
  Run tiger with random-number seed 33 for upto 100 steps

./bin/despot -q pocman -g 1 -d 60 -t 5
  Run pocman with undiscounted return, search depth 60, and 5 seconds per move

The following parameters give good results for the sample problems, and were
used in the results reported in the paper.

+------------+------------+------------------+-------+---------+
| Problem    | Particles | Pruning Constant | Depth | Discount |
+------------+------------+------------------+-------+---------+
| Tag        | 500       | 0.01             | 90    | 0.95     |
| LaserTag   | 500       | 0.01             | 90    | 0.95     |
| RockSample | 500       | 0.1              | 90    | 0.95     |
| Tiger      | 50        | 0                | 90    | 0.95     |
| Bridge     | 100       | 0                | 90    | 0.95     |
| Pocman     | 500       | x                | 60    | 1        |
+------------+------------+-----------------+-------+----------+
  

======================
COMMAND-LINE ARGUMENTS
======================

--help                 Print usage and exit.

-q <arg>               Problem name.
--problem=<arg>

-m <arg>               Path to model-parameters file, if any.
--model-params=<arg>

-d <arg>               Maximum depth of search tree (default 90).
--depth=<arg>

-g <arg>               Discount factor (default 0.95).
--discount=<arg>

-r <arg>               Random number seed (default 42).
--seed=<arg>

-t <arg>               Search time per move, in seconds (default 1).
--timeout=<arg>

-n <arg>               Number of particles (default 500).
--nparticles=<arg>

-p <arg>               Pruning constant (default no pruning).
--prune=<arg>

-s <arg>               Number of steps to simulate (default 0 = infinite).
--simlen=<arg>

-l <arg>               Lower bound strategy, if applicable. Can be either
--lbtype=<arg>         "mode" or "random". See src/Lower_bound/lower_bound_policy_mode.h
                       and src/lower_bound/lower_bound_policy_random.h for
                       more details.

-b <arg>               Belief-update strategy, if applicable. Can be either
--belief=<arg>         "particle" or "exact". See src/belief_update/belief_update_particle.h
                       and src/belief_update/belief_update_exact.h for more
                       details.

-k <arg>               Knowledge level for random lower bound policy, if
--knowledge=<arg>      applicable. Level 1 generates legal actions, and level
                       2 generates preferred actions. See src/lower_bound/lower_bound_policy_random.h
                       for more details.

-a                     Whether initial lower/upper bounds are approximate or
--approx-bounds        true (default false). If approximate, the solver allows
                       the case where an initial upper bound may be smaller
                       than the lower bound, bumping it up to the lower bound.
                       This may be the case in complex problems like Pocman
                       where a true upper that is also useful is hard to
                       compute.


====================
WRITING NEW PROBLEMS
====================

1. Read the Overview section of `despot/doc/Design.txt`, and the template
   for a problem specification in `despot/src/model.h`.
2. Specialize the template for a new problem (see the implementation of Tag 
   for an example).
3. Modify main.cpp to recognize the new problem.
4. Compile and run.


================
ACKNOWLEDGEMENTS
================

The implementation of Pocman is based on David Silver's POMCP implementation,
which can be found at http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Applications.html
